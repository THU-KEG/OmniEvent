import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union

from transformers import BertModel, BertTokenizerFast
from transformers import RobertaModel, RobertaTokenizerFast
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import MT5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers.utils import ModelOutput

from ..input_engineering.whitespace_tokenizer import WordLevelTokenizer, load_vocab, VOCAB_FILES_NAMES
from ..arguments import ModelArguments

def get_backbone(model_type: str,
                 model_name_or_path: str,
                 tokenizer_name: str,
                 markers: List[str],
                 model_args: Optional[ModelArguments] = None,
                 new_tokens: Optional[List[str]] = []):
    """Obtains the backbone model and tokenizer.

    Obtains the backbone model and tokenizer. The backbone model is selected from BERT, RoBERTa, T5, MT5, CNN, and LSTM,
    corresponding to a distinct tokenizer.

    Args:
        model_type (`str`):
            A string indicating the model being used as the backbone network.
        model_name_or_path (`str`):
            A string indicating the path of the pre-trained model.
        tokenizer_name (`str`):
            A string indicating the repository name for the model in the hub or a path to a local folder.
        markers (`List[str]`):
            A list of strings to mark the start and end position of event triggers and argument mentions.
        model_args (`optional`, defaults to `None`):
            The pre-defined arguments for the model.  TODO: The data type of `model_args` should be configured.
        new_tokens (`List[str]`, `optional`, defaults to []):
            A list of strings indicating new tokens to be added to the tokenizer's vocabulary.

    Returns:
        model (`Union[BertModel, RobertaModel, T5ForConditionalGeneration, CNN, LSTM]`):
            The backbone model, which is selected from BERT, RoBERTa, T5, MT5, CNN, and LSTM.
        tokenizer (`str`):
            The tokenizer proposed for the tokenization process, corresponds to the backbone model.
        config:
            The configurations of the model.         TODO: The data type of `config` should be configured.
    """
    if model_type == "bert":
        model = BertModel.from_pretrained(model_name_or_path)
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, never_split=markers)
    elif model_type == "roberta":
        model = RobertaModel.from_pretrained(model_name_or_path)
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name, never_split=markers, add_prefix_space=True)
    elif model_type == "bart":
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = BartTokenizerFast.from_pretrained(tokenizer_name, never_split=markers, add_prefix_space=True)
    elif model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name, never_split=markers)
    elif model_type == "mt5":
        model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name, never_split=markers)
    elif model_type == "cnn":
        tokenizer = WordLevelTokenizer.from_pretrained(model_args.vocab_file)
        model = CNN(model_args, len(tokenizer))
    elif model_type == 'lstm':
        tokenizer = WordLevelTokenizer.from_pretrained(model_args.vocab_file)
        model = LSTM(model_args, len(tokenizer))
    else:
        raise ValueError("No such model. %s" % model_type)

    for token in new_tokens:
        tokenizer.add_tokens(token, special_tokens=True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    config = model.config
    return model, tokenizer, config


class WordEmbedding(nn.Module):
    """Base class for word embedding.

    Base class for word embedding, in which the word embeddings are loaded from a pre-trained word embedding file and
    could be resized into a distinct size.

    Attributes:
        word_embeddings (`torch.Tensor`):
            A tensor representing the word embedding matrix, whose dimension is (number of tokens) * (embedding
            dimension).
        position_embeddings (`torch.Tensor`):
            A tensor representing the position embedding matrix, whose dimension is (number of positions) * (embedding
            dimension).
        dropout (`nn.Dropout`):
            An `nn.Dropout` layer for the dropout operation with the pre-defined dropout rate.
    """
    def __init__(self,
                 config,
                 vocab_size: int) -> None:
        """Constructs a `WordEmbedding`."""
        super(WordEmbedding, self).__init__()
        if not os.path.exists(os.path.join(config.vocab_file, VOCAB_FILES_NAMES["vocab_file"].replace("txt", "npy"))):
            embeddings = load_vocab(os.path.join(config.vocab_file, VOCAB_FILES_NAMES["vocab_file"]),
                                    return_embeddings=True)
            np.save(os.path.join(config.vocab_file, VOCAB_FILES_NAMES["vocab_file"].replace("txt", "npy")), embeddings)
        else:
            embeddings = np.load(os.path.join(config.vocab_file, VOCAB_FILES_NAMES["vocab_file"].replace("txt", "npy")))
        self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=False, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.num_position_embeddings, config.position_embedding_dim)
        self.register_buffer("position_ids", torch.arange(config.num_position_embeddings).expand((1, -1)))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resize_token_embeddings(vocab_size)
        self.config = config
        # event type embeddings
        if config.has_type_embeddings:
            self.type_embeddings = nn.Embedding(config.num_types, config.type_embedding_dim)

    def resize_token_embeddings(self,
                                vocab_size: int) -> None:
        """Resizes the embeddings from the pre-trained embedding dimension to pre-defined embedding size."""
        if len(self.word_embeddings.weight) > vocab_size:
            raise ValueError("Invalid vocab_size %d < original vocab size." % vocab_size)
        elif len(self.word_embeddings.weight) == vocab_size:
            pass
        else:
            num_added_token = vocab_size - len(self.word_embeddings.weight)
            embedding_dim = self.word_embeddings.weight.shape[1]
            average_embedding = torch.mean(self.word_embeddings.weight, dim=0).expand(1, -1)
            self.word_embeddings.weight = nn.Parameter(torch.cat(
                (
                    self.word_embeddings.weight.data,
                    average_embedding.expand(num_added_token, embedding_dim)
                )
            ))

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                position: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generates word embeddings and position embeddings and concatenates them together."""
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape[0], input_shape[1]
        position_ids = self.position_ids[:, :seq_length].expand(batch_size, seq_length)
        if position is not None:
            position_ids = torch.abs(position_ids - position.unsqueeze(1)).to(torch.long)
        # input embeddings & position embeddings 
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeds = torch.cat((inputs_embeds, position_embeds), dim=-1)
        if token_type_ids is not None and self.config.has_type_embeddings:
            embeds = torch.cat((embeds, self.type_embeddings(token_type_ids)), dim=-1)
        if self.config.dropout_after_wordvec:
            embeds = self.dropout(embeds)
        return embeds


class Output(ModelOutput):
    """A class for the model's output, containing the hidden states of the sequence."""
    last_hidden_state: torch.Tensor = None


class CNN(nn.Module):
    """A Convolutional Neural Network (CNN) as backbone model.

    A Convolutional Neural Network (CNN) as the backbone model, which comprises a 1-d convolutional layer, a relu
    activation layer, and a dropout layer. The last hidden state of the model would be returned.

    Attributes:
        config:
            The configurations of the model.
        embedding (`WordEmbedding`):
            A `WordEmbedding` instance representing the embedding matrices of tokens and positions.
        conv (`nn.Conv1d`):
            A `nn.Conv1d` layer representing 1-dimensional convolution layer.
        dropout (`nn.Dropout`):
            An `nn.Dropout` layer for the dropout operation with the pre-defined dropout rate.
    """
    def __init__(self,
                 config,
                 vocab_size: int,
                 kernel_size: Optional[int] = 3,
                 padding_size: Optional[int] = 1) -> None:
        """Constructs a `CNN`."""
        super(CNN, self).__init__()
        self.config = config
        self.embedding = WordEmbedding(config, vocab_size)
        in_channels = config.word_embedding_dim + config.position_embedding_dim + config.type_embedding_dim if config.has_type_embeddings else \
                        config.word_embedding_dim + config.position_embedding_dim
        self.conv = nn.Conv1d(in_channels,
                              config.hidden_size,
                              kernel_size,
                              padding=padding_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def resize_token_embeddings(self,
                                vocab_size: int) -> None:
        """Resizes the embeddings from the pre-trained embedding dimension to pre-defined embedding size."""
        self.embedding.resize_token_embeddings(vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                position: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = True) -> Union[Output, Tuple[torch.Tensor]]:
        """Conducts the convolution operations on the input tokens."""
        x = self.embedding(input_ids, token_type_ids, position)  # (B, L, H)
        x = x.transpose(1, 2)  # (B, H, L)
        x = F.relu(self.conv(x).transpose(1, 2))  # (B, H, L)
        # x = self.dropout(x)
        if return_dict:
            return Output(last_hidden_state=x)
        else:
            return x


class LSTM(nn.Module):
    """A Long Short-Term Memory (LSTM) network as backbone model.

    A bidirectional two-layered Long Short-Term Memory (LSTM) network as the backbone model, which utilizes recurrent
    computations for hidden states and addresses long-term information preservation and short-term input skipping
    using gated memory cells.

    Attributes:
        config:
            The configurations of the model.
        embedding (`WordEmbedding`):
            A `WordEmbedding` instance representing the embedding matrices of tokens and positions.
        rnn (`nn.LSTM`):
            A `nn.LSTM` layer representing a bi-directional two-layered LSTM network, which manipulates the word
            embedding and position embedding for recurrent computations.
        dropout (`nn.Dropout`):
            An `nn.Dropout` layer for the dropout operation with the pre-defined dropout rate.
       """
    def __init__(self,
                 config,
                 vocab_size: int) -> None:
        """Constructs a `LSTM`."""
        super(LSTM, self).__init__()
        self.config = config
        self.embedding = WordEmbedding(config, vocab_size)
        self.rnn = nn.LSTM(config.word_embedding_dim + config.position_embedding_dim,
                           config.hidden_size,
                           num_layers=2,
                           bidirectional=True,
                           batch_first=True,
                           dropout=config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def resize_token_embeddings(self,
                                vocab_size: int) -> None:
        """Resizes the embeddings from the pre-trained embedding dimension to pre-defined embedding size."""
        self.embedding.resize_token_embeddings(vocab_size)

    def prepare_pack_padded_sequence(self,
                                     input_ids: torch.Tensor,
                                     input_lengths: torch.Tensor,
                                     descending: Optional[bool] = True):
        """Sorts the input sequences based on their length."""
        sorted_input_lengths, indices = torch.sort(input_lengths, descending=descending)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_input_ids = input_ids[indices]
        return sorted_input_ids, sorted_input_lengths, desorted_indices

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                position: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = True):
        """Forward propagation of a LSTM network."""
        # add a pseudo input of max_length
        add_pseudo = max(torch.sum(attention_mask, dim=-1).tolist()) != input_ids.shape[1]
        if add_pseudo:
            input_ids = torch.cat((torch.zeros_like(input_ids[0]).unsqueeze(0), input_ids), dim=0)
            attention_mask = torch.cat((torch.ones_like(attention_mask[0]).unsqueeze(0), attention_mask), dim=0)
        input_length = torch.sum(attention_mask, dim=-1).to(torch.long)
        sorted_input_ids, sorted_seq_length, desorted_indices = self.prepare_pack_padded_sequence(input_ids,
                                                                                                  input_length)
        x = self.embedding(sorted_input_ids, position)  # (B, L, H)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(x, sorted_seq_length.cpu(), batch_first=True)
        self.rnn.flatten_parameters()
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        x = output[desorted_indices]
        if add_pseudo:
            x = self.dropout(x)[1:, :, :]  # remove the pseudo input
        else:
            x = self.dropout(x)

        if return_dict:
            return Output(
                last_hidden_state=x
            )
        else:
            return (x)
