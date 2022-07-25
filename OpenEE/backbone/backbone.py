from audioop import bias
import os 
import pdb
from unicodedata import bidirectional
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from transformers import BertModel, BertTokenizerFast 
from transformers import RobertaModel, RobertaTokenizerFast
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import MT5ForConditionalGeneration
from transformers.utils import ModelOutput 

from ..input_engineering.tokenizer import WordLevelTokenizer, load_vocab, VOCAB_FILES_NAMES


def get_backbone(model_type, model_name_or_path, tokenizer_name, markers, 
                model_args=None,
                new_tokens:list = []
    ):
    if model_type == "bert":
        model = BertModel.from_pretrained(model_name_or_path)
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, never_split=markers)
    elif model_type == "roberta":
        model = RobertaModel.from_pretrained(model_name_or_path)
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name, never_split=markers)
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
        tokenizer.add_tokens(token, special_tokens = True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
        # word_embeddings = model.get_input_embeddings()
        # word_embeddings.weight.data[-len(new_tokens):, :] = torch.zeros((len(new_tokens), model.config.hidden_size))

    config = model.config
    return model, tokenizer, config


class WordEmbedding(nn.Module):
    def __init__(self, config, vocab_size) -> None:
        super(WordEmbedding, self).__init__()
        if not os.path.exists(os.path.join(config.vocab_file, VOCAB_FILES_NAMES["vocab_file"].replace("txt", "npy"))):
            embeddings = load_vocab(os.path.join(config.vocab_file, VOCAB_FILES_NAMES["vocab_file"]), return_embeddings=True)
            np.save(os.path.join(config.vocab_file, VOCAB_FILES_NAMES["vocab_file"].replace("txt", "npy")), embeddings)
        else:
            embeddings = np.load(os.path.join(config.vocab_file, VOCAB_FILES_NAMES["vocab_file"].replace("txt", "npy")))
        self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=False, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.num_position_embeddings, config.position_embedding_dim)
        self.register_buffer("position_ids", torch.arange(config.num_position_embeddings).expand((1, -1)))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resize_token_embeddings(vocab_size)

    def resize_token_embeddings(self, vocab_size):
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
                input_ids, 
                position_ids=None
        ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape[0], input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length].expand(batch_size, seq_length)
        # input embeddings & position embeddings 
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeds = torch.cat((inputs_embeds, position_embeds), dim=-1)    
        embeds = self.dropout(embeds)
        return embeds


class Output(ModelOutput):
    last_hidden_state: torch.Tensor = None 


class CNN(nn.Module):
    def __init__(self, 
                 config, 
                 vocab_size,
                 kernel_size=3, 
                 padding_size=1
        ) -> None:
        super(CNN, self).__init__()
        self.config = config
        self.embedding = WordEmbedding(config, vocab_size)
        self.conv = nn.Conv1d(config.word_embedding_dim+config.position_embedding_dim, 
                              config.hidden_size, 
                              kernel_size, 
                              padding=padding_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def resize_token_embeddings(self, vocab_size):
        self.embedding.resize_token_embeddings(vocab_size)

    def forward(self, 
                input_ids,
                attention_mask,
                token_type_ids,
                return_dict=True
        ):
        x = self.embedding(input_ids) # (B, L, H)
        x = x.transpose(1, 2)          # (B, H, L)
        x = F.relu(self.conv(x).transpose(1, 2))       # (B, H, L)
        x = self.dropout(x)
        if return_dict:
            return Output(
                    last_hidden_state = x
                )
        else:
            return (x) 

class LSTM(nn.Module):
    def __init__(self, 
                 config, 
                 vocab_size
        ) -> None:
        super(LSTM, self).__init__()
        self.config = config
        self.embedding = WordEmbedding(config, vocab_size)
        self.rnn = nn.LSTM(config.word_embedding_dim+config.position_embedding_dim, 
                            config.hidden_size, 
                            num_layers=2, 
                            bidirectional=True, 
                            batch_first=True, 
                            dropout=config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def resize_token_embeddings(self, vocab_size):
        self.embedding.resize_token_embeddings(vocab_size)


    def prepare_pack_padded_sequence(self, input_ids, input_lengths, descending=True):
        sorted_input_lengths, indices = torch.sort(input_lengths, descending=descending)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_input_ids = input_ids[indices]
        return sorted_input_ids, sorted_input_lengths, desorted_indices


    def forward(self, 
                input_ids,
                attention_mask,
                token_type_ids,
                return_dict=True
        ):
        input_length = torch.sum(attention_mask, dim=-1).to(torch.long)
        sorted_input_ids, sorted_seq_length, desorted_indices = self.prepare_pack_padded_sequence(input_ids, input_length)
        x = self.embedding(sorted_input_ids) # (B, L, H)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(x, sorted_seq_length.cpu(), batch_first=True)
        self.rnn.flatten_parameters()
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        x = output[desorted_indices]
        x = self.dropout(x)
        # pdb.set_trace()
        if return_dict:
            return Output(
                    last_hidden_state = x
                )
        else:
            return (x) 