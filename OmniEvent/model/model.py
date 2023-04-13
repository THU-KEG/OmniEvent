import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Union
from transformers import BartForConditionalGeneration, MT5ForConditionalGeneration, T5ForConditionalGeneration

from OmniEvent.aggregation.aggregation import get_aggregation, aggregate
from OmniEvent.head.head import get_head
from OmniEvent.head.classification import LinearHead
from OmniEvent.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    ArgumentParser
)
from OmniEvent.utils import check_web_and_convert_path


def get_model(model_args,
              backbone,
              task_type="EAE"):
    """Returns the model proposed to be utilized for training and prediction.

    Returns the model proposed to be utilized for training and prediction based on the pre-defined paradigm. The
    paradigms of training and prediction include token classification, sequence labeling, Sequence-to-Sequence
    (Seq2Seq), and Machine Reading Comprehension (MRC).

    Args:
        model_args:
            The arguments of the model for training and prediction.
        backbone:
            The backbone model obtained from the `get_backbone()` method.

    Returns:
        The model method/class proposed to be utilized for training and prediction.
    """
    if model_args.paradigm == "token_classification":
        return ModelForTokenClassification(model_args, backbone)
    elif model_args.paradigm == "sequence_labeling":
        return ModelForSequenceLabeling(model_args, backbone)
    elif model_args.paradigm == "seq2seq":
        return backbone
    elif model_args.paradigm == "mrc":
        if task_type == "EAE":
            return ModelForMRC(model_args, backbone)
        elif task_type == "ED":
            return ModelForSequenceLabeling(model_args, backbone)
        else:
            raise ValueError
    else:
        raise ValueError("No such paradigm")


def get_model_cls(model_args):
    if model_args.paradigm == "token_classification":
        return ModelForTokenClassification
    elif model_args.paradigm == "sequence_labeling":
        return ModelForSequenceLabeling
    elif model_args.paradigm == "seq2seq":
        if model_args.model_type == "bart":
            return BartForConditionalGeneration
        elif model_args.model_type == "t5":
            return T5ForConditionalGeneration
        elif model_args.model_type == "mt5":
            return MT5ForConditionalGeneration
        else:
            raise ValueError("Invalid model_type %s" % model_args.model_type)
    elif model_args.paradigm == "mrc":
        return ModelForMRC
    else:
        raise ValueError("No such paradigm")


class BaseModel(nn.Module):

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike], backbone=None, model_args=None, **kwargs):
        if model_args is None:
            parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
            model_args, _, _ = parser.from_pretrained(model_name_or_path, **kwargs)
        path = check_web_and_convert_path(model_name_or_path, 'model')
        model = get_model(model_args, backbone)
        model.load_state_dict(torch.load(path), strict=False)
        return model


class ModelForTokenClassification(BaseModel):
    """BERT model for token classification.

    BERT model for token classification, which firstly obtains hidden states through the backbone model, then aggregates
    the hidden states through the aggregation method/class, and finally classifies each token to their corresponding
    label through token-wise linear transformation.

    Attributes:
        config:
            The configurations of the model.
        backbone:
            The backbone network obtained from the `get_backbone()` method to output initial hidden states.
        aggregation:
            The aggregation method/class for aggregating the hidden states output by the backbone network.
        cls_head (`ClassificationHead`):
            A `ClassificationHead` instance classifying each token into its corresponding label through a token-wise
            linear transformation.
    """

    def __init__(self,
                 config,
                 backbone) -> None:
        """Constructs a `ModelForTokenClassification`."""
        super(ModelForTokenClassification, self).__init__()
        self.config = config
        self.backbone = backbone
        self.aggregation = get_aggregation(config)
        self.cls_head = get_head(config)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                trigger_left: Optional[torch.Tensor] = None,
                trigger_right: Optional[torch.Tensor] = None,
                argument_left: Optional[torch.Tensor] = None,
                argument_right: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Manipulates the inputs through a backbone, aggregation, and classification module,
           returns the predicted logits and loss."""
        # backbone encode
        if self.config.model_type in ["cnn", "lstm"]:
            outputs = self.backbone(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position=trigger_left if argument_left is None else argument_left,
                        return_dict=True)
        else:
            outputs = self.backbone(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    return_dict=True)
        hidden_states = outputs.last_hidden_state
        # aggregation 
        hidden_state = aggregate(self.config,
                                 self.aggregation,
                                 hidden_states,
                                 attention_mask,
                                 trigger_left,
                                 trigger_right,
                                 argument_left,
                                 argument_right,
                                 embeddings=self.backbone.embedding.word_embeddings(input_ids) if self.config.model_type=="cnn" else None)
        # classification
        logits = self.cls_head(hidden_state)
        # compute loss 
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return dict(loss=loss, logits=logits)


class ModelForSequenceLabeling(BaseModel):
    """BERT model for sequence labeling.

    BERT model for sequence labeling, which firstly obtains hidden states through the backbone model, then labels each
    token to their corresponding label, and finally decodes the label through a Conditional Random Field (CRF) module.

    Attributes:
        config:
            The configurations of the model.
        backbone:
            The backbone network obtained from the `get_backbone()` method to output initial hidden states.
        cls_head (`ClassificationHead`):
            A `ClassificationHead` instance classifying each token into its corresponding label through a token-wise
            linear transformation.
    """

    def __init__(self,
                 config,
                 backbone) -> None:
        """Constructs a `ModelForSequenceLabeling`."""
        super(ModelForSequenceLabeling, self).__init__()
        self.config = config
        self.backbone = backbone
        self.cls_head = LinearHead(config)
        self.head = get_head(config)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Manipulates the inputs through a backbone, classification, and CRF module,
           returns the predicted logits and loss."""
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True)
        hidden_states = outputs.last_hidden_state
        if self.config.dropout_after_encoder:
            hidden_states = F.dropout(hidden_states, p=0.2)
        # classification
        logits = self.cls_head(hidden_states)  # [batch_size, seq_length, num_labels]
        # compute loss 
        loss = None
        if labels is not None:
            if self.config.head_type != "crf":
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            else:
                # CRF
                labels[:, 0] = 0
                mask = labels != -100
                tags = labels * mask.to(torch.long)
                loss = -self.head(emissions=logits,
                                  tags=tags,
                                  mask=mask,
                                  reduction="token_mean")
                labels[:, 0] = -100
        else:
            if self.config.head_type == "crf":
                mask = torch.ones_like(logits[:, :, 0])
                preds = self.head.decode(emissions=logits, mask=mask)
                logits = torch.LongTensor(preds)

        return dict(loss=loss, logits=logits)


class ModelForMRC(BaseModel):
    """BERT Model for Machine Reading Comprehension (MRC).

    BERT model for Machine Reading Comprehension (MRC), which firstly obtains hidden states through the backbone model,
    then predicts the start and end logits of each mention type through an MRC head.

    Attributes:
        config:
            The configurations of the model.
        backbone:
            The backbone network obtained from the `get_backbone()` method to output initial hidden states.
        mrc_head (`MRCHead`):
            A `ClassificationHead` instance classifying the hidden states into start and end logits of each mention type
            through token-wise linear transformations.
    """

    def __init__(self,
                 config,
                 backbone) -> None:
        """Constructs a `ModelForMRC`."""
        super(ModelForMRC, self).__init__()
        self.backbone = backbone
        self.mrc_head = get_head(config)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                argument_left: Optional[torch.Tensor] = None,
                argument_right: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Manipulates the inputs through a backbone and a MRC head module,
           returns the predicted start and logits and loss."""
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True)
        hidden_states = outputs.last_hidden_state
        start_logits, end_logits = self.mrc_head(hidden_states)
        total_loss = None
        # pdb.set_trace()
        if argument_left is not None and argument_right is not None:
            # If we are on multi-GPU, split add a dimension
            if len(argument_left.size()) > 1:
                argument_left = argument_left.squeeze(-1)
            if len(argument_right.size()) > 1:
                argument_right = argument_right.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            argument_left = argument_left.clamp(0, ignored_index)
            argument_right = argument_right.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, argument_left)
            end_loss = loss_fct(end_logits, argument_right)
            total_loss = (start_loss + end_loss) / 2

        logits = torch.cat((start_logits, end_logits), dim=-1)  # [batch_size, seq_length*2]
        return dict(loss=total_loss, logits=logits)
