import os 
import torch 
import torch.nn as nn 

from typing import Tuple, Dict, Optional, Union

from transformers import BartForConditionalGeneration, MT5ForConditionalGeneration, T5ForConditionalGeneration

from OpenEE.aggregation.aggregation import get_aggregation, aggregate
from OpenEE.head.head import get_head
from OpenEE.head.classification import LinearHead
from OpenEE.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    ArgumentParser
)
from OpenEE.utils import check_web_and_convert_path


def get_model(model_args, backbone):
    if model_args.paradigm == "token_classification":
        return ModelForTokenClassification(model_args, backbone)
    elif model_args.paradigm == "sequence_labeling":
        return ModelForSequenceLabeling(model_args, backbone)
    elif model_args.paradigm == "seq2seq":
        return backbone
    elif model_args.paradigm == "mrc":
        return ModelForMRC(model_args, backbone)
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
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike], config=None, **kwargs):
        if config is None:
            parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
            model_args, _, _ = parser.from_pretrained(model_name_or_path, **kwargs)
        path = check_web_and_convert_path(model_name_or_path, 'model')
        model = get_model(model_args)
        model.load_state_dict(torch.load(path), strict=False)
        return model


class ModelForTokenClassification(BaseModel):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForTokenClassification, self).__init__()
        self.config = config
        self.backbone = backbone 
        self.aggregation = get_aggregation(config)
        self.cls_head = get_head(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        trigger_left: Optional[torch.Tensor] = None, 
        trigger_right: Optional[torch.Tensor] = None, 
        argument_left: Optional[torch.Tensor] = None, 
        argument_right: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=attention_mask, \
                                token_type_ids=token_type_ids,
                                return_dict=True)   
        hidden_states = outputs.last_hidden_state
        # aggregation 
        hidden_state = aggregate(self.config, 
                                 self.aggregation, 
                                 hidden_states, 
                                 trigger_left,
                                 trigger_right,
                                 argument_left,
                                 argument_right)
        # classification
        logits = self.cls_head(hidden_state)
        # compute loss 
        loss = None 
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return dict(loss=loss, logits=logits)
    

class ModelForSequenceLabeling(BaseModel):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForSequenceLabeling, self).__init__()
        self.config = config 
        self.backbone = backbone 
        self.cls_head = LinearHead(config)
        self.head = get_head(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=attention_mask, \
                                token_type_ids=token_type_ids,
                                return_dict=True)   
        hidden_states = outputs.last_hidden_state
        # classification
        logits = self.cls_head(hidden_states) # [batch_size, seq_length, num_labels]
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
                labels = labels * mask.to(torch.long)
                loss = -self.crf(emissions=logits, 
                                tags=labels,
                                mask=mask,
                                reduction = "token_mean")
        else:
            if self.config.head_type == "crf":
                preds = self.crf.decode(emissions=logits, mask=mask)
                logits = torch.LongTensor(preds)

        return dict(loss=loss, logits=logits)


class ModelForMRC(BaseModel):
    """Model for machine reading comprehension"""

    def __init__(self, config, backbone) -> None:
        super(ModelForMRC, self).__init__()
        self.backbone = backbone
        self.mrc_head = get_head(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None,
        argument_left: Optional[torch.Tensor] = None,
        argument_right: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=attention_mask, \
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
        logits = torch.cat((start_logits, end_logits), dim=-1) # [batch_size, seq_length*2]
        
        return dict(loss=total_loss, logits=logits)

