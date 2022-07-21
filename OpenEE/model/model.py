import pdb
from numpy import dtype
import torch 
import torch.nn as nn 

from typing import Tuple, Dict, Optional

from OpenEE.aggregation.aggregation import get_aggregation, aggregate
from OpenEE.head.classification import (
    ClassificationHead,
    MRCHead
)
from OpenEE.head.crf import CRF


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


class ModelForTokenClassification(nn.Module):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForTokenClassification, self).__init__()
        self.config = config
        self.backbone = backbone 
        self.aggregation = get_aggregation(config)
        self.cls_head = ClassificationHead(config)

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
    

class ModelForSequenceLabeling(nn.Module):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForSequenceLabeling, self).__init__()
        self.backbone = backbone 
        self.crf = CRF(config.num_labels, batch_first=True)
        self.cls_head = ClassificationHead(config)

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
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            # CRF
            # mask = labels != -100
            # labels = labels * mask.to(torch.long)
            # loss = -self.crf(emissions=logits, 
            #                 tags=labels,
            #                 mask=mask,
            #                 reduction = "token_mean")
        else:
            # preds = self.crf.decode(emissions=logits, mask=mask)
            # logits = torch.LongTensor(preds)
            pass 

        return dict(loss=loss, logits=logits)


class ModelForMRC(nn.Module):
    """Model for machine reading comprehension"""

    def __init__(self, config, backbone) -> None:
        super(ModelForMRC, self).__init__()
        self.backbone = backbone
        self.mrc_head = MRCHead(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
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
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        logits = torch.cat((start_logits, end_logits), dim=-1) # [batch_size, seq_length*2]
        
        return dict(loss=total_loss, logits=logits)

