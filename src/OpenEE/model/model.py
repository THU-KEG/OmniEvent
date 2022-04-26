import pdb
from numpy import dtype
import torch 
import torch.nn as nn 

from typing import Tuple, Dict, Optional

from OpenEE.aggregation.aggregation import select_cls, select_marker, DynamicPooling
from OpenEE.head.classification import ClassificationHead
from OpenEE.head.crf import CRF


def get_model(model_args, backbone):
    if model_args.paradigm == "token_classification":
        return ModelForTokenClassification(model_args, backbone)
    elif model_args.paradigm == "sequence_labeling":
        return ModelForSequenceLabeling(model_args, backbone)
    elif model_args.paradigm == "seq2seq":
        return backbone
    else:
        raise ValueError("No such paradigm")


class ModelForTokenClassification(nn.Module):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForTokenClassification, self).__init__()
        self.backbone = backbone 
        self.aggregation = DynamicPooling(config)
        self.cls_head = ClassificationHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        trigger_left_mask: Optional[torch.Tensor] = None, 
        trigger_right_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
        ) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=attention_mask, \
                                token_type_ids=token_type_ids,
                                return_dict=True)   
        hidden_states = outputs.last_hidden_state
        # aggregation 
        # hidden_state = self.aggregation.select_cls(hidden_states)
        hidden_state = self.aggregation(hidden_states, trigger_left_mask, attention_mask)
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
        # pdb.set_trace()
        hidden_states = outputs.last_hidden_state
        # classification
        logits = self.cls_head(hidden_states) # [batch_size, seq_length, num_labels]
        # compute loss 
        mask = labels != 100
        loss = None 
        if labels is not None:
            # loss_fn = nn.CrossEntropyLoss()
            # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            # CRF
            loss = -self.crf(emissions=logits, 
                            tags=labels,
                            mask=mask,
                            reduction = "token_mean")
        else:
            preds = self.crf.decode(emissions=logits, mask=mask)
            logits = torch.LongTensor(preds)

        return dict(loss=loss, logits=logits)
