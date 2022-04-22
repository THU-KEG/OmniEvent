import pdb 
import torch 
import torch.nn as nn 

from typing import Tuple, Dict 

from OpenEE.aggregation.aggregation import SimpleAggregation
from OpenEE.head.classification import ClassificationHead



class ModelForTokenClassification(nn.Module):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForTokenClassification, self).__init__()
        self.backbone = backbone 
        self.aggregation = SimpleAggregation()
        self.cls_head = ClassificationHead(config)


    def forward(self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        trigger_left_mask: torch.Tensor, 
        trigger_right_mask: torch.Tensor,
        labels: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=input_mask, \
                                token_type_ids=segment_ids,
                                return_dict=True)   
        hidden_states = outputs.last_hidden_state
        # aggregation 
        # hidden_state = self.aggregation.select_cls(hidden_states)
        hidden_state = self.aggregation.dynamic_pooling(hidden_states, trigger_left_mask, input_mask)
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
        self.cls_head = ClassificationHead(config)


    def forward(self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        trigger_left_mask: torch.Tensor, 
        trigger_right_mask: torch.Tensor,
        labels: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=input_mask, \
                                token_type_ids=segment_ids,
                                return_dict=True)   
        # pdb.set_trace()
        hidden_states = outputs.last_hidden_state
        # classification
        logits = self.cls_head(hidden_states) # [batch_size, seq_length, num_labels]
        # compute loss 
        loss = None 
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            # pdb.set_trace()
        return dict(loss=loss, logits=logits)
        
        
