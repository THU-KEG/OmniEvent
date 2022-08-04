import pdb
import torch 
import torch.nn as nn 

from numpy import dtype
from typing import Tuple, Dict, Optional

from OpenEE.aggregation.aggregation import get_aggregation, aggregate
from OpenEE.head.classification import (
    ClassificationHead,
    MRCHead
)
from OpenEE.head.crf import CRF


def get_model(model_args,
              backbone):
    """Returns the model to be utilized for training and prediction.

    Returns the model to be utilized for training and prediction based on the pre-defined paradigm.

    Args:
        model_args:
            The arguments of the model for training and prediction.
        backbone:
            The backbone model obtained from the `get_backbone()` method.

    Returns:
        The model class of the model to be utilized for training and prediction.
    """
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
    """Bert model for token classification.

    Bert model for token classification, which firstly obtains hidden states through the backbone model, then aggregates
    the hidden states through the aggregation method/class, and finally classifies each token to their corresponding
    label.

    Attributes:
        config:
            The configurations of the model.
        backbone:
            The backbone network proposed to output initial hidden states.
        aggregation:
            The aggregation method/class for aggregating the hidden states output by the backbone network.
        cls_head (`ClassificationHead`):
            A `ClassificationHead` instance classifying each token into its corresponding label.
    """

    def __init__(self,
                 config,
                 backbone) -> None:
        """Constructs a `ModelForTokenClassification`."""
        super(ModelForTokenClassification, self).__init__()
        self.config = config
        self.backbone = backbone 
        self.aggregation = get_aggregation(config)
        self.cls_head = ClassificationHead(config)

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
           returns the predicted and loss."""
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
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
    """Bert model for sequence labeling.

    Bert model for sequence labeling, which firstly obtains hidden states through the backbone model, then labels each
    token to their corresponding label, and finally decodes the label through a Conditional Random Field (CRF) module.

    Attributes:
        config:
            The configurations of the model.
        backbone:
            The backbone network proposed to output initial hidden states.
        cls_head (`ClassificationHead`):
            A `ClassificationHead` instance classifying each token into its corresponding label.
    """

    def __init__(self,
                 config,
                 backbone) -> None:
        """Constructs a `ModelForSequenceLabeling`."""
        super(ModelForSequenceLabeling, self).__init__()
        self.backbone = backbone 
        self.crf = CRF(config.num_labels, batch_first=True)
        self.cls_head = ClassificationHead(config)

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
    """Model for Machine Reading Comprehension (MRC).

    Bert model for Machine Reading Comprehension (MRC), which firstly obtains hidden states through the backbone model,
    then predicts the start and end logits through the MRC head.

    Attributes:
        config:
            The configurations of the model.
        backbone:
            The backbone network proposed to output initial hidden states.
        mrc_head (`MRCHead`):
            A `MRCHead` instance classifying the hidden states into start and end logits.
    """

    def __init__(self,
                 config,
                 backbone) -> None:
        """Constructs a `ModelForMRC`."""
        super(ModelForMRC, self).__init__()
        self.backbone = backbone
        self.mrc_head = MRCHead(config)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                argument_left: Optional[torch.Tensor] = None,
                argument_right: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Manipulates the inputs through a backbone, and MRC head module,
           returns the predicted logits and loss."""
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
        logits = torch.cat((start_logits, end_logits), dim=-1) # [batch_size, seq_length*2]
        
        return dict(loss=total_loss, logits=logits)
