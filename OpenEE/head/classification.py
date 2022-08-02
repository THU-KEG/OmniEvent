from turtle import forward
import torch 
import torch.nn as nn


class ClassificationHead(nn.Module):
    """A classification head for classifying hidden states to label distributions.

    A classification head for classifying hidden states to label distributions by selecting the label with the highest
    probability corresponding to each logit.

    Attributes:
        classifier (`nn.Linear`):
            An `nn.Linear` layer classifying each logit into its corresponding label.
    """
    def __init__(self,
                 config) -> None:
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Linear(config.hidden_size*config.head_scale, config.num_labels)

    def forward(self,
                hidden_state: torch.Tensor) -> torch.Tensor:
        """Classifies hidden_state to label distribution.
        
        Args:
            hidden_state: Aggregated textual information. shape: [batch_size, ..., hidden_size]
        
        Returns:
            logits: Raw, unnormalized scores for each label. shape: [batch_size, ..., num_labels]
        """
        logits = self.classifier(hidden_state)
        return logits


class MRCHead(nn.Module):
    """A classification head for the Machine Reading Comprehension (MRC) model.

    A classification head for the Machine Reading Comprehension (MRC) model by predicting the answer of each question
    corresponding to a mention type. The classifier returns two logits indicating the start and end position of each
    mention.

    Attributes:
        qa_outputs (`nn.Linear`):
            An `nn.Linear` layer indicates the start and end position of each mention corresponding to the question.
    """
    def __init__(self,
                 config) -> None:
        """Constructs a `MRCHead`."""
        super(MRCHead, self).__init__()
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
    
    def forward(self,
                hidden_state: torch.Tensor):
        """The forward propagation of `MRCHead`."""
        logits = self.qa_outputs(hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits


