from turtle import forward
import torch 
import torch.nn as nn



class ClassificationHead(nn.Module):
    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        scale = 2 if config.aggregation=="dm" else 1
        self.classifier = nn.Linear(config.hidden_size*scale, config.num_labels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Classify hidden_state to label distribution.
        
        Args:
            hidden_state: Aggregated textual information. shape: [batch_size, ..., hidden_size]
        
        Returns:
            logits: Raw, unnormalized scores for each label. shape: [batch_size, ..., num_labels]
        """
        logits = self.classifier(hidden_state)
        return logits



class MRCHead(nn.Module):
    pass 

