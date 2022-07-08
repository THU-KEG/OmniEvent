import pdb 
import torch 
import torch.nn as nn 


def select_cls(hidden_states: torch.Tensor) -> torch.Tensor:
    """Select CLS token as for textual information.
    
    Args:
        hidden_states: Hidden states encoded by backbone. shape: [batch_size, max_seq_length, hidden_size]

    Returns:
        hidden_state: Aggregated information. shape: [batch_size, hidden_size]
    """
    return hidden_states[:, 0, :]
    

def select_marker(hidden_states: torch.Tensor, marker_index: torch.Tensor) -> torch.Tensor:
    """Select marker token as for textual information.
    
    Args:
        hidden_states: Hidden states encoded by backbone. shape: [batch_size, max_seq_length, hidden_size]

    Returns:
        hidden_state: Aggregated information. shape: [batch_size, hidden_size]
    """
    batch_size = hidden_states.size(0)
    batch_indice = torch.arange(batch_size)
    return hidden_states[batch_indice, marker_index, :]


class DynamicPooling(nn.Module):
    def __init__(self, config) -> None:
        super(DynamicPooling, self).__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout()

    def forward(self, hidden_states: torch.Tensor, position_mask: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """Dynamic multi-pooling

        Args:
            hidden_states: [batch_size, max_seq_length, hidden_size]
            position_mask: [batch_size, max_seq_length]
        
        Returns:
            hidden_state: [batch_size, hidden_size*2]
        """
        minimum = -(2**32-1)
        left_mask = (1 - position_mask) * minimum
        right_mask = (1-(input_mask-position_mask)) * minimum
        left_hidden_state, _ = torch.max(hidden_states + left_mask.unsqueeze(-1), dim=1)
        right_hidden_state, _ = torch.max(hidden_states + right_mask.unsqueeze(-1), dim=1)
        pooled_output = torch.cat((left_hidden_state, right_hidden_state), dim=-1)
        # pooled_output = self.dense(hidden_state)
        # pooled_output = self.activation(pooled_output)
        # pooled_output = self.dropout(pooled_output)
        return pooled_output


# To do 
# - GCN 
# - ... 

        
    