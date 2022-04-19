import torch 
import torch.nn as nn 


class SimpleAggregation(object):
    """Simple aggregation such as index selection for textual information."""

    def __init__(self) -> None:
        pass 


    def select_cls(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Select CLS token as for textual information.
        
        Args:
            hidden_states: Hidden states encoded by backbone. shape: [batch_size, max_seq_length, hidden_size]

        Returns:
            hidden_state: Aggregated information. shape: [batch_size, hidden_size]
        """
        return hidden_states[:, 0, :]
    

    def select_marker(self, hidden_states: torch.Tensor, marker_index: torch.Tensor) -> torch.Tensor:
        """Select marker token as for textual information.
        
        Args:
            hidden_states: Hidden states encoded by backbone. shape: [batch_size, max_seq_length, hidden_size]

        Returns:
            hidden_state: Aggregated information. shape: [batch_size, hidden_size]
        """
        batch_size = hidden_states.size(0)
        batch_indice = torch.arange(batch_size)
        return hidden_states[batch_indice, marker_index, :]


    def dynamic_pooling(self, hidden_states: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        pass 


# To do 
# - GCN 
# - ... 

        
    