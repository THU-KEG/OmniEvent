import pdb 
import torch 
import torch.nn as nn 
import torch.nn.functional as F



def get_aggregation(config):
    if config.aggregation == "cls":
        return select_cls
    elif config.aggregation == "marker":
        return select_marker
    elif config.aggregation == "dynamic_pooling":
        return DynamicPooling(config)
    elif config.aggregation == "max_pooling":
        return max_pooling
    else:
        raise ValueError("Invaild %s aggregation method" % config.aggregation)


def aggregate(config,
              method, 
              hidden_states, 
              trigger_left,
              trigger_right,
              argument_left,
              argument_right):
    if config.aggregation == "cls":
        return method(hidden_states)
    elif config.aggregation == "marker":
        if argument_left is not None:
            return method(hidden_states, argument_left, argument_right)
        else:
            return method(hidden_states, trigger_left, trigger_right)
    elif config.aggregation == "max_pooling":
        return method(hidden_states)
    elif config.aggregation == "dynamic_pooling":
        return method(hidden_states, trigger_left, argument_left)
    else:
        raise ValueError("Invaild %s aggregation method" % config.aggregation)


def max_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, seq_length, hidden_size = hidden_states.size()
    pooled_states = F.max_pool1d(input=hidden_states.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
    return pooled_states


def select_cls(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states[:, 0, :]
    

def select_marker(hidden_states: torch.Tensor, 
                  left: torch.Tensor,
                  right: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_states.size(0)
    batch_indice = torch.arange(batch_size)
    left_states = hidden_states[batch_indice, left.to(torch.long), :]
    right_states = hidden_states[batch_indice, right.to(torch.long), :]
    marker_output = torch.cat((left_states, right_states), dim=-1)
    return marker_output


class DynamicPooling(nn.Module):
    def __init__(self, config) -> None:
        super(DynamicPooling, self).__init__()
        self.dense = nn.Linear(config.hidden_size*config.head_scale, config.hidden_size*config.head_scale)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout()
    
    def get_mask(self, position, batch_size, seq_length, device):
        all_masks = []
        for i in range(batch_size):
            mask = torch.zeros((seq_length), dtype=torch.int16, device=device)
            mask[:int(position[i])] = 1 
            all_masks.append(mask.to(torch.bool))
        all_masks = torch.stack(all_masks, dim=0)
        return all_masks

    def max_pooling(self, hidden_states, mask):
        batch_size, seq_length, hidden_size = hidden_states.size()
        conved = hidden_states.transpose(1, 2)
        conved = conved.transpose(0, 1)
        states = (conved * mask).transpose(0, 1)
        states += torch.ones_like(states)
        pooled_states = F.max_pool1d(input=states, kernel_size=seq_length).contiguous().view(batch_size, hidden_size)
        pooled_states -= torch.ones_like(pooled_states)
        return pooled_states

    def forward(self, 
                hidden_states: torch.Tensor, 
                trigger_position: torch.Tensor, 
                argument_position: torch.Tensor=None) -> torch.Tensor:
        """Dynamic multi-pooling
        Args:
            hidden_states: [batch_size, max_seq_length, hidden_size]
            trigger_position: [batch_size]
            argument_position: [batch_size]
        
        Returns:
            hidden_state: [batch_size, hidden_size*2]
        """
        batch_size, seq_length = hidden_states.size()[:2]
        trigger_mask = self.get_mask(trigger_position, batch_size, seq_length, hidden_states.device)
        if argument_position is not None:
            argument_mask = self.get_mask(argument_position, batch_size, seq_length, hidden_states.device)
            left_mask = torch.logical_and(trigger_mask, argument_mask).to(torch.float32) 
            middle_mask = torch.logical_xor(trigger_mask, argument_mask).to(torch.float32) 
            right_mask = 1 - torch.logical_or(trigger_mask, argument_mask).to(torch.float32)
            # pooling 
            left_states = self.max_pooling(hidden_states, left_mask)
            middle_states = self.max_pooling(hidden_states, middle_mask)
            right_states = self.max_pooling(hidden_states, right_mask)
            pooled_output = torch.cat((left_states, middle_states, right_states), dim=-1)
        else:
            left_mask = trigger_mask.to(torch.float32)
            right_mask = 1 - left_mask
            left_states = self.max_pooling(hidden_states, left_mask)
            right_states = self.max_pooling(hidden_states, right_mask)
            pooled_output = torch.cat((left_states, right_states), dim=-1)

        return pooled_output


# To do 
# - GCN 
# - ... 

        
    