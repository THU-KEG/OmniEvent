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
# - Test MOGANED
class MOGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, K, dropout, device, alpha=0.2):
        super(MOGCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.in_drop = dropout
        self.K = K

        self.layers_a = nn.ModuleList()
        self.layers_b = nn.ModuleList()
        self.layers_c = nn.ModuleList()
        for i in range(self.K):
            self.layers_a.append(GraphAttentionLayer(in_features=in_dim, out_features=hidden_dim,
                                                     dropout=dropout, alpha=alpha, device=device, concat=False))
            self.layers_b.append(GraphAttentionLayer(in_features=in_dim, out_features=hidden_dim,
                                                     dropout=dropout, alpha=alpha, device=device, concat=False))
            self.layers_c.append(GraphAttentionLayer(in_features=in_dim, out_features=hidden_dim,
                                                     dropout=dropout, alpha=alpha, device=device, concat=False))

        self.Wawa = nn.Sequential(
            nn.Linear(self.hidden_dim, 100),
            nn.Tanh(),
        )
        self.Ctx = nn.Linear(100, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_a, adj_b, adj_c = adj[:, 0, :, :], adj[:, 1, :, :], adj[:, 2, :, :]

        # The GAC procedures.
        hs = []
        for layer in range(self.K):
            h_layer = self.layers_a[layer](matmuls(adj_a, layer), hidden_states) + \
                      self.layers_b[layer](matmuls(adj_b, layer), hidden_states) + \
                      self.layers_c[layer](matmuls(adj_c, layer), hidden_states)
            hs.append(h_layer)

        # The aggregation procedures.
        s_ctxs = []
        for layer in range(self.K):
            s_layer = self.Wawa(hs[layer])
            ctx_apply = self.Ctx(s_layer)
            s_ctxs.append(ctx_apply)
        vs = F.softmax(torch.cat(s_ctxs, dim=2), dim=2)  # [batch_size, max_len, 3]
        h_concats = torch.cat([torch.unsqueeze(hs[layer], 2) for layer in range(self.K)], dim=2)
        final_h = torch.sum(torch.mul(torch.unsqueeze(vs, 3), h_concats), dim=2)

        return final_h


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903.
    """
    def __init__(self, in_features, out_features, dropout, alpha, device, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, adj, input):
        h = torch.matmul(input, self.W)       # [B, N, D]
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1),
                             h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_features)  # [B, N, N, 2D]
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(3))                          # [B ,N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)  # [B, N, N]
        h_prime = torch.matmul(attention, h)  # [B, N, D]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


def matmuls(a, times):
    res = a
    for i in range(times):
        res = torch.matmul(res, a)
    return res
        
    