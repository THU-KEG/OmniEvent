import torch
import torch.nn as nn 
import torch.nn.functional as F

from typing import Optional


def get_aggregation(config):
    """Obtains the aggregation method to be utilized.

    Obtains the aggregation method to be utilized based on the model's configurations. The aggregation methods include
    selecting the `<cls>`s' representations, selecting the markers' representations, max-pooling, and dynamic
    multi-pooling.

    Args:
        config:
            The configurations of the model.

    Returns:
        The proposed method/class for the aggregation process.
        TODO: The data type of the variable `method` should be configured.
    """
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
              hidden_states: torch.Tensor,
              attention_mask: torch.Tensor,
              trigger_left: torch.Tensor,
              trigger_right: torch.Tensor,
              argument_left: torch.Tensor,
              argument_right: torch.Tensor,
              embeddings: Optional[torch.Tensor]=None):
    """Aggregates information to each position.

    Aggregates information to each position. The aggregation methods include selecting the "cls"s' representations,
    selecting the markers' representations, max-pooling, and dynamic multi-pooling.

    Args:
        config:
            The configurations of the model.
        method:
            The method proposed to be utilized in the aggregation process.
            TODO: The data type of the variable `method` should be configured.
        hidden_states (`torch.Tensor`):
            A tensor representing the hidden states output by the backbone model.
        trigger_left (`torch.Tensor`):
            A tensor indicating the left position of the triggers.
        trigger_right (`torch.Tensor`):
            A tensor indicating the right position of the triggers.
        argument_left (`torch.Tensor`):
            A tensor indicating the left position of the arguments.
        argument_right (`torch.Tensor`):
            A tensor indicating the right position of the arguments.
    """
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
        return method(hidden_states, attention_mask, trigger_left, embeddings, argument_left, argument_right)
    else:
        raise ValueError("Invaild %s aggregation method" % config.aggregation)


def max_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    """Applies the max-pooling operation over the sentence representation.

    Applies the max-pooling operation over the representation of the entire input sequence to capture the most useful
    information. The operation processes on the hidden states, which are output by the backbone model.

    Args:
        hidden_states (`torch.Tensor`):
            A tensor representing the hidden states output by the backbone model.

    Returns:
        pooled_states (`torch.Tensor`):
            A tensor represents the max-pooled hidden states, containing the most useful information of the sequence.
    """
    batch_size, seq_length, hidden_size = hidden_states.size()
    pooled_states = F.max_pool1d(input=hidden_states.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
    return pooled_states


def select_cls(hidden_states: torch.Tensor) -> torch.Tensor:
    """Returns the representations of the `<cls>` tokens.

    Returns the representations of each sequence's `<cls>` token by slicing the hidden state tensor output by the
    backbone model. The representations of the `<cls>` tokens contain general information of the sequences.

    Args:
        hidden_states (`torch.Tensor`):
            A tensor represents the hidden states output by the backbone model.

    Returns:
        `torch.Tensor`:
            A tensor containing the representations of each sequence's `<cls>` token.
    """
    return hidden_states[:, 0, :]
    

def select_marker(hidden_states: torch.Tensor, 
                  left: torch.Tensor,
                  right: torch.Tensor) -> torch.Tensor:
    """Returns the representations of the marker tokens.

    Returns the representations of each sequence's marker tokens by slicing the hidden state tensor output by the
    backbone model.

    Args:
        hidden_states (`torch.Tensor`):
            A tensor representing the hidden states output by the backbone model.
        left (`torch.Tensor`):
            A tensor indicates the left position of the markers.
        right (`torch.Tensor`):
            A tensor indicates the right position of the markers.

    Returns:
        marker_output (`torch.Tensor`):
            A tensor containing the representations of each sequence's marker tokens by concatenating their left and
            right token's representations.
    """
    batch_size = hidden_states.size(0)
    batch_indice = torch.arange(batch_size)
    left_states = hidden_states[batch_indice, left.to(torch.long), :]
    right_states = hidden_states[batch_indice, right.to(torch.long), :]
    marker_output = torch.cat((left_states, right_states), dim=-1)
    return marker_output


class DynamicPooling(nn.Module):
    """Dynamic multi-pooling layer for Convolutional Neural Network (CNN).

    Dynamic multi-pooling layer for Convolutional Neural Network (CNN), which is able to capture more valuable
    information within a sentence, particularly for some cases, such as multiple triggers are within a sentence and
    different argument candidate may play a different role with a different trigger.

    Attributes:
        dense (`nn.Linear`):
            TODO: The purpose of the linear layer should be configured.
        activation (`nn.Tanh`):
            An `nn.Tanh` layer representing the tanh activation function.
        dropout (`nn.Dropout`):
            An `nn.Dropout` layer for the dropout operation with the default dropout rate (0.5).
    """
    def __init__(self,
                 config) -> None:
        """Constructs a `DynamicPooling`."""
        super(DynamicPooling, self).__init__()
        self.dense = nn.Linear(config.hidden_size*config.head_scale, config.hidden_size*config.head_scale)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
    
    def get_mask(self,
                 position: torch.Tensor,
                 batch_size: int,
                 seq_length: int,
                 device: str) -> torch.Tensor:
        """Returns the mask indicating whether the token is padded or not."""
        all_masks = []
        for i in range(batch_size):
            mask = torch.zeros((seq_length), dtype=torch.int16, device=device)
            mask[:int(position[i])] = 1 
            all_masks.append(mask.to(torch.bool))
        all_masks = torch.stack(all_masks, dim=0)
        return all_masks

    def get_lexical_level_features(self, embeddings, position, max_seq_length):
        llf_idx = torch.stack([position-1, position, position+1], dim=0).to(torch.long)
        llf_idx[0] = llf_idx[0] * (llf_idx[0] != -1) + (position+2) * (llf_idx[0] == -1)
        llf_idx[2] = llf_idx[2] * (llf_idx[2] != max_seq_length) + (position-2) * (llf_idx[2] == max_seq_length)
        features = []
        for i in range(3):
            features.append(embeddings[torch.arange(embeddings.shape[0]), llf_idx[i]])
        features = torch.cat(features, dim=-1)
        return features

    def get_argument_lexical_features(self, embeddings, start, end, max_seq_length):
        mid_features = []
        for i in range(start.shape[0]):
            mid_features.append(torch.mean(embeddings[i, start[i]:end[i]+1], dim=0))
        mid_features = torch.stack(mid_features, dim=0)
        
        llf_idx = torch.stack([start-1, end+1], dim=0).to(torch.long)
        llf_idx[0] = llf_idx[0] * (llf_idx[0] != -1) + (end+2) * (llf_idx[0] == -1)
        llf_idx[1] = llf_idx[1] * (llf_idx[1] != max_seq_length) + (start-2) * (llf_idx[1] == max_seq_length)
        features = [mid_features]
        for i in range(2):
            features.append(embeddings[torch.arange(embeddings.shape[0]), llf_idx[i]])
        features = torch.cat(features, dim=-1)
        return features

    def max_pooling(self,
                    hidden_states: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """Conducts the max-pooling operation on the hidden states."""
        # import pdb; pdb.set_trace()
        batch_size, seq_length, hidden_size = hidden_states.size()
        mask = mask.unsqueeze(2)
        states = hidden_states * mask + mask * 100
        pooled_states = torch.max(states, dim=1)[0]
        pooled_states -= 100
        return pooled_states

    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor,
                trigger_position: torch.Tensor, 
                embeddings: Optional[torch.Tensor] = None,
                argument_left: Optional[torch.Tensor] = None,
                argument_right: Optional[torch.Tensor] = None,) -> torch.Tensor:
        """Conducts the dynamic multi-pooling process on the hidden states."""
        batch_size, seq_length = hidden_states.size()[:2]
        trigger_mask = self.get_mask(trigger_position, batch_size, seq_length, hidden_states.device)
        if embeddings is not None:
            lexical_features = self.get_lexical_level_features(embeddings, trigger_position, hidden_states.size(1))
        if argument_left is not None:
            if embeddings is not None:
                lexical_features = self.get_argument_lexical_features(embeddings, argument_left, argument_right, hidden_states.size(1))
                # lexical_features = self.get_lexical_level_features(embeddings, argument_left, hidden_states.size(1))
            argument_mask = self.get_mask(argument_left, batch_size, seq_length, hidden_states.device)
            left_mask = torch.logical_and(trigger_mask, argument_mask).to(torch.float32) * attention_mask
            middle_mask = torch.logical_xor(trigger_mask, argument_mask).to(torch.float32) * attention_mask
            right_mask = (1 - torch.logical_or(trigger_mask, argument_mask).to(torch.float32)) * attention_mask
            # import pdb; pdb.set_trace()
            # pooling 
            left_states = self.max_pooling(hidden_states, left_mask)
            middle_states = self.max_pooling(hidden_states, middle_mask)
            right_states = self.max_pooling(hidden_states, right_mask)
            pooled_output = torch.cat((left_states, middle_states, right_states), dim=-1)
        else:
            left_mask = trigger_mask.to(torch.float32) * attention_mask
            right_mask = (1 - left_mask) * attention_mask
            left_states = self.max_pooling(hidden_states, left_mask)
            right_states = self.max_pooling(hidden_states, right_mask)
            pooled_output = torch.cat((left_states, right_states), dim=-1)
        if embeddings is not None:
            final_output = torch.cat([pooled_output, lexical_features], dim=-1)
        else:
            final_output = pooled_output
        final_output = self.dropout(final_output)
        return final_output


# To do 
# - Test MOGANED
class MOGCN(nn.Module):
    """Multi-order Graph Convolutional Network (MOGAN).

    A Multi-order Graph Convolutional Network (MOGAN) class, which simply learns a list of representations over
    multi-order syntactic graphs by a few parallel Graph Attention Network (GAT) layers, which weights the importance of
    neighbors of each word in each syntactic graph during convolution.

    Attributes:
        in_dim (`int`):
            An integer indicating the dimension of GAT's input features.
        hidden_dim (`int`):
            An integer indicating the dimension of GAT's output features.
        device:
            The device of the operation, CPU or GPU.
            TODO: Configure the data type of the `device` variable.
        in_drop (`int`):
            An integer indicating the dropout rate.
        K (`int`):
            An integer indicating the number of times operating the graph attention convolution process.
        layers_a (`nn.ModuleList`):
            A GAT layer operating the first sub-matrix of the adjacency matrix of the first-order syntactic graph,
            A_along, containing the connection information of the first-order syntactic graph.
        layers_b (`nn.ModuleList`):
            A GAT layer operating the second sub-matrix of the adjacency matrix of the first-order syntactic graph,
            A_rev, which is a transpose matrix of A_along.
        layers_c (`nn.ModuleList`):
            A GAT layer operating the third sub-matrix of the adjacency matrix of the first-order syntactic graph,
            A_loop, which is an identity matrix.
        Wawa (`nn.Sequential`):
            An `nn.Sequential` container with a linear transformation and a tanh activation function, which is regarded
            as a graph attention convolutional function.
        Ctx (`nn.Linear`):
            An `nn.Linear` layer for computing the normalized weight of each neighbor when updating a node.
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 K: int,
                 dropout: int,
                 device,
                 alpha: Optional[int] = 0.2) -> None:
        """Constructs a `MOGCN`."""
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

    def forward(self,
                hidden_states: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """The forward propagation of `NOGCN`."""
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
    """Simple graph attention layer.

    A simple graph attention layer for the aggregation process, which is the sole layer throughout all of the GAT
    architectures, performing self-attention on all nodes, and aggregating the information based on the importance of
    neighbors of each node.

    Attributes:
        dropout (`int`):
            An integer indicating the dropout rate.
        in_features (`int`):
            An integer indicating the dimension of the input features.
        out_features (`int`):
            An integer indicating the dimension of the output features.
        alpha (`float`):
            A float variable indicating the negative slope of the leaky relu activation.
        W (`nn.Parameter`):
            An `nn.Parameter` instance representing the weight of the fully connecting layer, transforming high-
            dimensional features into low dimensions.
        a (`nn.Parameter`):
            An `nn.Parameter` instance indicating the initial attention weight between nodes.
        leaky_relu (`nn.LeakyReLU`):
            An `nn.LeakyReLU` layer representing the leaky relu activation function.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: int,
                 alpha: int,
                 device,
                 concat: Optional[bool] = False):
        """Constructs a `GraphAttentionLayer`."""
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
        """The forward propagation of a simple graph attention layer."""
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


def matmuls(a: torch.Tensor,
            times: int) -> torch.Tensor:
    """Multiplies the input matrix with itself for `times` times.

    Multiplies the input matrix with itself multiple times, in which each time of multiplication follows matrix-matrix
    multiplication.

    Args:
        a (`torch.Tensor`):
            A tensor representing the input matrix for multiplication.
        times (`int`):
            An integer indicating the number of times the matrix would be multiplied with itself.

    Returns:
        res (`torch.Tensor`):
            A tensor representing the matrix after `times` times multiplication of the given matrix.
    """
    res = a
    for i in range(times):
        res = torch.matmul(res, a)
    return res
