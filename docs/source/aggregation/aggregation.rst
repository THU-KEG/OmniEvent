Aggregation
===========

.. code-block:: python

    import pdb
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from typing import

``get_aggregation``
-------------------

Obtains the aggregation method to be utilized based on the model's configurations. The aggregation methods include
selecting the ``<cls>``s' representations, selecting the markers' representations, max-pooling, and dynamic
multi-pooling.

**Args:**

- ``config``: The configurations of the model.

**Returns:**

- The proposed method/class for the aggregation process.

.. code-block:: python

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

``aggregate``
-------------

Aggregates information to each position. The aggregation methods include selecting the "cls"s' representations,
selecting the markers' representations, max-pooling, and dynamic multi-pooling.

**Args:**

- ``config``: The configurations of the model.
- ``method``: The method proposed to be utilized in the aggregation process.
- ``hidden_states``: A tensor representing the hidden states output by the backbone model.
- ``trigger_left``: A tensor indicating the left position of the triggers.
- ``trigger_right``: A tensor indicating the right position of the triggers.
- ``argument_left``: A tensor indicating the left position of the arguments.
- ``argument_right``: A tensor indicating the right position of the arguments.

.. code-block:: python

    def aggregate(config,
                  method,
                  hidden_states: torch.Tensor,
                  trigger_left: torch.Tensor,
                  trigger_right: torch.Tensor,
                  argument_left: torch.Tensor,
                  argument_right: torch.Tensor):
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
            return method(hidden_states, trigger_left, argument_left)
        else:
            raise ValueError("Invaild %s aggregation method" % config.aggregation)

``max_pooling``
---------------

Applies the max-pooling operation over the representation of the entire input sequence to capture the most useful
information. The operation processes on the hidden states, which are output by the backbone model.

**Args:**

- ``hidden_states`: A tensor representing the hidden states output by the backbone model.

**Returns:**

- ``pooled_states``: A tensor represents the max-pooled hidden states, containing the most useful information of the sequence.

.. code-block:: python

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

``select_cls``
--------------

Returns the representations of each sequence's ``<cls>`` token by slicing the hidden state tensor output by the
backbone model. The representations of the ``<cls>`` tokens contain general information of the sequences.

**Args:**

- ``hidden_states``: A tensor represents the hidden states output by the backbone model.

**Returns:**

- A tensor containing the representations of each sequence's `<cls>` token.

.. code-block:: python

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


``select_marker``
-----------------

Returns the representations of each sequence's marker tokens by slicing the hidden state tensor output by the
backbone model.

**Args:**

- ``hidden_states``: A tensor representing the hidden states output by the backbone model.
- ``left``: A tensor indicates the left position of the markers.
- ``right``: A tensor indicates the right position of the markers.

**Returns:**

- ``marker_output`: A tensor containing the representations of each sequence's marker tokens by concatenating their left and right token's representations.

.. code-block:: python

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

``DynamicPooling``
------------------

Dynamic multi-pooling layer for Convolutional Neural Network (CNN), which is able to capture more valuable
information within a sentence, particularly for some cases, such as multiple triggers are within a sentence and
different argument candidate may play a different role with a different trigger.

**Attributes:**

- ``activation``: An `nn.Tanh` layer representing the tanh activation function.
- ``dropout``: An `nn.Dropout` layer for the dropout operation with the default dropout rate (0.5).

.. code-block:: python

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
            self.dropout = nn.Dropout()

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

        def max_pooling(self,
                        hidden_states: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
            """Conducts the max-pooling operation on the hidden states."""
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
                    argument_position: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Conducts the dynamic multi-pooling process on the hidden states."""
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
