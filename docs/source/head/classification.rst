Classification Head
===================

.. code-block:: python

    from turtle import forward
    import torch
    import torch.nn as nn

``LinearHead``
--------------

A token-wise classification head for classifying hidden states to label distributions through a linear
transformation, selecting the label with the highest probability corresponding to each logit.

**Attributes:**

- ``classifier``: An ``nn.Linear`` layer classifying each logit into its corresponding label.

.. code-block:: python

    class LinearHead(nn.Module):
        """A token-wise classification head for classifying the hidden states to label distributions.
        A token-wise classification head for classifying hidden states to label distributions through a linear
        transformation, selecting the label with the highest probability corresponding to each logit.
        Attributes:
            classifier (`nn.Linear`):
                An `nn.Linear` layer classifying each logit into its corresponding label.
        """
        def __init__(self, config):
            super(LinearHead, self).__init__()
            self.classifier = nn.Linear(config.hidden_size*config.head_scale, config.num_labels)

        def forward(self,
                    hidden_state: torch.Tensor) -> torch.Tensor:
            """Classifies hidden states to label distribution."""
            logits = self.classifier(hidden_state)
            return logits

``MRCHead``
-----------

A classification head for the Machine Reading Comprehension (MRC) paradigm, predicting the answer of each question
corresponding to a mention type. The classifier returns two logits indicating the start and end position of each
mention corresponding to the question.

**Attributes:**

- ``qa_outputs``: An ``nn.Linear`` layer transforming the hidden states to two logits, indicating the start and end position of a given mention type.

.. code-block:: python

    class MRCHead(nn.Module):
        """A token-wise classification head for the Machine Reading Comprehension (MRC) paradigm.
        A classification head for the Machine Reading Comprehension (MRC) paradigm, predicting the answer of each question
        corresponding to a mention type. The classifier returns two logits indicating the start and end position of each
        mention corresponding to the question.
        Attributes:
            qa_outputs (`nn.Linear`):
                An `nn.Linear` layer transforming the hidden states to two logits, indicating the start and end position
                of a given mention type.
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
