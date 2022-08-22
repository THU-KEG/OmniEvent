#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright Text2Event from https://github.com/luyaojie/Text2Event.
# Licensed under the MIT License.

import torch

from dataclasses import dataclass
from typing import Dict


@dataclass
class SumLabelSmoother:
    """A label-smoothing sum module operated on the pre-computed output from the model.

    A label-smoothing sum module operated on the pre-computed output from the model, which is a regularization technique
    that addresses the overfitting and overconfidence problems by adding some noises to decrease the weights of the
    actual samples when calculating losses.

    Attributes:
        epsilon (`float`, `optional`, defaults to 0.1):
            A float variable indicating the label smoothing factor.
        ignore_index (`int`, `optional`, defaults to -100):
            An integer representing the index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self,
                 model_output: Dict[str, torch.Tensor],
                 labels: torch.Tensor) -> float:
        """Conducts the label smoothing process."""
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels.clamp_min_(0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        # num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum()  # / num_active_elements
        smoothed_loss = smoothed_loss.sum()  # / (num_active_elements * log_probs.shape[-1])
        eps_i = self.epsilon / log_probs.size(-1)
        return (1 - self.epsilon) * nll_loss + eps_i * smoothed_loss