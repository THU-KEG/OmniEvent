#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from dataclasses import dataclass


@dataclass
class SumLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels):
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
