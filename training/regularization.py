"""Regularization utilities for training loops."""

from __future__ import annotations

import torch
import torch.nn as nn


def l2_regularization(model: nn.Module) -> torch.Tensor:
    """Compute the sum of squared parameters for ``model``.

    The returned scalar tensor can be multiplied by a regularisation
    coefficient before being added to the overall loss.
    """
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        reg += torch.sum(p ** 2)
    return reg
