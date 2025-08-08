"""Exponential Moving Average (EMA) utility.

This module defines a simple utility class that maintains a moving
average of model parameters. The EMA copy serves as the slowly
changing target encoder in self‑supervised JEPA training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """Maintain an exponential moving average of model parameters.

    Args:
        model: The source model whose parameters will be averaged.
        decay: EMA decay factor between 0 and 1. Higher values result in
            slower updates.
    """

    def __init__(self, model: nn.Module, decay: float = 0.99) -> None:
        self.decay = decay
        self.params = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters given the source model."""
        for ema_p, p in zip(self.params, model.parameters()):
            ema_p.mul_(self.decay)
            ema_p.add_(p.detach() * (1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters into the target model."""
        for p, ema_p in zip(model.parameters(), self.params):
            p.data.copy_(ema_p)
