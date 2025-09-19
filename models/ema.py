"""Exponential Moving Average (EMA) utility.

This module defines a simple utility class that maintains a moving
average of model parameters. The EMA copy serves as the slowly
changing target encoder in self‑supervised JEPA training.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn

_NO_GRAD = getattr(torch, "no_grad", lambda: nullcontext())


class EMA:
    """Maintain an exponential moving average of model parameters.

    Args:
        model: The source model whose parameters will be averaged.
        decay: EMA decay factor between 0 and 1. Higher values result in
            slower updates.
    """

    def __init__(self, model: nn.Module, decay: float = 0.99, use_fp32: bool = True) -> None:
        # Coerce decay to a plain float (handle accidental list/tuple/np types)
        try:
            if isinstance(decay, (list, tuple)):
                decay = decay[0]
            import numpy as _np
            if isinstance(decay, _np.ndarray):
                decay = decay.flatten()[0]
        except Exception:
            pass
        self.decay = float(decay)
        self.use_fp32 = use_fp32
        # Clone on the model's current device; optionally store in fp32 for stability
        self.params = []
        for p in model.parameters():
            buf = p.detach().clone()
            if self.use_fp32 and torch.is_floating_point(buf):
                buf = buf.float()
            buf = buf.to(p.device)
            self.params.append(buf)

        # Buffers (e.g., BatchNorm running stats) must also be tracked. They do
        # not participate in gradient updates but strongly influence model
        # behaviour, so we keep an exact copy instead of an EMA average.
        self.buffers = []
        for buf in model.buffers():
            clone = buf.detach().clone()
            if self.use_fp32 and torch.is_floating_point(clone):
                clone = clone.float()
            clone = clone.to(buf.device)
            self.buffers.append(clone)

    def set_decay(self, decay: float) -> None:
        """Dynamically update EMA decay/momentum during training."""
        self.decay = float(decay)

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters given the source model."""
        with _NO_GRAD():
            for i, p in enumerate(model.parameters()):
                if not p.requires_grad:
                    continue
                ema_p = self.params[i]
                # Ensure device/dtype match for the update
                if ema_p.device != p.device:
                    ema_p = ema_p.to(p.device)
                    self.params[i] = ema_p
                src = p.detach()
                if self.use_fp32:
                    src = src.to(torch.float32)
                else:
                    src = src.to(ema_p.dtype)
                # ema = decay*ema + (1-decay)*param  (stable in-place)
                ema_p.lerp_(src, float(1.0 - float(self.decay)))

            # Buffers are copied directly without decay to keep running stats in
            # sync with the source model.
            for i, buf in enumerate(model.buffers()):
                if i >= len(self.buffers):
                    break
                ema_buf = self.buffers[i]
                if ema_buf.device != buf.device:
                    ema_buf = ema_buf.to(buf.device)
                    self.buffers[i] = ema_buf
                tensor = buf.detach()
                if self.use_fp32 and torch.is_floating_point(tensor):
                    tensor = tensor.to(torch.float32)
                else:
                    tensor = tensor.to(ema_buf.dtype)
                ema_buf.copy_(tensor)


    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters into the target model."""
        with _NO_GRAD():
            for p, ema_p in zip(model.parameters(), self.params):
                p.data.copy_(ema_p.to(p.dtype).to(p.device))
            for buf, ema_buf in zip(model.buffers(), self.buffers):
                buf.data.copy_(ema_buf.to(buf.dtype).to(buf.device))

    def copy_from(self, model: nn.Module) -> None:
        """Overwrite EMA buffers with parameters from ``model``."""

        with _NO_GRAD():
            for i, src in enumerate(model.parameters()):
                if i >= len(self.params):
                    break
                buf = self.params[i]
                tensor = src.detach()
                if self.use_fp32 and torch.is_floating_point(tensor):
                    tensor = tensor.to(torch.float32)
                else:
                    tensor = tensor.to(buf.dtype)
                self.params[i].copy_(tensor.to(buf.device))

            for i, src in enumerate(model.buffers()):
                if i >= len(self.buffers):
                    break
                buf = self.buffers[i]
                tensor = src.detach()
                if self.use_fp32 and torch.is_floating_point(tensor):
                    tensor = tensor.to(torch.float32)
                else:
                    tensor = tensor.to(buf.dtype)
                self.buffers[i].copy_(tensor.to(buf.device))

    def to(self, device: torch.device) -> None:
        """Move EMA buffers to a device."""
        with _NO_GRAD():
            for i, ema_p in enumerate(self.params):
                self.params[i] = ema_p.to(device)
            for i, ema_buf in enumerate(self.buffers):
                self.buffers[i] = ema_buf.to(device)
