"""Tests for the EMA utility."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.ema import EMA


class _TinyBatchNormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.linear(x))


def test_ema_copies_batchnorm_buffers() -> None:
    """EMA should copy BatchNorm running stats into the target model."""

    torch.manual_seed(0)
    model = _TinyBatchNormNet()
    ema = EMA(model, decay=0.5)

    model.train()
    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(16, 4))

    # BatchNorm buffers should have diverged from their initial values.
    running_mean = model.bn.running_mean.clone()
    running_var = model.bn.running_var.clone()
    assert not torch.allclose(running_mean, torch.zeros_like(running_mean))
    assert not torch.allclose(running_var, torch.ones_like(running_var))

    ema.update(model)

    target = _TinyBatchNormNet()
    ema.copy_to(target)

    torch.testing.assert_close(target.bn.running_mean, model.bn.running_mean)
    torch.testing.assert_close(target.bn.running_var, model.bn.running_var)
    assert target.bn.num_batches_tracked == model.bn.num_batches_tracked
