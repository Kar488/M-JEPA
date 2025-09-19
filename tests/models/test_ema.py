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


def test_ema_copy_from_syncs_buffers() -> None:
    """EMA.copy_from should mirror buffer updates from the source model."""

    torch.manual_seed(1)
    source = _TinyBatchNormNet().train()
    ema = EMA(source, decay=0.8)

    with torch.no_grad():
        for _ in range(4):
            source(torch.randn(8, 4))

    ema.copy_from(source)

    target = _TinyBatchNormNet()
    ema.copy_to(target)

    torch.testing.assert_close(target.bn.running_mean, source.bn.running_mean)
    torch.testing.assert_close(target.bn.running_var, source.bn.running_var)
    assert target.bn.num_batches_tracked == source.bn.num_batches_tracked


def test_ema_state_dict_roundtrip_preserves_buffers() -> None:
    """Serialising and restoring EMA should preserve parameter and buffer snapshots."""

    torch.manual_seed(2)
    model = _TinyBatchNormNet().train()
    ema = EMA(model, decay=0.6, use_fp32=True)

    with torch.no_grad():
        for _ in range(6):
            model(torch.randn(10, 4))

    ema.update(model)
    snapshot = ema.state_dict()

    # Load into an EMA helper initialised with different settings to ensure state controls behaviour
    other_model = _TinyBatchNormNet()
    restored = EMA(other_model, decay=0.2, use_fp32=False)
    restored.load_state_dict(snapshot)

    assert restored.decay == snapshot["decay"]
    assert restored.use_fp32 == snapshot["use_fp32"]

    probe = _TinyBatchNormNet()
    restored.copy_to(probe)

    torch.testing.assert_close(probe.bn.running_mean, model.bn.running_mean)
    torch.testing.assert_close(probe.bn.running_var, model.bn.running_var)
    assert probe.bn.num_batches_tracked == model.bn.num_batches_tracked

