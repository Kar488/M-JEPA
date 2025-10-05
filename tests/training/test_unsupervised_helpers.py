import types

import types

import pytest

torch = pytest.importorskip("torch")

import training.unsupervised as unsup


def test_is_grad_scaler_detection():
    scaler = unsup.GradScaler()
    assert unsup._is_grad_scaler(scaler)
    assert not unsup._is_grad_scaler(object())


def test_make_wandb_handlers(monkeypatch):
    logs = []
    finishes = []

    monkeypatch.setattr(unsup, "is_main_process", lambda: True)

    class DummyWB:
        def __init__(self):
            self.run = types.SimpleNamespace(log=self.log)

        def log(self, payload, **kwargs):
            logs.append(payload)

        def finish(self):
            finishes.append(True)

    wb = DummyWB()
    log_fn, finish_fn = unsup._make_wandb_handlers(wb)
    log_fn({"loss": 1.0})
    finish_fn()
    assert logs == [{"loss": 1.0}]
    assert finishes == [True]


def test_should_compile_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert not unsup._should_compile_models(False, device, 1000)
    assert not unsup._should_compile_models(True, torch.device("cpu"), 1000)
    assert not unsup._should_compile_models(True, device, 10)
    if device.type != "cpu":
        assert unsup._should_compile_models(True, device, 1024)


def test_maybe_pin_returns_tensor():
    tensor = torch.ones(2, 2)
    pinned = unsup._maybe_pin(tensor)
    assert isinstance(pinned, torch.Tensor)
    with_device = unsup._maybe_pin(tensor, device="cpu")
    assert with_device.device.type == "cpu"

