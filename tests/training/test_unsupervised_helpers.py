import os
import types

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from data.mdataset import GraphData

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


def test_maybe_pin_handles_type_error(monkeypatch):
    tensor = torch.ones(1)
    outputs = []

    def fake_pin(self, *args, **kwargs):
        outputs.append(kwargs)
        if kwargs:
            raise TypeError("unexpected kwargs")
        return torch.zeros_like(self)

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin, raising=False)
    pinned = unsup._maybe_pin(tensor, device="cuda")
    assert torch.equal(pinned, torch.zeros_like(tensor))
    assert outputs[0] == {"device": "cuda"}


def test_graph_serialisation_roundtrip():
    graph = GraphData(
        x=np.ones((2, 3), dtype=float),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_attr=np.ones((2, 1), dtype=float),
        pos=np.zeros((2, 3), dtype=float),
    )
    graph.custom = torch.tensor([1.0, 2.0])

    state = unsup._graph_to_serialisable(graph)
    assert state["extras"]["custom"].tolist() == [1.0, 2.0]

    restored = unsup._graph_from_serialisable(state)
    assert np.array_equal(np.asarray(restored.edge_index), graph.edge_index)
    assert getattr(restored, "custom").tolist() == [1.0, 2.0]


def test_resolve_ckpt_dir_fallback_on_permission_error(tmp_path, monkeypatch):
    unwritable = tmp_path / "locked"
    unwritable.mkdir()
    unwritable.chmod(0o500)

    stage_dir = tmp_path / "stage"
    monkeypatch.setenv("STAGE_OUTPUTS_DIR", str(stage_dir))

    fallback = unsup._resolve_ckpt_dir(str(unwritable / "ckpts"))
    assert fallback == os.path.join(str(stage_dir), "ckpts")
    assert os.path.isdir(fallback)

    unwritable.chmod(0o700)

