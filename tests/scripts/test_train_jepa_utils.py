import argparse
import types

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts.train_jepa import (
    aggregate_metrics,
    load_config,
    resolve_device,
    _maybe_to,
    _maybe_labels,
    _infer_num_classes,
    _iter_params,
    _maybe_state_dict,
    _to_bool,
)


def test_aggregate_metrics_empty():
    assert aggregate_metrics([]) == {}


def test_aggregate_metrics_mean_std_ignore_head():
    metrics = [
        {"acc": 1.0, "loss": 0.5, "head": 1},
        {"acc": 2.0, "loss": 1.5, "head": 2},
    ]
    agg = aggregate_metrics(metrics)
    assert agg == {
        "acc_mean": pytest.approx(1.5),
        "acc_std": pytest.approx(0.5),
        "loss_mean": pytest.approx(1.0),
        "loss_std": pytest.approx(0.5),
    }


def test_resolve_device_prefers_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch, "empty", lambda *a, **k: torch.tensor([]))
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda:1") == "cpu"


def test_resolve_device_handles_initialisation_failures(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def _boom(*_args, **_kwargs):  # pragma: no cover - behaviour exercised in test
        raise RuntimeError("cuda error: initialization error")

    monkeypatch.setattr(torch, "empty", _boom)
    assert resolve_device("cuda:0") == "cpu"


def test_load_config_valid(tmp_path):
    content = {"a": 1, "b": {"c": 2}}
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(content))
    loaded = load_config(str(cfg_file))
    assert loaded == content


def test_load_config_missing(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(str(missing))


def test_maybe_to_and_labels():
    captured = {}

    class Dummy:
        def to(self, device):
            captured["device"] = device

    module = Dummy()
    assert _maybe_to(module, "cpu") is module
    assert captured["device"] == "cpu"

    ds = types.SimpleNamespace(y=[0, 1, 1])
    labels = _maybe_labels(ds)
    assert labels.tolist() == [0, 1, 1]


def test_infer_num_classes():
    ds = types.SimpleNamespace(num_classes=4)
    assert _infer_num_classes(ds) == 4

    ds2 = types.SimpleNamespace(labels=np.array([0, 1, 1, 0]))
    assert _infer_num_classes(ds2) == 2


def test_iter_params_and_state_dict():
    module = torch.nn.Linear(2, 1)
    params = _iter_params(module)
    assert any(p is module.weight for p in params)

    wrapper = types.SimpleNamespace(encoder=module)
    assert _iter_params(wrapper)

    class DummyState:
        def state_dict(self):
            return {"w": 1}

    assert _maybe_state_dict(DummyState()) == {"w": 1}
    assert _maybe_state_dict(object()) is None


def test_to_bool_parsing():
    assert _to_bool("true") is True
    assert _to_bool("0") is False
    with pytest.raises(argparse.ArgumentTypeError):
        _to_bool("maybe")
