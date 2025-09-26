import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts.train_jepa import aggregate_metrics, load_config, resolve_device


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
