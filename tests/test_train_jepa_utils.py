import pytest
import torch

from scripts.train_jepa import aggregate_metrics, resolve_device


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
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda:1") == "cpu"
