import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")


@dataclass
class GraphData:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray | None = None


class GraphDataset:
    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = np.asarray(labels) if labels is not None else None


def _setup_mdataset(monkeypatch):
    mod = types.ModuleType("data.mdataset")
    mod.GraphData = GraphData
    mod.GraphDataset = GraphDataset
    monkeypatch.setitem(sys.modules, "data.mdataset", mod)


class DummyEncoder(torch.nn.Module):
    def forward(self, graphs):
        if isinstance(graphs, list):
            return torch.stack([torch.tensor(g.x.sum(axis=0)) for g in graphs])
        return torch.tensor(graphs.x.sum(axis=0))


def _make_ds(labels):
    graphs = [GraphData(np.ones((2, 2), dtype=np.float32), np.array([[0, 1], [1, 0]])) for _ in labels]
    return GraphDataset(graphs, labels)


def test_train_linear_head_classification(monkeypatch):
    _setup_mdataset(monkeypatch)
    from training.supervised_multi import train_linear_head_earlystop

    train_ds = _make_ds([0, 1, 0, 1])
    val_ds = _make_ds([0, 1])
    enc = DummyEncoder()
    metrics = train_linear_head_earlystop(
        enc,
        train_ds,
        val_ds,
        task_type="classification",
        epochs=2,
        lr=0.01,
        batch_size=2,
        device="cpu",
        patience=1,
    )
    assert "val_roc_auc" in metrics


def test_train_linear_head_regression(monkeypatch):
    _setup_mdataset(monkeypatch)
    from training.supervised_multi import train_linear_head_earlystop

    train_ds = _make_ds([0.0, 1.0, 0.5, 1.5])
    val_ds = _make_ds([0.0, 1.0])
    enc = DummyEncoder()
    metrics = train_linear_head_earlystop(
        enc,
        train_ds,
        val_ds,
        task_type="regression",
        epochs=2,
        lr=0.01,
        batch_size=2,
        device="cpu",
        patience=1,
    )
    assert "val_rmse" in metrics