import sys
import types
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GraphData:
    x: np.ndarray
    edge_index: np.ndarray


class GraphDataset:
    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = np.asarray(labels) if labels is not None else None


md = types.ModuleType("data.mdataset")
md.GraphData = GraphData
md.GraphDataset = GraphDataset
sys.modules["data.mdataset"] = md

from training.multitask import train_multilabel_head  # noqa: E402


class DummyEncoder(torch.nn.Module):
    def forward(self, graphs):
        if isinstance(graphs, list):
            outs = [torch.tensor(g.x.sum(axis=0), dtype=torch.float32) for g in graphs]
            return torch.stack(outs)
        return torch.tensor(graphs.x.sum(axis=0), dtype=torch.float32)


def make_graph() -> GraphData:
    x = np.ones((2, 3), dtype=np.float32)
    edge_index = np.zeros((2, 0), dtype=np.int64)
    return GraphData(x=x, edge_index=edge_index)


def test_train_multilabel_head(monkeypatch):
    graphs = [make_graph() for _ in range(10)]
    labels = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    enc = DummyEncoder()
    metrics = train_multilabel_head(
        graphs, labels, enc, epochs=1, batch_size=4, device="cpu"
    )

    assert metrics["tasks_evaluated"] == 2
    assert 0.0 <= metrics["val_roc_auc_macro"] <= 1.0
    assert 0.0 <= metrics["val_pr_auc_macro"] <= 1.0