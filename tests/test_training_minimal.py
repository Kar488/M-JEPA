import sys
import types
from dataclasses import dataclass

import numpy as np
import torch

# Stubs for modules requiring optional dependencies
data_dataset = types.ModuleType("data.mdataset")

@dataclass
class GraphData:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray | None = None


class GraphDataset:
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

data_dataset.GraphData = GraphData
data_dataset.GraphDataset = GraphDataset
sys.modules["data.mdataset"] = data_dataset

utils_pooling = types.ModuleType("utils.pooling")

def global_mean_pool(node_embeddings, graph_ptr=None):
    if graph_ptr is None:
        return node_embeddings.mean(dim=0)
    start = 0
    out = []
    for end in graph_ptr:
        out.append(node_embeddings[start:end].mean(dim=0))
        start = int(end)
    return torch.stack(out, dim=0)

utils_pooling.global_mean_pool = global_mean_pool
sys.modules["utils.pooling"] = utils_pooling

# data.augment stub
data_augment = types.ModuleType("data.augment")

def apply_graph_augmentations(g, **kwargs):
    return g

data_augment.apply_graph_augmentations = apply_graph_augmentations
sys.modules["data.augment"] = data_augment

from training.unsupervised import train_jepa
from training.train_on_embeddings import train_linear_on_embeddings_classification


def make_graph():
    x = np.random.randn(3, 4).astype(np.float32)
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    edge_attr = np.ones((edge_index.shape[1], 1), dtype=np.float32)
    return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)


def test_train_jepa_minimal_epoch():
    g = make_graph()
    dataset = GraphDataset([g])

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x, adj, edge_attr=None):
            return self.lin(x)

    encoder = DummyEncoder()
    ema_encoder = DummyEncoder()

    class DummyPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, h):
            return self.lin(h)

    predictor = DummyPredictor()

    class DummyEMA:
        def update(self, model):
            pass

    ema = DummyEMA()

    losses = train_jepa(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema,
        epochs=1,
        batch_size=1,
        lr=1e-3,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
    )
    assert len(losses) == 1


def test_train_linear_on_embeddings():
    X = np.random.randn(8, 4)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    metrics = train_linear_on_embeddings_classification(X, y, max_iter=10)
    assert "roc_auc" in metrics
