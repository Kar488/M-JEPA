
import numpy as np
import torch
import torch.nn as nn

from training.supervised import stratified_split, train_linear_head


class DummyEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.hidden_dim = dim
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.linear(x)


class TinyGraph:
    def __init__(self, feat: np.ndarray):
        self.feat = torch.tensor(feat, dtype=torch.float32)

    def to_tensors(self):
        return self.feat.unsqueeze(0), torch.zeros((1, 1), dtype=torch.float32)


class DummyDataset:
    def __init__(self, labels, feat_dim=4):
        self.labels = np.array(labels, dtype=np.float32)
        self.graphs = [TinyGraph(np.random.randn(feat_dim)) for _ in labels]
        self.smiles = None

    def __len__(self):
        return len(self.graphs)

    def get_batch(self, indices):
        xs = []
        adjs = []
        for idx in indices:
            x, adj = self.graphs[idx].to_tensors()
            xs.append(x)
            adjs.append(adj)
        batch_x = torch.cat(xs, dim=0)
        batch_adj = torch.block_diag(*adjs)
        ptr = torch.arange(0, len(indices) + 1, dtype=torch.long)
        batch_labels = torch.tensor(self.labels[indices], dtype=torch.float32)
        return batch_x, batch_adj, ptr, batch_labels


def test_stratified_split_balanced():
    idx = list(range(8))
    labels = np.array([0, 1] * 4)
    tr, val, te = stratified_split(idx, labels, 0.5, 0.25)
    assert len(tr) == 4 and len(val) == 2 and len(te) == 2
    assert sorted(tr + val + te) == idx


def test_stratified_split_single_class():
    orig = list(range(10))
    labels = np.zeros(10)
    tr, val, te = stratified_split(orig.copy(), labels, 0.6, 0.2)
    assert len(tr) + len(val) + len(te) == 10
    assert set(tr + val + te) == set(orig)



def test_train_linear_head_classification():
    np.random.seed(0)
    torch.manual_seed(0)
    labels = [0] * 10 + [1] * 10
    dataset = DummyDataset(labels)
    enc = DummyEncoder(4)
    metrics = train_linear_head(
        dataset,
        enc,
        "classification",
        epochs=2,
        batch_size=4,
        lr=0.01,
        patience=1,
        device="cpu",
    )
    assert {"roc_auc", "pr_auc", "head"} <= metrics.keys()
    assert isinstance(metrics["head"], nn.Module)


def test_train_linear_head_regression():
    np.random.seed(0)
    torch.manual_seed(0)
    labels = np.linspace(0.0, 1.0, 8)
    dataset = DummyDataset(labels)
    enc = DummyEncoder(4)
    metrics = train_linear_head(
        dataset,
        enc,
        "regression",
        epochs=1,
        batch_size=4,
        lr=0.01,
        patience=0,
        device="cpu",
    )
    assert {"rmse", "mae", "head"} <= metrics.keys()
    assert isinstance(metrics["head"], nn.Module)