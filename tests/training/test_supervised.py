from __future__ import annotations
import random

import numpy as np
import torch
import torch.nn as nn
import pytest

torch = pytest.importorskip("torch")

from training.supervised import (
    stratified_split,
    train_linear_head,
    _pool_batch_embeddings,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor

class DummyEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.hidden_dim = dim
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:  # noqa: ARG002
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


def test_pool_batch_embeddings_handles_variable_sizes():
    node_emb = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    batch_ptr = torch.tensor([0, 2, 5], dtype=torch.long)

    pooled = _pool_batch_embeddings(node_emb, batch_ptr)
    expected = torch.stack(
        (
            node_emb[:2].mean(dim=0),
            node_emb[2:].mean(dim=0),
        )
    )

    assert torch.allclose(pooled, expected)


def test_pool_batch_embeddings_validates_ptr_lengths():
    node_emb = torch.zeros((4, 2))
    batch_ptr = torch.tensor([0, 3, 6], dtype=torch.long)

    with pytest.raises(ValueError, match="batch_ptr does not describe"):
        _pool_batch_embeddings(node_emb, batch_ptr)


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


def test_train_linear_head_respects_max_batches(monkeypatch):
    np.random.seed(0)
    torch.manual_seed(0)

    labels = [0, 1] * 10
    dataset = DummyDataset(labels)
    enc = DummyEncoder(4)

    step_counter = {"count": 0}
    real_adam = torch.optim.Adam

    def counting_adam(params, lr=0.01, *args, **kwargs):
        opt = real_adam(params, lr=lr, *args, **kwargs)
        original_step = opt.step

        def _wrapped_step(*a, **k):
            step_counter["count"] += 1
            return original_step(*a, **k)

        opt.step = _wrapped_step  # type: ignore[assignment]
        return opt

    monkeypatch.setattr(torch.optim, "Adam", counting_adam)

    train_linear_head(
        dataset,
        enc,
        "classification",
        epochs=3,
        batch_size=2,
        lr=0.01,
        patience=0,
        device="cpu",
        max_batches=3,
    )

    assert step_counter["count"] == 3


def test_train_linear_head_caches_embeddings(monkeypatch):
    import training.supervised as sup_mod

    original_encode = sup_mod._encode_graph
    call_counter = {"count": 0}

    def counting_encode(*args, **kwargs):
        call_counter["count"] += 1
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(sup_mod, "_encode_graph", counting_encode)

    labels = [0, 1] * 8  # Balanced dataset with enough samples

    def _run(cache_flag: bool) -> int:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        call_counter["count"] = 0
        dataset = DummyDataset(labels)
        encoder = DummyEncoder(4)
        train_linear_head(
            dataset,
            encoder,
            "classification",
            epochs=2,
            batch_size=4,
            lr=0.01,
            patience=0,
            device="cpu",
            cache_graph_embeddings=cache_flag,
        )
        return call_counter["count"]

    calls_without_cache = _run(False)
    calls_with_cache = _run(True)

    assert calls_with_cache < calls_without_cache


def test_train_linear_head_recovers_from_emfile(monkeypatch):
    import training.supervised as sup_mod

    def _tensor_to(self, *args, **kwargs):  # noqa: ARG002
        return self

    def _module_to(self, *args, **kwargs):  # noqa: ARG002
        return self

    monkeypatch.setattr(torch.Tensor, "to", _tensor_to)
    monkeypatch.setattr(torch.nn.Module, "to", _module_to)
    monkeypatch.setattr(DummyEncoder, "parameters", lambda self, recurse=True: iter(()))

    def fake_move(batch, device, non_blocking):  # noqa: ARG001
        if len(batch) == 5:
            return batch
        batch_x, batch_adj, batch_ptr, batch_labels = batch
        return batch_x, batch_adj, batch_ptr, batch_labels, {}

    monkeypatch.setattr(sup_mod, "_move_batch_to_device", fake_move)

    failure = {"raised": False}
    original_encode = sup_mod._encode_graph

    def flaky_encode(*args, **kwargs):
        if not failure["raised"]:
            failure["raised"] = True
            raise RuntimeError("Too many open files encountered during pinning")
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(sup_mod, "_encode_graph", flaky_encode)

    labels = [0, 1] * 6
    dataset = DummyDataset(labels)
    encoder = DummyEncoder(4)

    metrics = train_linear_head(
        dataset,
        encoder,
        "classification",
        epochs=1,
        batch_size=4,
        lr=0.01,
        patience=0,
        device="cuda",
    )

    assert failure["raised"]
    assert "head" in metrics
