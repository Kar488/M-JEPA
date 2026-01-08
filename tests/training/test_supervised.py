from __future__ import annotations
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import pytest

torch = pytest.importorskip("torch")

import training.supervised as supervised_mod

from data.mdataset import GraphData, GraphDataset
from models.base import EncoderBase


from training.supervised import (
    _resolve_cuda_spawn_context,
    _pool_batch_embeddings,
    _build_layerwise_param_groups,
    stratified_split,
    train_linear_head,
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


class TinyEncoder(nn.Module):
    """Small encoder to exercise layer-wise parameter grouping."""

    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.stem = nn.Linear(hidden_dim, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):  # pragma: no cover - helpers only
        return self.encoder(self.stem(x)) * self.scale


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
def test_pool_batch_embeddings_accepts_float_ptr():
    node_embeddings = torch.tensor(
        [
            [1.0, 1.0],
            [3.0, 3.0],
            [2.0, 4.0],
            [6.0, 2.0],
            [5.0, 1.0],
        ],
        dtype=torch.float32,
    )
    batch_ptr = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float32)

    pooled = _pool_batch_embeddings(node_embeddings, batch_ptr)

    expected = torch.tensor([[2.0, 2.0], [13.0 / 3.0, 7.0 / 3.0]], dtype=torch.float32)
    assert torch.allclose(pooled, expected)


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


def test_train_linear_head_accepts_two_logit_head():
    np.random.seed(1)
    torch.manual_seed(1)
    labels = [0] * 8 + [1] * 8
    dataset = DummyDataset(labels)
    enc = DummyEncoder(4)

    class TwoLogitHead(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, 2)

        def forward(self, emb):  # noqa: D401, ARG002 - match signature used in training loop
            return self.linear(emb)

    head = TwoLogitHead(enc.hidden_dim)

    metrics = train_linear_head(
        dataset,
        enc,
        "classification",
        epochs=1,
        batch_size=4,
        lr=0.01,
        patience=0,
        device="cpu",
        head=head,
    )

    assert {"roc_auc", "pr_auc", "head"} <= metrics.keys()
    assert metrics["head"] is head


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


def test_train_linear_head_passes_pos_weight(monkeypatch):
    np.random.seed(0)
    torch.manual_seed(0)
    labels = [0] * 6 + [1] * 6
    dataset = DummyDataset(labels)
    enc = DummyEncoder(4)

    captured = {}
    real_loss_cls = supervised_mod.nn.BCEWithLogitsLoss

    class RecordingLoss(real_loss_cls):
        def __init__(self, *args, **kwargs):
            captured["pos_weight"] = kwargs.get("pos_weight")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(supervised_mod.nn, "BCEWithLogitsLoss", RecordingLoss)

    train_linear_head(
        dataset,
        enc,
        "classification",
        epochs=1,
        batch_size=4,
        lr=0.01,
        patience=0,
        device="cpu",
        pos_weight=torch.tensor([2.5], dtype=torch.float32),
    )

    assert "pos_weight" in captured
    assert torch.is_tensor(captured["pos_weight"])
    assert captured["pos_weight"].shape == (1,)
    assert captured["pos_weight"].device.type == "cpu"
    assert captured["pos_weight"].item() == pytest.approx(2.5)


def test_train_linear_head_switches_mode_when_metric_missing(monkeypatch):
    np.random.seed(0)
    torch.manual_seed(0)
    labels = [0, 1] * 6
    dataset = DummyDataset(labels)
    enc = DummyEncoder(4)

    def metrics_stub(y_true, y_pred):
        return {"roc_auc": float("nan")}

    monkeypatch.setattr(
        supervised_mod,
        "compute_classification_metrics",
        metrics_stub,
        raising=False,
    )

    record = {}

    class RecordingEarlyStopping:
        def __init__(self, patience, mode):
            record["init_mode"] = mode
            record["instance"] = self
            self.mode = mode
            self.best = None
            self.counter = 0
            self.patience = patience

        def step(self, value):
            record.setdefault("calls", []).append((self.mode, value))
            self.best = value
            self.counter = 0
            return False

    monkeypatch.setattr(
        supervised_mod,
        "EarlyStopping",
        RecordingEarlyStopping,
        raising=False,
    )

    train_linear_head(
        dataset,
        enc,
        "classification",
        epochs=1,
        batch_size=4,
        lr=0.01,
        patience=2,
        device="cpu",
        early_stop_metric="val_auc",
    )

    assert record["init_mode"] == "max"
    assert record["calls"] and record["calls"][0][0] == "min"
    assert record["instance"].mode == "min"


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


class MeanGraphEncoder(EncoderBase):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.seen_graphs: list[GraphData] = []

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:  # type: ignore[override]
        assert isinstance(g, GraphData)
        self.seen_graphs.append(g)
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        return x.mean(dim=0)


def test_encoder_base_handles_sequence_wrapped_graph():
    graph = GraphData(
        x=np.array([[1.0, 0.0], [3.0, 1.0]], dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    wrapped = [graph, {"label": 1}]

    encoder = MeanGraphEncoder()
    out = encoder([wrapped])

    assert out.shape == (1, graph.x.shape[1])
    assert encoder.seen_graphs == [graph]


def test_train_linear_head_raises_when_ddp_init_fails(monkeypatch, caplog):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    def _failing_init(*_args, **_kwargs):
        raise RuntimeError("ddp init failed")

    monkeypatch.setattr("utils.ddp.init_distributed", _failing_init, raising=False)

    np.random.seed(1)
    torch.manual_seed(1)
    labels = [0, 1, 0, 1]
    dataset = DummyDataset(labels)
    encoder = DummyEncoder(4)

    caplog.set_level(logging.WARNING, logger="training.supervised")

    with pytest.raises(RuntimeError, match="DDP initialisation failed"):
        train_linear_head(
            dataset,
            encoder,
            "classification",
            epochs=1,
            batch_size=2,
            lr=1e-3,
            patience=1,
            devices=2,
            device="cpu",
        )

    assert "refusing single-process fallback" in caplog.text
    assert os.environ.get("WORLD_SIZE") is None
    assert os.environ.get("LOCAL_WORLD_SIZE") is None
    assert os.environ.get("RANK") is None
    assert os.environ.get("LOCAL_RANK") is None


def test_train_linear_head_retries_with_gloo_on_duplicate_devices(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    init_calls: list[str | None] = []

    def _init_with_duplicate_failure(*_args, **_kwargs):
        init_calls.append(os.environ.get("DDP_FORCE_BACKEND"))
        if len(init_calls) == 1:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES contains duplicate entries (0, 0) but "
                "LOCAL_WORLD_SIZE=2. Each distributed rank must map to a unique GPU; "
                "adjust the mask or reduce --devices."
            )
        return True

    monkeypatch.setattr(
        "utils.ddp.init_distributed",
        _init_with_duplicate_failure,
        raising=False,
    )
    monkeypatch.setattr(
        supervised_mod.torch.distributed,
        "is_available",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        supervised_mod.torch.distributed,
        "is_initialized",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        supervised_mod.torch.distributed,
        "get_rank",
        lambda: 0,
        raising=False,
    )
    monkeypatch.setattr(
        supervised_mod.torch.distributed,
        "get_world_size",
        lambda: 2,
        raising=False,
    )
    monkeypatch.setattr(
        supervised_mod.torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    class _DummyDDP:  # noqa: N801 - mimics torch class name
        def __init__(self, module, **_kwargs):
            self.module = module

        @property
        def training(self):  # type: ignore[override]
            return self.module.training

        def __getattr__(self, name: str):
            return getattr(self.module, name)

        def __call__(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(
        supervised_mod.nn.parallel,
        "DistributedDataParallel",
        _DummyDDP,
        raising=False,
    )

    np.random.seed(1)
    torch.manual_seed(1)
    labels = [0, 1, 0, 1]
    dataset = DummyDataset(labels)
    encoder = DummyEncoder(4)

    metrics = train_linear_head(
        dataset,
        encoder,
        "classification",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        patience=1,
        devices=2,
        device="cpu",
    )

    assert len(init_calls) == 2
    assert init_calls[0] in (None, "")
    assert init_calls[1] == "gloo"
    assert "roc_auc" in metrics or "val_auc" in metrics
    assert metrics["head"].training  # type: ignore[index]


def test_train_linear_head_requires_ddp_for_multi_device(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    monkeypatch.setattr("utils.ddp.init_distributed", lambda *_a, **_k: False, raising=False)

    np.random.seed(1)
    torch.manual_seed(1)
    labels = [0, 1, 0, 1]
    dataset = DummyDataset(labels)
    encoder = DummyEncoder(4)

    with pytest.raises(RuntimeError, match="Requested multiple devices without active DDP"):
        train_linear_head(
            dataset,
            encoder,
            "classification",
            epochs=1,
            batch_size=2,
            lr=1e-3,
            patience=1,
            devices=2,
            device="cpu",
        )


def test_train_linear_head_uses_encode_graph_cache(monkeypatch):
    graphs = [
        GraphData(
            x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
        ),
        GraphData(
            x=np.array([[0.5, -1.0], [1.5, 2.0]], dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
        ),
        GraphData(
            x=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
        ),
    ]
    labels = np.linspace(0.0, 1.0, len(graphs)).astype(np.float32)
    dataset = GraphDataset(graphs, labels=labels)
    encoder = MeanGraphEncoder()

    def _forbid_pool(*args, **kwargs):  # noqa: ANN001, ANN002
        raise AssertionError("pooling path should not be used when encode_graph is available")

    monkeypatch.setattr(supervised_mod, "_pool_batch_embeddings", _forbid_pool)

    metrics = train_linear_head(
        dataset,
        encoder,
        "regression",
        epochs=0,
        batch_size=2,
        lr=0.01,
        patience=0,
        device="cpu",
        cache_graph_embeddings=True,
    )

    assert "head" in metrics
    seen_ids = {id(g) for g in encoder.seen_graphs}
    assert seen_ids == {id(g) for g in graphs}


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


def test_layerwise_param_groups_decay():
    encoder = TinyEncoder()
    groups = _build_layerwise_param_groups(encoder, base_lr=0.01, decay=0.5)
    lrs = [g.get("lr") for g in groups]
    assert max(lrs) == pytest.approx(0.01)
    assert min(lrs) < max(lrs)


def test_train_linear_head_handles_focal_dynamic_and_calibration():
    np.random.seed(2)
    torch.manual_seed(2)
    labels = [0] * 8 + [1] * 4
    dataset = DummyDataset(labels)
    enc = DummyEncoder(4)
    metrics = train_linear_head(
        dataset,
        enc,
        "classification",
        epochs=2,
        batch_size=2,
        lr=0.01,
        patience=1,
        device="cpu",
        use_focal_loss=True,
        dynamic_pos_weight=True,
        oversample_minority=True,
        calibrate_probabilities=True,
    )

    assert "roc_auc" in metrics
    assert metrics.get("calibration/method") in {None, "temperature", "isotonic"}
def test_resolve_cuda_spawn_context_prefers_spawn(monkeypatch):
    sentinel = object()

    class _DummyMP:
        @staticmethod
        def get_context(method):
            assert method == "spawn"
            return sentinel

    monkeypatch.setattr(supervised_mod.torch, "multiprocessing", _DummyMP(), raising=False)

    assert _resolve_cuda_spawn_context("cuda") is sentinel


def test_resolve_cuda_spawn_context_returns_none_on_failure(monkeypatch):
    class _DummyMP:
        @staticmethod
        def get_context(method):
            assert method == "spawn"
            raise RuntimeError("no spawn available")

    monkeypatch.setattr(supervised_mod.torch, "multiprocessing", _DummyMP(), raising=False)

    assert _resolve_cuda_spawn_context("cuda") is None
