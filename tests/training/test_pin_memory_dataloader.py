import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from data.mdataset import GraphData, GraphDataset
from training.unsupervised import (
    GraphBatch,
    _AugmentedPairDataset,
    _backoff_num_workers,
    _build_graph_dataloader,
    _collate_graph_pair,
    _maybe_pin,
)


def _make_graph(node_count: int) -> GraphData:
    x = np.random.randn(node_count, 4).astype(np.float32)
    edge_index = np.zeros((2, 0), dtype=np.int64)
    return GraphData(x=x, edge_index=edge_index)


def test_pinned_memory_dataloader_iterates():
    graphs = [_make_graph(4) for _ in range(6)]
    dataset = GraphDataset(graphs)

    augmenter = lambda g: (g, g)  # identity pair
    pair_dataset = _AugmentedPairDataset(dataset.graphs, augmenter)

    loader = _build_graph_dataloader(
        pair_dataset,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        collate_fn=_collate_graph_pair,
    )

    batches = 0
    for ctx_state, tgt_state in loader:
        ctx_batch = GraphBatch.from_packed(ctx_state)
        tgt_batch = GraphBatch.from_packed(tgt_state)
        assert isinstance(ctx_batch.x, torch.Tensor)
        assert isinstance(tgt_batch.x, torch.Tensor)
        batches += 1
        if batches >= 3:
            break

    assert batches > 0


def test_backoff_num_workers_progressively_reduces_workers():
    assert _backoff_num_workers(8) == 4
    assert _backoff_num_workers(3) == 1
    assert _backoff_num_workers(1) == 0
    assert _backoff_num_workers(0) == 0


def test_maybe_pin_handles_legacy_signature(monkeypatch):
    tensor = torch.zeros(1)
    calls = {"count": 0}

    def legacy_pin(self):
        calls["count"] += 1
        return self

    monkeypatch.setattr(torch.Tensor, "pin_memory", legacy_pin, raising=False)

    out = _maybe_pin(tensor, device="cuda")

    assert out is tensor
    assert calls["count"] == 2  # once with the device hint, once without
