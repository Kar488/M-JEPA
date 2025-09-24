import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from data.mdataset import GraphData, GraphDataset
from training.unsupervised import (
    GraphBatch,
    _AugmentedPairDataset,
    _build_graph_dataloader,
    _collate_graph_pair,
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
