import numpy as np

from data.mdataset import GraphData
from training.unsupervised import GraphBatch, _build_graph_dataloader
from data.augment import mask_subgraph, generate_views


def _make_graph(n_nodes: int) -> GraphData:
    x = np.arange(n_nodes * 2, dtype=np.float32).reshape(n_nodes, 2)
    edges = []
    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.ones((edge_index.shape[1], 1), dtype=np.float32)
    return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)


def test_batch_iter_and_mask_subgraph():
    graphs = [_make_graph(4), _make_graph(3), _make_graph(5)]
    loader = _build_graph_dataloader(
        graphs,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )
    batches = [GraphBatch.from_packed(b) for b in loader]
    assert len(batches) == 2
    assert sum(batch.num_graphs for batch in batches) == len(graphs)

    ctx, tgt = generate_views(
        graphs[0],
        structural_ops=[lambda g: mask_subgraph(g, mask_ratio=0.5, contiguous=False)],
    )
    assert ctx.x.shape[0] + tgt.x.shape[0] == graphs[0].x.shape[0]
    if ctx.edge_index.size > 0:
        assert ctx.edge_index.max() < ctx.x.shape[0]
    if tgt.edge_index.size > 0:
        assert tgt.edge_index.max() < tgt.x.shape[0]
