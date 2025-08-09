import numpy as np

from data.dataset import GraphData
from training.unsupervised import _batch_iter, _mask_subgraph


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
    batches = list(_batch_iter(graphs, batch_size=2))
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == len(graphs)

    ctx, tgt = _mask_subgraph(graphs[0], mask_ratio=0.5, contiguous=False)
    assert ctx.x.shape[0] + tgt.x.shape[0] == graphs[0].x.shape[0]
    if ctx.edge_index.size > 0:
        assert ctx.edge_index.max() < ctx.x.shape[0]
    if tgt.edge_index.size > 0:
        assert tgt.edge_index.max() < tgt.x.shape[0]
