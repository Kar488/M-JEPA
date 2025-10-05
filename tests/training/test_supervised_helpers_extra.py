from types import SimpleNamespace

import numpy as np

import training.supervised as sup
from data.mdataset import GraphData


def _make_graph(num_nodes: int) -> GraphData:
    x = np.ones((num_nodes, 2), dtype=np.float32)
    edges = np.array([[i, (i + 1) % num_nodes] for i in range(num_nodes)], dtype=np.int64).T
    return GraphData(x=x, edge_index=edges)


def test_simple_pack_batch_shapes():
    dataset = SimpleNamespace(
        graphs=[_make_graph(2), _make_graph(3)],
        labels=np.array([0, 1], dtype=float),
    )
    batch_x, batch_adj, ptr, labels, extras = sup._simple_pack_batch(dataset, [0, 1], "classification")
    assert batch_x.shape == (5, 2)
    assert batch_adj.shape == (5, 5)
    assert ptr.tolist() == [0, 2, 5]
    assert labels.shape == (2,)
    assert extras["edge_index"].shape[0] == 2


def test_graph_batch_collator_uses_dataset_batch():
    dataset = SimpleNamespace(
        graphs=[_make_graph(2), _make_graph(3)],
        labels=np.array([0, 1], dtype=float),
    )
    collator = sup._GraphBatchCollator(dataset, "classification")
    batch_x, *_ = collator([0, 1])
    assert batch_x.shape[0] == 5


def test_stratified_split_balanced():
    labels = np.array([0, 1, 0, 1, 0, 1])
    indices = list(range(len(labels)))
    train, val, test = sup.stratified_split(indices, labels, train_frac=0.5, val_frac=0.25)
    assert set(train) and set(val) and set(test)
    assert {labels[i] for i in train} <= {0, 1}

