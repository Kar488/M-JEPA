import importlib
import random
import sys
import types

import numpy as np
import torch

# training.supervised imports data.scaffold_split which requires RDKit. Provide a
# minimal stub so we can import stratified_split without pulling in RDKit.
dummy_scaffold = types.ModuleType("data.scaffold_split")
dummy_scaffold.scaffold_split = lambda *args, **kwargs: ([], [], [])
sys.modules.setdefault("data.scaffold_split", dummy_scaffold)

from training.supervised import stratified_split

# import GraphDataset and GraphData
mdataset = importlib.import_module('data.mdataset')
GraphDataset = getattr(mdataset, 'GraphDataset')
GraphData = getattr(mdataset, 'GraphData')


def _make_synthetic_dataset():
    smiles = ['aa', 'bbb']
    labels = [0, 1]
    ds = GraphDataset.from_smiles_list(smiles, labels=labels, add_3d=False)
    return ds


def test_smiles_to_graph_fallback_features_edges():
    ds = _make_synthetic_dataset()
    g0, g1 = ds.graphs

    # graph for 'aa' -> 2 nodes chain
    expected_x0 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    expected_e0 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    assert np.array_equal(g0.x, expected_x0)
    assert np.array_equal(g0.edge_index, expected_e0)
    assert g0.edge_attr is None

    # graph for 'bbb' -> 3 nodes chain
    expected_x1 = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32
    )
    expected_e1 = np.array([[0, 1, 1, 2], [1, 2, 0, 1]], dtype=np.int64)
    assert np.array_equal(g1.x, expected_x1)
    assert np.array_equal(g1.edge_index, expected_e1)
    assert g1.edge_attr is None


def test_get_batch_splitting_and_labels():
    ds = _make_synthetic_dataset()
    X, A, ptr, labels = ds.get_batch([0, 1])

    assert X.shape == (5, 2)
    assert A.shape == (5, 5)
    expected_A = torch.tensor(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(A, expected_A)
    assert torch.equal(ptr, torch.tensor([0, 2, 5], dtype=torch.long))
    assert torch.equal(labels, torch.tensor([0.0, 1.0]))


def test_stratified_split_balanced():
    smiles = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']
    labels = np.array([0, 0, 0, 1, 1, 1])
    ds = GraphDataset.from_smiles_list(smiles, labels=labels)
    indices = list(range(len(ds)))

    random.seed(0)
    train, val, test = stratified_split(indices, ds.labels, train_frac=0.5, val_frac=0.25)

    # expected two graphs per split and coverage of both classes
    assert len(train) == len(val) == len(test) == 2
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train + val + test) == set(indices)
    for subset in [train, val, test]:
        assert set(ds.labels[subset]) == {0, 1}

