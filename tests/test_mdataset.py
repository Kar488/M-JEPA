import importlib
import random
import sys
import types

import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - torch not available
    torch = types.SimpleNamespace()
import pytest
import logging
# Check if RDKit is available
try:
    from rdkit import Chem
    RDKit_AVAILABLE = True
except ImportError:
    RDKit_AVAILABLE = False

    
# training.supervised imports data.scaffold_split which requires RDKit. Provide a
# minimal stub so we can import stratified_split without pulling in RDKit.
dummy_scaffold = types.ModuleType("data.scaffold_split")
dummy_scaffold.scaffold_split_indices = lambda *args, **kwargs: ([], [], [])
sys.modules.setdefault("data.scaffold_split", dummy_scaffold)

from training.supervised import stratified_split

# import GraphDataset and GraphData
from data import GraphDataset

def _make_synthetic_dataset():
    smiles = [
            "CCO",
            "CCN"
        ]
    labels = [0, 1]
    ds = GraphDataset.from_smiles_list(smiles, labels=labels, add_3d=False)
    return ds


def test_smiles_to_graph_fallback_features_edges():
    ds = _make_synthetic_dataset()
    g0, g1 = ds.graphs

    if RDKit_AVAILABLE:
        # For "CCO" (ethanol, C2H5OH): 3 heavy atoms + 6 hydrogens = 9 nodes
        assert g0.x.shape == (9, 4), f"Expected [9, 4] for CCO, got {g0.x.shape}"
        # Check node features: [atomic_num, degree, aromatic, hybridization]
        expected_types = [
            [6, 4, 0, 4],  # Carbon (sp3, degree 4 with hydrogens)
            [6, 4, 0, 4],  # Carbon (sp3, degree 4 with hydrogens)
            [8, 2, 0, 4],  # Oxygen (sp3, degree 2)
            [1, 1, 0, 0],  # Hydrogen
            [1, 1, 0, 0],  # Hydrogen
            [1, 1, 0, 0],  # Hydrogen
            [1, 1, 0, 0],  # Hydrogen
            [1, 1, 0, 0],  # Hydrogen
            [1, 1, 0, 0],  # Hydrogen
        ]
        assert np.allclose(g0.x, expected_types), f"Node features mismatch for CCO"
        assert g0.edge_index.shape[1] == 16, f"Expected 16 edges for CCO, got {g0.edge_index.shape[1]}"  # 8 bonds * 2 (undirected)
        assert g0.edge_attr.shape == (16, 7), f"Expected [16, 7] edge attributes, got {g0.edge_attr.shape}"

        # For "CCN" (ethylamine, C2H5NH2): 3 heavy atoms + 7 hydrogens = 10 nodes
        assert g1.x.shape == (10, 4), f"Expected [10, 4] for CCN, got {g1.x.shape}"
    else:
        # Fallback: graph for "CCO" -> 2 nodes chain
        expected_x0 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        expected_e0 = np.array([[0, 1], [1, 0]], dtype=np.int64)
        assert np.array_equal(g0.x, expected_x0)
        assert np.array_equal(g0.edge_index, expected_e0)
        assert g0.edge_attr is None

        # Fallback: graph for "CCN" -> 3 nodes chain
        expected_x1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        expected_e1 = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        assert np.array_equal(g1.x, expected_x1)
        assert np.array_equal(g1.edge_index, expected_e1)
        assert g1.edge_attr is None


def test_get_batch_splitting_and_labels():
    ds = _make_synthetic_dataset()
    X, A, ptr, labels = ds.get_batch([0, 1])

    if RDKit_AVAILABLE:
        # For "CCO" (9 nodes) + "CCN" (10 nodes) = 19 nodes, 4 features
        assert X.shape == (19, 4), f"Expected [19, 4] for batch, got {X.shape}"
        assert A.shape == (19, 19), f"Expected [19, 19] adjacency, got {A.shape}"
        assert torch.equal(ptr, torch.tensor([0, 9, 19], dtype=torch.long)), f"Expected [0, 9, 19] ptr, got {ptr.tolist()}"
        assert torch.equal(labels, torch.tensor([0.0, 1.0])), f"Expected [0.0, 1.0] labels, got {labels.tolist()}"
    else:
        # Fallback: 2 nodes (CCO) + 3 nodes (CCN) = 5 nodes, 2 features
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
    smiles = [
            "CCO",
            "CCN",
            "CCC",
            "c1ccccc1",
            "CC(=O)O",
            "CCOCC",
            ]
   
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


@pytest.mark.skipif(not RDKit_AVAILABLE, reason="RDKit not installed")
def test_smiles_to_graph_kekulize_warning(caplog):
    bad_smiles = "c1cccc1"
    with caplog.at_level(logging.WARNING, logger="data.mdataset"):
        g = GraphDataset.smiles_to_graph(bad_smiles)

    assert any("Kekulization failed" in rec.message for rec in caplog.records)

    n = len(bad_smiles)
    expected_x = np.stack(
        [np.arange(n, dtype=np.float32), (np.arange(n) % 3).astype(np.float32)],
        axis=1,
    )
    expected_e = np.stack(
        [
            np.concatenate([np.arange(n - 1), np.arange(1, n)]).astype(np.int64),
            np.concatenate([np.arange(1, n), np.arange(n - 1)]).astype(np.int64),
        ],
        axis=0,
    )

    assert np.array_equal(g.x, expected_x)
    assert np.array_equal(g.edge_index, expected_e)
    assert g.edge_attr is None

