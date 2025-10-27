import logging
import random
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:  # Determine whether RDKit is available without skipping the whole module
    import rdkit  # noqa: F401

    RDKit_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when RDKit is absent
    RDKit_AVAILABLE = False


# training.supervised imports data.scaffold_split which requires RDKit. Provide a
# minimal stub so we can import stratified_split without pulling in RDKit.
dummy_scaffold = types.ModuleType("data.scaffold_split")
dummy_scaffold.scaffold_split_indices = lambda *args, **kwargs: ([], [], [])
sys.modules.setdefault("data.scaffold_split", dummy_scaffold)

# import GraphDataset and GraphData
from data import GraphDataset  # noqa: E402
from data.mdataset import (
    EDGE_BASE_DIM,
    EDGE_TOTAL_DIM,
    GRAPH_CACHE_VERSION,
    GraphData,
    _fallback_graph_from_string,
)  # noqa: E402
from training.supervised import stratified_split  # noqa: E402


def _make_synthetic_dataset():
    smiles = ["CCO", "CCN"]
    labels = [0, 1]
    ds = GraphDataset.from_smiles_list(smiles, labels=labels, add_3d=False)
    return ds


def test_graphdataset_normalises_missing_edge_flags():
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    incomplete = GraphData(
        x=np.zeros((2, 1), dtype=np.float32),
        edge_index=edge_index,
        edge_attr=np.ones((2, EDGE_TOTAL_DIM - 1), dtype=np.float32),
    )
    reference = GraphData(
        x=np.zeros((2, 1), dtype=np.float32),
        edge_index=edge_index,
        edge_attr=np.arange(2 * EDGE_TOTAL_DIM, dtype=np.float32).reshape(
            2, EDGE_TOTAL_DIM
        ),
    )

    dataset = GraphDataset([incomplete, reference])

    assert dataset.edge_dim == EDGE_TOTAL_DIM
    assert incomplete.edge_attr.shape == (2, EDGE_TOTAL_DIM)
    # The padded column should be zeros, preserving original features.
    assert np.allclose(incomplete.edge_attr[:, :-1], 1.0)
    assert np.allclose(incomplete.edge_attr[:, -1], 0.0)


def test_graphdataset_pads_small_edge_attrs_to_base_width():
    edge_index = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.int64)
    raw_attr = np.full((3, EDGE_BASE_DIM - 2), 2.0, dtype=np.float64)
    graph = GraphData(
        x=np.zeros((2, 1), dtype=np.float32),
        edge_index=edge_index,
        edge_attr=raw_attr,
    )

    dataset = GraphDataset([graph])

    assert dataset.edge_dim == EDGE_BASE_DIM
    assert graph.edge_attr.shape == (3, EDGE_BASE_DIM)
    assert graph.edge_attr.dtype == np.float32
    assert np.allclose(graph.edge_attr[:, : raw_attr.shape[1]], 2.0)
    assert np.allclose(graph.edge_attr[:, raw_attr.shape[1] :], 0.0)


def test_schema_metadata_tracks_cache_version():
    dataset = _make_synthetic_dataset()

    meta = dataset.schema_metadata
    assert meta["cache_version"] == GRAPH_CACHE_VERSION

    legacy_meta = {k: v for k, v in meta.items() if k != "cache_version"}
    with pytest.raises(ValueError) as excinfo:
        dataset.validate_cached_schema(legacy_meta, source="legacy.pkl")

    assert "cache version" in str(excinfo.value)


def test_smiles_to_graph_add_3d_provides_pos():
    g = GraphDataset.smiles_to_graph("CCO", add_3d=True, random_seed=0)
    assert g.pos is not None
    assert g.pos.shape[1] == 3


def test_smiles_to_graph_add_3d_fills_missing_coords(monkeypatch):
    if not RDKit_AVAILABLE:
        pytest.skip("RDKit not installed")

    calls: list[tuple] = []

    def _always_fail(*args, **kwargs):
        calls.append((args, kwargs))
        return None, "failed"

    monkeypatch.setattr(
        "data.mdataset._generate_conformer_coords",
        _always_fail,
    )

    g = GraphDataset.smiles_to_graph("CCO", add_3d=True, random_seed=0)

    assert calls, "expected conformer generator to be invoked"
    assert g.pos is not None
    assert g.pos.shape == (g.x.shape[0], 3)
    assert np.allclose(g.pos, 0.0)


def test_graphdataset_backfills_missing_pos_when_any_present():
    edge_index = np.zeros((2, 0), dtype=np.int64)
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

    with_pos = GraphData(
        x=np.zeros((2, 1), dtype=np.float32),
        edge_index=edge_index,
        pos=pos,
    )
    missing_pos = GraphData(
        x=np.zeros((3, 1), dtype=np.float32),
        edge_index=edge_index,
    )

    dataset = GraphDataset([with_pos, missing_pos])

    assert dataset.graphs[0].pos is not None
    assert dataset.graphs[0].pos.dtype == np.float32
    assert np.allclose(dataset.graphs[0].pos, pos.astype(np.float32))

    assert dataset.graphs[1].pos is not None
    assert dataset.graphs[1].pos.shape == (3, 3)
    assert np.allclose(dataset.graphs[1].pos, 0.0)


def test_fallback_graph_adds_pos_when_requested():
    word = "fallback"
    g = _fallback_graph_from_string(word, add_pos=True)
    n = max(2, min(10, len(word)))

    assert g.pos is not None
    assert g.pos.shape == (n, 3)
    assert g.x.shape == (n, 5)
    assert np.allclose(g.pos[:, 0], np.arange(n, dtype=np.float32))
    assert np.allclose(g.x[:, -3:], g.pos)


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
        assert np.allclose(g0.x, expected_types), "Node features mismatch for CCO"
        assert (
            g0.edge_index.shape[1] == 16
        ), f"Expected 16 edges for CCO, got {g0.edge_index.shape[1]}"
        assert g0.edge_attr.shape == (
            16,
            7,
        ), f"Expected [16, 7] edge attributes, got {g0.edge_attr.shape}"

        # For "CCN" (ethylamine, C2H5NH2): 3 heavy atoms + 7 hydrogens = 10 nodes
        assert g1.x.shape == (10, 4), f"Expected [10, 4] for CCN, got {g1.x.shape}"
    else:
        # Fallback graphs deterministically form linear chains whose size depends on the
        # SMILES length.  Verify the construction for the two molecules we use here.
        def _chain_expectations(length: int) -> tuple[np.ndarray, np.ndarray]:
            n = max(2, min(10, length))
            positions = np.arange(n, dtype=np.float32)
            feats = np.stack([positions, (positions % 3).astype(np.float32)], axis=1)
            rows = np.concatenate([np.arange(n - 1), np.arange(1, n)])
            cols = np.concatenate([np.arange(1, n), np.arange(n - 1)])
            edge_index = np.stack([rows.astype(np.int64), cols.astype(np.int64)], axis=0)
            return feats, edge_index

        expected_x0, expected_e0 = _chain_expectations(len("CCO"))
        expected_x1, expected_e1 = _chain_expectations(len("CCN"))

        assert np.array_equal(g0.x, expected_x0)
        assert np.array_equal(g0.edge_index, expected_e0)
        assert g0.edge_attr is None

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
        assert torch.equal(
            ptr, torch.tensor([0, 9, 19], dtype=torch.long)
        ), f"Expected [0, 9, 19] ptr, got {ptr.tolist()}"
        assert torch.equal(
            labels, torch.tensor([0.0, 1.0])
        ), f"Expected [0.0, 1.0] labels, got {labels.tolist()}"
    else:
        n0 = max(2, min(10, len("CCO")))
        n1 = max(2, min(10, len("CCN")))
        total_nodes = n0 + n1

        assert X.shape == (total_nodes, 2)
        assert A.shape == (total_nodes, total_nodes)

        expected_A = torch.zeros((total_nodes, total_nodes), dtype=torch.float32)
        # Fill block-diagonal adjacency for each linear chain graph
        offsets = [0, n0]
        for offset, size in zip(offsets, (n0, n1)):
            for idx in range(size - 1):
                expected_A[offset + idx, offset + idx + 1] = 1.0
                expected_A[offset + idx + 1, offset + idx] = 1.0

        assert torch.equal(A, expected_A)
        assert torch.equal(ptr, torch.tensor([0, n0, total_nodes], dtype=torch.long))
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
    train, val, test = stratified_split(
        indices, ds.labels, train_frac=0.5, val_frac=0.25
    )

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
