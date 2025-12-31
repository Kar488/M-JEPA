# tests/utils/test_bond_feats.py
import types
import numpy as np
import pytest 
import torch
rdkit = pytest.importorskip("rdkit", reason="RDKit required")
from rdkit import Chem  # type: ignore
if not hasattr(Chem, "BondType"):
    pytest.skip("RDKit missing bond enums", allow_module_level=True)
import utils.bond_feats as bf
attach_bond_features_from_smiles = bf.attach_bond_features_from_smiles


def test_attach_builds_edges_and_features_for_benzene(monkeypatch):
    # Ensure the module-level 'np' name is available
    monkeypatch.setattr(bf, "np", np, raising=False)
    # Provide a directed edge_index so the function only aligns features (avoids local-np branch)
    g = types.SimpleNamespace(edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64), edge_attr=None)
    out = attach_bond_features_from_smiles(g, "c1ccccc1")  # benzene
    assert out.edge_index.shape[0] == 2
    E = out.edge_index.shape[1]
    assert out.edge_attr.shape == (E, 13)

    # For benzene, bonds are aromatic and in a ring
    # One-hot bond type: AROMATIC is index 3 in the first 4 dims
    # Flags (indices 4..6): [conj, in_ring, aromatic]
    assert np.all(out.edge_attr[:, 3] == 1), "expected aromatic one-hot"
    assert np.all(out.edge_attr[:, 5] == 1), "expected in-ring flag"
    assert np.all(out.edge_attr[:, 6] == 1), "expected aromatic flag"


def test_aligns_existing_edges_double_bond(monkeypatch):
    monkeypatch.setattr(bf, "np", np, raising=False)
    # With an explicit directed edge_index, we must align features to it
    g = types.SimpleNamespace()
    g.edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)  # two directed edges
    out = attach_bond_features_from_smiles(g, "C=C")
    assert out.edge_attr.shape == (2, 13)
    # Double bond → index 1 in the first 4 dims
    assert np.all(out.edge_attr[:, 1] == 1)


def test_attach_respects_target_edge_dim_and_sets_flag(monkeypatch):
    monkeypatch.setattr(bf, "np", np, raising=False)
    pos = np.random.rand(2, 3).astype(np.float32)
    g = types.SimpleNamespace(
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_attr=None,
        pos=pos,
    )
    out = attach_bond_features_from_smiles(
        g,
        "C=C",
        target_edge_dim=18,
    )
    assert out.edge_attr.shape == (2, 18)
    # Column 7 (0-based) stores the has_3d flag when padding to 18 dims
    assert np.allclose(out.edge_attr[:, 7], 1.0)
    # Geometry section (columns 8 onwards) stays zero by default
    assert np.allclose(out.edge_attr[:, 8:], 0.0)
