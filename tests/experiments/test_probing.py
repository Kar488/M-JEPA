import sys
import types
import numpy as np
import torch
import pytest

pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

# Provide a minimal stub for rdkit so experiments.probing can be imported
rdkit = types.ModuleType("rdkit")
Chem = types.ModuleType("Chem")
Scaffolds = types.ModuleType("Scaffolds")
MurckoScaffold = types.SimpleNamespace(GetScaffoldForMol=lambda mol: None)
Chem.MolFromSmiles = lambda s: None
Chem.MolToSmiles = lambda mol, isomericSmiles=True: ""
Scaffolds.MurckoScaffold = MurckoScaffold
rdkit.Chem = Chem
sys.modules.setdefault("rdkit", rdkit)
sys.modules.setdefault("rdkit.Chem", Chem)
sys.modules.setdefault("rdkit.Chem.Scaffolds", Scaffolds)
sys.modules.setdefault("rdkit.Chem.Scaffolds.MurckoScaffold", MurckoScaffold)
rdMolTransforms = types.ModuleType("rdMolTransforms")
Chem.rdMolTransforms = rdMolTransforms
sys.modules.setdefault("rdkit.Chem.rdMolTransforms", rdMolTransforms)

from experiments.probing import (
    _adj_to_edge_index,
    _to_pyg,
    compute_embeddings,
    linear_probe_classification,
    linear_probe_regression,
    clustering_quality,
)


def test_adj_to_edge_index():
    adj = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    edge_index = _adj_to_edge_index(adj)
    expected = torch.tensor([[0, 1, 2], [1, 2, 0]])
    assert torch.equal(edge_index, expected)


def test_to_pyg_from_adj():
    x = np.ones((3, 2))
    adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    g = types.SimpleNamespace(x=x, adj=adj, y=np.array([0]))
    out = _to_pyg(g)
    assert isinstance(out, Data)
    assert out.x.shape == (3, 2)
    assert out.edge_index.size(1) == 3
    assert out.y.item() == 0


def test_to_pyg_preserves_pos():
    x = np.ones((2, 4), dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    pos = np.array([[0.0, 1.0, 2.0], [1.0, 0.5, -0.5]], dtype=np.float32)
    g = types.SimpleNamespace(x=x, edge_index=edge_index, pos=pos)

    out = _to_pyg(g)
    assert isinstance(out, Data)
    assert out.pos is not None
    assert out.pos.shape == (2, 3)
    assert torch.allclose(out.pos, torch.as_tensor(pos))


def test_to_pyg_fills_missing_edge_attr():
    x = np.ones((2, 4), dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_attr = np.array([[1.0, 0.5], [0.3, 0.2]], dtype=np.float32)
    with_attr = types.SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr)
    without_attr = types.SimpleNamespace(x=x, edge_index=edge_index)

    pyg_with = _to_pyg(with_attr)
    dim = int(pyg_with.edge_attr.size(1))
    pyg_without = _to_pyg(without_attr, edge_attr_dim=dim)

    assert isinstance(pyg_without, Data)
    assert pyg_without.edge_attr is not None
    assert pyg_without.edge_attr.shape == (edge_index.shape[1], dim)
    assert torch.allclose(pyg_without.edge_attr, torch.zeros_like(pyg_without.edge_attr))


def test_compute_embeddings():
    g1 = Data(x=torch.ones((3, 4)), edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))
    g2 = Data(x=torch.zeros((2, 4)), edge_index=torch.tensor([[0, 1], [1, 0]]))
    dataset = types.SimpleNamespace(graphs=[g1, g2])

    class DummyEncoder(torch.nn.Module):
        def forward(self, x, adj):
            return x

    enc = DummyEncoder()
    H = compute_embeddings(dataset, enc, batch_size=1)
    assert H.shape == (2, 4)


def test_linear_probe_classification():
    np.random.seed(0)
    X = np.random.randn(20, 5)
    y = np.array([0, 1] * 10)
    metrics = linear_probe_classification(X, y, use_scaffold=False)
    assert set(metrics.keys()) == {
        "probe_roc_auc",
        "probe_pr_auc",
        "probe_acc",
        "probe_brier",
    }
    for v in metrics.values():
        assert isinstance(v, float)


def test_linear_probe_regression():
    np.random.seed(0)
    X = np.random.randn(20, 5)
    y = np.linspace(0, 1, 20)
    metrics = linear_probe_regression(X, y, use_scaffold=False)
    assert set(metrics.keys()) == {
        "probe_rmse",
        "probe_mae",
        "probe_r2",
    }
    for v in metrics.values():
        assert isinstance(v, float)


def test_clustering_quality_branches():
    small = np.random.randn(2, 3)
    assert clustering_quality(small)["cluster_silhouette"] == 0.0

    constant = np.ones((5, 3))
    assert clustering_quality(constant, n_clusters=4)["cluster_silhouette"] == 0.0

    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
    res = clustering_quality(X, n_clusters=2)
    sil = res["cluster_silhouette"]
    assert 0.0 <= sil <= 1.0


def test_clustering_quality_handles_missing_values():
    X = np.array(
        [
            [np.nan, 0.0],
            [0.1, 0.2],
            [1.0, np.nan],
            [1.1, 1.0],
            [2.0, 2.2],
        ]
    )

    res = clustering_quality(X, n_clusters=3)
    sil = res["cluster_silhouette"]
    assert np.isfinite(sil)
