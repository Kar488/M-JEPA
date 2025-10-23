import inspect
import sys
import types

import numpy as np
import pytest

# Provide minimal stubs for heavy optional dependencies
try:  # pragma: no cover - optional dependency
    import rdkit  # noqa: F401
except Exception:  # pragma: no cover
    rdkit_stub = types.ModuleType("rdkit")
    chem_stub = types.ModuleType("Chem")
    scaffolds_stub = types.ModuleType("Scaffolds")
    scaffolds_stub.MurckoScaffold = types.SimpleNamespace(
        GetScaffoldForMol=lambda mol: None
    )
    chem_stub.Scaffolds = scaffolds_stub
    rdkit_stub.Chem = chem_stub
    sys.modules["rdkit"] = rdkit_stub
    sys.modules["rdkit.Chem"] = chem_stub
    sys.modules["rdkit.Chem.Scaffolds"] = scaffolds_stub

try:  # pragma: no cover - optional dependency
    import sklearn  # noqa: F401
except Exception:  # pragma: no cover
    sklearn_stub = types.ModuleType("sklearn")
    metrics_stub = types.ModuleType("metrics")

    def _dummy_metric(*args, **kwargs):
        return 0.0

    # populate common metrics
    for name in [
        "roc_auc_score",
        "average_precision_score",
        "brier_score_loss",
        "mean_absolute_error",
        "mean_squared_error",
        "r2_score",
    ]:
        setattr(metrics_stub, name, _dummy_metric)
    metrics_stub.__getattr__ = lambda name: _dummy_metric
    sklearn_stub.metrics = metrics_stub
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = metrics_stub

from data.mdataset import GraphData, GraphDataset
from main import (
    _build_unlabeled_dataset_from_smiles,
    _edge_dim_or_none,
    _ensure_labels_inplace_local,
)
from utils.dataset import load_directory_dataset, load_parquet_dataset

pd = pytest.importorskip("pandas")
try:
    import pyarrow  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    try:
        import fastparquet  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover
        pytest.skip(
            "pyarrow or fastparquet is required for parquet tests",
            allow_module_level=True,
        )


def test_build_unlabeled_dataset_fallback(monkeypatch):
    monkeypatch.delattr(GraphDataset, "from_smiles_list", raising=False)
    smiles = ["CCO", "CCN"]
    ds = _build_unlabeled_dataset_from_smiles(smiles)
    assert len(ds.graphs) == len(smiles)
    for g in ds.graphs:
        assert np.all(g.x == 1.0)
        n = g.x.shape[0]
        assert g.edge_index.shape == (2, 2 * (n - 1))


def test_ensure_labels_inplace_local_dtypes():
    g = GraphData(
        x=np.ones((2, 1), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    ds_cls = GraphDataset([g, g])
    _ensure_labels_inplace_local(ds_cls, "classification")
    assert ds_cls.labels.dtype == np.int64
    assert ds_cls.labels.tolist() == [0, 0]

    ds_reg = GraphDataset([g], labels=[1.5])
    _ensure_labels_inplace_local(ds_reg, "regression")
    assert ds_reg.labels.dtype == np.float32
    assert ds_reg.labels[0] == pytest.approx(1.5)


def test_load_parquet_dataset(tmp_path):
    df = pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0.0, 1.0]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    ds = load_parquet_dataset(str(path), label_col="label")
    assert len(ds.graphs) == 2
    assert ds.labels.tolist() == [0.0, 1.0]


def test_load_directory_dataset(tmp_path):
    df1 = pd.DataFrame({"smiles": ["CCO"], "label": [1.0]})
    df2 = pd.DataFrame({"smiles": ["CCN"], "label": [0.0]})
    df1.to_parquet(tmp_path / "a.parquet")
    df2.to_parquet(tmp_path / "b.parquet")
    ds = load_directory_dataset(str(tmp_path), label_col="label")
    assert len(ds.graphs) == 2


def test_graphdataset_interface():
    sig = inspect.signature(GraphDataset.__init__)
    assert "labels" in sig.parameters

    assert hasattr(GraphDataset, "from_parquet")
    assert callable(GraphDataset.from_parquet)

    assert hasattr(GraphDataset, "from_directory")
    assert callable(GraphDataset.from_directory)


def test_graphdataset_normalises_feature_dims():
    g0 = GraphData(
        x=np.ones((3, 7), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    g1 = GraphData(
        x=np.ones((2, 5), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )

    dataset = GraphDataset([g0, g1])

    assert dataset.graphs[0].x.shape[1] == 7
    assert dataset.graphs[1].x.shape[1] == 7
    assert np.all(dataset.graphs[1].x[:, 5:] == 0)


def test_edge_dim_or_none():
    g0 = GraphData(
        x=np.ones((1, 1), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    ds0 = GraphDataset([g0])
    assert _edge_dim_or_none(ds0) is None

    g1 = GraphData(
        x=np.ones((1, 1), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_attr=np.ones((0, 3), dtype=np.float32),
    )
    ds1 = GraphDataset([g1])
    assert _edge_dim_or_none(ds1) == 3
