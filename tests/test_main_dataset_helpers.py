import numpy as np
import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("sklearn")
pytest.importorskip("torch_geometric")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from main import (
    _build_unlabeled_dataset_from_smiles,
    _ensure_labels_inplace_local,
    load_parquet_dataset,
    load_directory_dataset,
)
from data.mdataset import GraphDataset, GraphData

pd = pytest.importorskip("pandas")
try:
    import pyarrow  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    pytest.importorskip("fastparquet")


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
