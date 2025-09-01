import json
import sys
import types
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from scripts import download_unlabeled
from scripts.download_unlabeled import (
    save_parquet,
    save_shards,
    stream_pubchem,
    stream_zinc,
)

torch = pytest.importorskip("torch")


def require_parquet_engine():
    try:
        import pyarrow  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover
        try:
            import fastparquet  # type: ignore  # noqa: F401
        except Exception:  # pragma: no cover
            pytest.skip("pyarrow or fastparquet is required for parquet support")


@pytest.fixture(autouse=True)
def mock_graphdataset(monkeypatch):
    import numpy as np

    class DummyGraph:
        def __init__(self):
            self.x = np.zeros((1, 1))
            self.edge_index = np.zeros((2, 0), dtype=int)
            self.edge_attr = None

    def fake_from_smiles_list(smiles_list, *args, **kwargs):
        graphs = [DummyGraph() for _ in smiles_list]
        return types.SimpleNamespace(smiles=smiles_list, graphs=graphs)

    dummy_ds = types.SimpleNamespace(
        from_smiles_list=staticmethod(fake_from_smiles_list)
    )
    monkeypatch.setattr(download_unlabeled, "GraphDataset", dummy_ds, raising=False)


def test_stream_zinc_success(monkeypatch):
    responses = [
        Mock(status_code=200, text="id1 C\nid2 O\n"),
        Mock(status_code=200, text=""),
    ]
    monkeypatch.setattr(requests, "get", lambda url, timeout: responses.pop(0))

    gen = stream_zinc(batch_size=2, start_page=5, sleep=0)
    page, smiles = next(gen)
    assert page == 5
    assert smiles == ["C", "O"]
    with pytest.raises(StopIteration):
        next(gen)


def test_stream_zinc_http_error(monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda url, timeout: Mock(status_code=500, text="err")
    )
    gen = stream_zinc(batch_size=2, sleep=0)
    with pytest.raises(StopIteration):
        next(gen)


def test_stream_pubchem_success(monkeypatch):
    responses = [
        Mock(status_code=200, text="C\nO\n"),
        Mock(status_code=200, text=""),
    ]
    monkeypatch.setattr(requests, "get", lambda url, timeout: responses.pop(0))

    gen = stream_pubchem(batch_size=2, start_cid=10, sleep=0)
    cid, smiles = next(gen)
    assert cid == 10
    assert smiles == ["C", "O"]
    with pytest.raises(StopIteration):
        next(gen)


def test_stream_pubchem_http_error(monkeypatch):
    responses = [
        Mock(status_code=500, text=""),
        Mock(status_code=200, text="N\n"),
        Mock(status_code=200, text=""),
    ]
    monkeypatch.setattr(requests, "get", lambda url, timeout: responses.pop(0))

    gen = stream_pubchem(batch_size=1, start_cid=1, sleep=0)
    cid, smiles = next(gen)
    assert cid == 1
    assert smiles == ["N"]
    with pytest.raises(StopIteration):
        next(gen)


def test_save_shards(tmp_path):
    require_parquet_engine()
    smiles = ["C", "O", "N"]
    save_shards(smiles, tmp_path, shard_size=2)
    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files == ["0000.parquet", "0001.parquet"]
    df0 = pd.read_parquet(tmp_path / "0000.parquet")
    df1 = pd.read_parquet(tmp_path / "0001.parquet")
    assert df0["smiles"].tolist() == ["C", "O"]
    assert df1["smiles"].tolist() == ["N"]


def test_save_parquet(tmp_path):
    require_parquet_engine()
    out_file = tmp_path / "mols.parquet"
    save_parquet(["C", "O"], out_file)
    assert out_file.exists()
    df = pd.read_parquet(out_file)
    assert df["smiles"].tolist() == ["C", "O"]
    assert set(df.columns) == {"smiles", "x", "edge_index", "edge_attr"}


def test_cli_main(monkeypatch, tmp_path):
    require_parquet_engine()

    def fake_stream_zinc(batch_size, start_page=1, sleep=0.5):
        yield 1, ["C"]

    def fake_stream_pubchem(batch_size, start_cid=1, sleep=0.5):
        yield 1, ["O"]

    monkeypatch.setattr(download_unlabeled, "stream_zinc", fake_stream_zinc)
    monkeypatch.setattr(download_unlabeled, "stream_pubchem", fake_stream_pubchem)

    out_file = tmp_path / "out.parquet"
    args = [
        "download_unlabeled.py",
        "--out-root",
        str(tmp_path),
        "--out-file",
        str(out_file),
        "--total",
        "2",
        "--batch-size",
        "1",
        "--sleep",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", args)
    download_unlabeled.main()

    assert out_file.exists()
    df = pd.read_parquet(out_file)
    assert df["smiles"].tolist() == ["C", "O"]
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress == {"zinc_page": 2, "pubchem_cid": 2}
