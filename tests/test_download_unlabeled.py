import pandas as pd 
import sys
import types
import pytest
from unittest.mock import Mock
import requests

# Minimal torch stub for modules that import it
torch_stub = types.SimpleNamespace()
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", torch_stub)

from scripts.download_unlabeled import (
    save_parquet,
    save_shards,
    stream_zinc,
    stream_pubchem,
)


def test_save_shards(tmp_path):
    smiles = ["C", "O"]
    save_shards(smiles, tmp_path, shard_size=1)
    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files == ["0000.parquet", "0001.parquet"]
    df = pd.read_parquet(tmp_path / "0000.parquet")
    assert set(df.columns) == {"smiles", "x", "edge_index", "edge_attr"}


def test_save_parquet(tmp_path):
    smiles = ["C", "O"]
    out_file = tmp_path / "unlabelled.parquet"
    save_parquet(smiles, out_file)
    assert out_file.exists()
    df = pd.read_parquet(out_file)
    assert len(df) == 2

def test_stream_zinc(monkeypatch):
    responses = [
        Mock(status_code=200, text="id1 C\nid2 O\n"),
        Mock(status_code=200, text=""),
    ]

    def fake_get(url, timeout):
        return responses.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)

    gen = stream_zinc(batch_size=2, start_page=5, sleep=0)
    page, smiles = next(gen)
    assert page == 5
    assert smiles == ["C", "O"]
    with pytest.raises(StopIteration):
        next(gen)


def test_stream_pubchem(monkeypatch):
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


def test_save_parquet_error(tmp_path):
    smiles = ["C", "O"]
    out_file = tmp_path / "unlabelled.parquet"
    
    # Simulate an error in saving
    with pytest.raises(Exception):
        save_parquet(smiles, out_file, raise_error=True)
    
    # Ensure file was not created
    assert not out_file.exists()   

def test_save_shards_error(tmp_path):
    smiles = ["C", "O"]
    
    # Simulate an error in saving
    with pytest.raises(Exception):
        save_shards(smiles, tmp_path, shard_size=1, raise_error=True)
    
    # Ensure no files were created
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 0