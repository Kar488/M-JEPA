import sys
import types
import numpy as np
import pandas as pd

from data import moleculenet_dc
import deepchem as dc


class DummyDataset:
    def __init__(self, ids, y):
        self.ids = ids
        self.y = y

def _ds_len(ds):
    # robust length for DeepChem Datasets
    for attr in ("ids", "y", "X"):
        if hasattr(ds, attr):
            return len(getattr(ds, attr))
    return None

def test_download_moleculenet_to_parquet(tmp_path, monkeypatch):
    # Force use of fastparquet to avoid missing engine
    pd.options.io.parquet.engine = "fastparquet"

    out = moleculenet_dc.download_moleculenet_to_parquet("esol", out_dir=str(tmp_path))

    tasks, (train, valid, test), _ = dc.molnet.load_delaney(featurizer="Raw")
    expected_tasks = ",".join(tasks)
    assert out["tasks"] == expected_tasks

    paths = {
        "train": tmp_path / "esol_train.parquet",
        "valid": tmp_path / "esol_valid.parquet",
        "test":  tmp_path / "esol_test.parquet",
    }
    dsets = {"train": train, "valid": valid, "test": test}

    for split, ds in dsets.items():
        df = pd.read_parquet(paths[split])

        # columns present
        assert "smiles" in df.columns
        for t in tasks:
            assert t in df.columns

        # row counts match (when determinable)
        n = _ds_len(ds)
        if n is not None:
            assert len(df) == n
        else:
            assert len(df) > 0
