import sys
import types
import numpy as np
import pandas as pd

from data import moleculenet_dc


class DummyDataset:
    def __init__(self, ids, y):
        self.ids = ids
        self.y = y


def test_download_moleculenet_to_parquet(tmp_path, monkeypatch):
    # Force use of fastparquet to avoid missing engine
    pd.options.io.parquet.engine = "fastparquet"

    # Pretend DeepChem is installed
    monkeypatch.setattr(moleculenet_dc, "_ensure_dc", lambda: True)

    train = DummyDataset(["C", "CC"], np.array([[1.0], [2.0]]))
    valid = DummyDataset(["O"], np.array([[3.0]]))
    test = DummyDataset(["N"], np.array([[4.0]]))
    tasks = ["prop"]

    def load_delaney(featurizer="Raw"):
        return tasks, (train, valid, test), []

    dc_stub = types.SimpleNamespace(molnet=types.SimpleNamespace(load_delaney=load_delaney))
    monkeypatch.setitem(sys.modules, "deepchem", dc_stub)

    out = moleculenet_dc.download_moleculenet_to_parquet("esol", out_dir=str(tmp_path))

    assert set(out.keys()) == {"train", "valid", "test", "tasks"}
    assert out["tasks"] == "prop"

    train_df = pd.read_parquet(tmp_path / "esol_train.parquet")
    valid_df = pd.read_parquet(tmp_path / "esol_valid.parquet")
    test_df = pd.read_parquet(tmp_path / "esol_test.parquet")

    assert train_df["smiles"].tolist() == ["C", "CC"]
    assert train_df["prop"].tolist() == [1.0, 2.0]
    assert valid_df["smiles"].tolist() == ["O"]
    assert test_df["smiles"].tolist() == ["N"]
