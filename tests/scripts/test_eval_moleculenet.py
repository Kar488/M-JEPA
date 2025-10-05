import argparse
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("training.supervised")

from scripts import eval_moleculenet as em  # noqa: E402


def test_read_dataset_files_collects(monkeypatch):
    data = {
        "train": pd.DataFrame({"smiles": ["C"], "y": [0.1]}),
        "val": pd.DataFrame({"smiles": ["CC"], "y": [0.2]}),
        "test": pd.DataFrame({"smiles": ["CCC"], "y": [0.3]}),
    }

    def fake_exists(path):
        return any(str(path).endswith(f"{split}/0000.parquet") for split in data)

    def fake_read(path):
        for split, df in data.items():
            if str(path).endswith(f"{split}/0000.parquet"):
                return df
        raise FileNotFoundError

    monkeypatch.setattr(em.Path, "exists", fake_exists)
    monkeypatch.setattr(em.pd, "read_parquet", fake_read)

    combined = em._read_dataset_files(em.Path("/tmp/data"))
    assert list(combined["smiles"]) == ["C", "CC", "CCC"]


def test_read_dataset_files_missing(monkeypatch):
    monkeypatch.setattr(em.Path, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError):
        em._read_dataset_files(em.Path("/tmp/empty"))


def test_evaluate_dataset_aggregates(monkeypatch):
    class DummyDS:
        graphs = [object()]
        smiles = ["C"]

    df = pd.DataFrame({"smiles": ["C"], "y": [1.0]})

    def fake_load(name, root):
        return DummyDS(), ["y"], df

    monkeypatch.setattr(em, "load_moleculenet_dataset", fake_load)
    monkeypatch.setattr(
        em, "train_linear_head", lambda *a, **k: {"roc_auc": 0.5, "head": object()}
    )
    monkeypatch.setattr(em, "set_seed", lambda *a, **k: None)

    encoder = object()
    res = em.evaluate_dataset("task", "classification", encoder, Path("."), "cpu", 1, 1)
    assert res["roc_auc_mean"] == 0.5
    assert res["roc_auc_std"] == 0.0


def test_main_writes_results(tmp_path, monkeypatch):
    args = argparse.Namespace(
        encoder_checkpoint="enc.pt",
        data_root=str(tmp_path),
        output=str(tmp_path / "out.csv"),
        device="cpu",
        input_dim=1,
        hidden_dim=2,
        num_layers=1,
        gnn_type="gcn",
        devices=1,
        patience=1,
    )
    monkeypatch.setattr(em, "parse_args", lambda: args)

    class DummyEncoder:
        def __init__(self, **kw):
            pass

        def load_state_dict(self, state):
            pass

        def to(self, device):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(em, "GNNEncoder", DummyEncoder)
    monkeypatch.setattr(em, "DATASETS", {"foo": "classification"})
    monkeypatch.setattr(em, "evaluate_dataset", lambda *a, **k: {"roc_auc_mean": 0.5})
    em.main()
    assert Path(args.output).exists()


def test_load_moleculenet_dataset(monkeypatch):
    df = pd.DataFrame({"smiles": ["C", "CC"], "y1": [0, 1], "y2": [1, 0]})

    class FakeGraphDataset:
        def __init__(self, graphs, labels=None, smiles=None):
            self.graphs = graphs
            self.labels = labels
            self.smiles = smiles

        @classmethod
        def from_smiles_list(cls, smiles, labels=None):
            return cls([f"g_{i}" for i in range(len(smiles))], labels, smiles)

    monkeypatch.setattr(em, "_read_dataset_files", lambda path: df)
    monkeypatch.setattr(em, "GraphDataset", FakeGraphDataset)

    dataset, label_cols, table = em.load_moleculenet_dataset("tox21", em.Path("/tmp"))
    assert len(dataset.graphs) == 2
    assert set(label_cols) == {"y1", "y2"}
    assert table.equals(df)
