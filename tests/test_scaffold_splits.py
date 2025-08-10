import sys
import types
import numpy as np
import pandas as pd


def _dummy_scaffold(smiles: str) -> str:
    return smiles[0]


def _stub_rdkit(monkeypatch):
    chem = types.ModuleType("Chem")
    scaffolds = types.ModuleType("Scaffolds")
    murcko = types.SimpleNamespace()
    scaffolds.MurckoScaffold = murcko
    rdkit = types.ModuleType("rdkit")
    rdkit.Chem = chem
    monkeypatch.setitem(sys.modules, "rdkit", rdkit)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", chem)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.Scaffolds", scaffolds)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.Scaffolds.MurckoScaffold", murcko)


def test_scaffold_split_indices_deterministic(monkeypatch):
    _stub_rdkit(monkeypatch)
    from data import moleculenet_scaffold as ms

    monkeypatch.setattr(ms, "smiles_to_scaffold", _dummy_scaffold)
    smiles = ["A1", "A2", "B1", "B2", "C1"]

    train1, val1, test1 = ms.scaffold_split_indices(smiles, train_frac=0.4, val_frac=0.2, seed=0)
    train2, val2, test2 = ms.scaffold_split_indices(smiles, train_frac=0.4, val_frac=0.2, seed=123)

    assert np.array_equal(train1, train2)
    assert np.array_equal(val1, val2)
    assert np.array_equal(test1, test2)

    assert train1.tolist() == [0, 1]
    assert val1.tolist() == [2, 3]
    assert test1.tolist() == [4]

    split_for = {}
    for split, idxs in zip(["train", "val", "test"], [train1, val1, test1]):
        for i in idxs:
            sc = _dummy_scaffold(smiles[i])
            if sc in split_for:
                assert split_for[sc] == split
            else:
                split_for[sc] = split


def test_write_scaffold_splits(tmp_path, monkeypatch):
    _stub_rdkit(monkeypatch)
    from data import moleculenet_scaffold as ms

    monkeypatch.setattr(ms, "smiles_to_scaffold", _dummy_scaffold)
    smiles = ["A1", "A2", "B1", "B2", "C1"]
    df = pd.DataFrame(
        {
            "smiles": smiles,
            "x": [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[5.0], [6.0]],
                [[7.0], [8.0]],
                [[9.0], [10.0]],
            ],
            "edge_index": [[[0, 1], [1, 0]]] * 5,
            "edge_attr": [[[1.0], [1.0]]] * 5,
            "y": list(range(5)),
        }
    )
    pd.options.io.parquet.engine = "fastparquet"
    ms.write_scaffold_splits(df, "smiles", str(tmp_path), fmt="parquet", train_frac=0.4, val_frac=0.2, seed=0)

    train_df = pd.read_parquet(tmp_path / "train" / "0000.parquet")
    val_df = pd.read_parquet(tmp_path / "val" / "0000.parquet")
    test_df = pd.read_parquet(tmp_path / "test" / "0000.parquet")

    assert train_df["smiles"].tolist() == ["A1", "A2"]
    assert val_df["smiles"].tolist() == ["B1", "B2"]
    assert test_df["smiles"].tolist() == ["C1"]
