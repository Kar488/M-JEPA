import sys
import types
import numpy as np


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


def test_scaffold_split_assigns_by_scaffold(monkeypatch):
    _stub_rdkit(monkeypatch)
    from data import scaffold_split as ss

    monkeypatch.setattr(ss, "smiles_to_scaffold", lambda s: s[0])

    smiles = ["A1", "A2", "B1", "B2", "C1"]
    train, val, test = ss.scaffold_split(smiles, train_frac=0.4, val_frac=0.2, seed=0)

    assert sorted(train.tolist()) == [0, 1]
    assert sorted(val.tolist()) == [2, 3]
    assert test.tolist() == [4]

    split_for = {}
    for split, idxs in zip(["train", "val", "test"], [train, val, test]):
        for i in idxs:
            sc = smiles[i][0]
            if sc in split_for:
                assert split_for[sc] == split
            else:
                split_for[sc] = split

    # Deterministic for same seed
    tr2, va2, te2 = ss.scaffold_split(smiles, train_frac=0.4, val_frac=0.2, seed=0)
    assert np.array_equal(train, tr2)
    assert np.array_equal(val, va2)
    assert np.array_equal(test, te2)
