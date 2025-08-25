# tests/test_scaffold_split_real_smiles.py
import sys
import types
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

pytest.importorskip("fastparquet")

import importlib.util, sys, pathlib, inspect, os

def load_scaffold_module_fresh():
    path = pathlib.Path(__file__).resolve().parents[1] / "data" / "scaffold_split.py"
    mod_name = "scaffold_under_test"

    # ensure we don't reuse a previous module object
    sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None, "spec_from_file_location returned None"
    assert spec.origin == str(path), f"Spec origin mismatch: {spec.origin}"
    assert spec.loader is not None, "No loader for scaffold_split.py (cannot execute)"

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    # PROVE it's the right scaffold
    assert hasattr(mod, "smiles_to_scaffold"), "smiles_to_scaffold missing"
    fn_file = inspect.getsourcefile(mod.smiles_to_scaffold)
    assert fn_file is not None
    # samefile handles case/sep differences on Windows
    assert os.path.samefile(fn_file, str(path)), f"Loaded from {fn_file}, expected {path}"
    # also confirm module file path
    assert os.path.samefile(mod.__file__, str(path)), f"Module file {mod.__file__}"

    return mod


def _dummy_scaffold(smiles: str) -> str:
    # Keep it simple & deterministic: bucket by first letter (upper-cased).
    # This makes "c1ccccc1" -> "C", which is fine for unit testing split logic.
    return smiles[0].upper()

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

def _import_ms():
    from data import scaffold_split as ms
    return ms

def test_scaffold_split_indices_with_real_smiles(monkeypatch):
    _stub_rdkit(monkeypatch)
    ms = load_scaffold_module_fresh()
    # Force deterministic, fake scaffold function (we're testing split logic, not chemistry)
    monkeypatch.setattr(ms, "smiles_to_scaffold", _dummy_scaffold)

    # Real-ish SMILES chosen to create 3 buckets by first letter: C(2), N(2), O(1)
    smiles = ["CCO", "CCN", "NCC", "NC=O", "O=C=O"]

    train1, val1, test1 = ms.scaffold_split_indices(smiles, train_frac=0.4, val_frac=0.2, seed=0)
    train2, val2, test2 = ms.scaffold_split_indices(smiles, train_frac=0.4, val_frac=0.2, seed=123)

    # Determinism across seeds (your function should be seed-invariant with this stub)
    assert np.array_equal(train1, train2)
    assert np.array_equal(val1, val2)
    assert np.array_equal(test1, test2)

    # Sizes should respect 2–2–1 due to bucket constraint
    assert len(train1) == 2
    assert len(val1) == 2
    assert len(test1) == 1

    # Disjoint & covering
    all_idx = set(train1.tolist()) | set(val1.tolist()) | set(test1.tolist())
    assert all_idx == set(range(len(smiles)))
    assert set(train1).isdisjoint(val1) and set(train1).isdisjoint(test1) and set(val1).isdisjoint(test1)

    # Each scaffold must appear in only one split
    split_for = {}
    for split_name, idxs in [("train", train1), ("val", val1), ("test", test1)]:
        for i in idxs:
            sc = _dummy_scaffold(smiles[i])
            assert split_for.get(sc, split_name) == split_name
            split_for[sc] = split_name

def test_write_scaffold_splits_with_real_smiles(tmp_path, monkeypatch):
    _stub_rdkit(monkeypatch)
    ms = load_scaffold_module_fresh()
    monkeypatch.setattr(ms, "smiles_to_scaffold", _dummy_scaffold)

    smiles = ["CCO", "CCN", "NCC", "NC=O", "O=C=O"]  # C(2), N(2), O(1)
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

    outdir = tmp_path
    ms.write_scaffold_splits(
        df, "smiles", str(outdir), fmt="parquet", train_frac=0.4, val_frac=0.2, seed=0
    )

    train_df = pd.read_parquet(outdir / "train" / "0000.parquet")
    val_df   = pd.read_parquet(outdir / "val"   / "0000.parquet")
    test_df  = pd.read_parquet(outdir / "test"  / "0000.parquet")

    # Check counts (don’t assume exact ordering of indices)
    assert len(train_df) == 2
    assert len(val_df) == 2
    assert len(test_df) == 1

    # Scaffolds shouldn’t be split across files
    def buckets(ss):
        return { _dummy_scaffold(s) for s in ss }

    train_sc = buckets(train_df["smiles"])
    val_sc   = buckets(val_df["smiles"])
    test_sc  = buckets(test_df["smiles"])

    assert train_sc.isdisjoint(val_sc)
    assert train_sc.isdisjoint(test_sc)
    assert val_sc.isdisjoint(test_sc)

    # And the union of scaffolds equals what we expect: {"C","N","O"}
    assert train_sc | val_sc | test_sc == {"C", "N", "O"}
