"""
Tiny Tox21 case study runner.
- If you provide --csv, uses that; else uses samples/tox21_mini.csv
- Calls run_tox21_case_study if available; otherwise, runs a minimal fallback using GraphDataset.
"""
# tests/test_tox21_mini.py
from pathlib import Path
import numpy as np
import pytest

import importlib

dataset_module = importlib.import_module(f"data.mdataset")
GraphDataset = getattr(dataset_module, "GraphDataset")

def _find_csv():
    root = Path(__file__).resolve().parents[1]
    for p in [root / "samples" / "tox21_mini.csv", Path.cwd() / "samples" / "tox21_mini.csv"]:
        if p.exists():
            return p
    pytest.skip("samples/tox21_mini.csv not found")

def _require_label(csv, col="NR-AR"):
    import pandas as pd
    cols = set(pd.read_csv(csv, nrows=0).columns)
    if col not in cols:
        pytest.skip(f"Column '{col}' not found in {csv}")

def test_tox21_dataset_loads_and_labels(wb):
    csv = _find_csv()
    _require_label(csv, "NR-AR")
    
    ds = GraphDataset.from_csv(
        str(csv), smiles_col="smiles", label_col="NR-AR", cache_dir=None
    )

    assert len(ds.graphs) > 0
    assert ds.labels is not None
    assert ds.labels.shape[0] == len(ds.graphs)
    # labels should be binary 0/1 for NR-AR
    uniq = set(np.unique(ds.labels))
    assert uniq.issubset({0, 1}) and len(uniq) >= 1
    assert np.asarray(ds.labels).dtype.kind in "iu"

def test_tox21_minipipeline_rank_and_filter(wb):
    csv = _find_csv()
    _require_label(csv, "NR-AR")
    ds = GraphDataset.from_csv(
        str(csv), smiles_col="smiles", label_col="NR-AR", cache_dir=None
    )

    # Minimal scoring + “remove top fraction” pipeline (deterministic)
    rng = np.random.default_rng(0)
    scores = rng.random(len(ds.graphs))
    top_fraction = 0.2
    k = max(1, int(len(scores) * top_fraction))

    keep = np.ones(len(scores), dtype=bool)
    keep[np.argsort(scores)[-k:]] = False

    y = ds.labels
    mean_true = float(y.mean())
    mean_after_pred = float(y[keep].mean())

    # Basic sanity checks (no assumptions about improvement)
    assert 0.0 <= mean_true <= 1.0
    assert 0.0 <= mean_after_pred <= 1.0
    assert keep.sum() == len(ds.graphs) - k
