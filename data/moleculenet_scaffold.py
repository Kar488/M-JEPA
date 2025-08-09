from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def smiles_to_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=True)
    except Exception:
        return ""


def scaffold_split_indices(
    smiles: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bemis–Murcko scaffold split (non‑random, grouped by scaffold)."""
    rng = np.random.RandomState(seed)
    by_scaffold: Dict[str, List[int]] = defaultdict(list)
    for i, sm in enumerate(smiles):
        by_scaffold[smiles_to_scaffold(sm)].append(i)

    # Sort scaffolds by descending frequency
    groups = sorted(by_scaffold.values(), key=lambda x: len(x), reverse=True)

    N = len(smiles)
    train_cut, val_cut = int(train_frac * N), int((train_frac + val_frac) * N)

    train_idx, val_idx, test_idx = [], [], []
    total = 0
    for g in groups:
        if total < train_cut:
            train_idx.extend(g)
        elif total < val_cut:
            val_idx.extend(g)
        else:
            test_idx.extend(g)
        total += len(g)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def write_scaffold_splits(
    df: pd.DataFrame,
    smiles_col: str,
    out_dir: str,
    fmt: str = "parquet",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> None:
    train_idx, val_idx, test_idx = scaffold_split_indices(
        df[smiles_col].astype(str).tolist(), train_frac, val_frac, seed
    )

    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sub = df.iloc[idx].reset_index(drop=True)
        path = f"{out_dir}/{split}"
        import os

        os.makedirs(path, exist_ok=True)
        out_file = f"{path}/0000.{fmt}"
        if fmt == "parquet":
            sub.to_parquet(out_file, index=False)
        elif fmt == "csv":
            sub.to_csv(out_file, index=False)
        else:
            raise ValueError("fmt must be parquet or csv")
