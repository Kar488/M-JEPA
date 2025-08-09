from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def smiles_to_scaffold(smiles: str) -> str:
    """Convert a SMILES string to its Bemis–Murcko scaffold SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=True)
    except Exception:
        return ""


def scaffold_split(
    smiles: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test by Bemis–Murcko scaffolds.

    Scaffolds are grouped and assigned to splits in order of decreasing
    group size.  This ensures molecules sharing a scaffold are placed in
    the same split, reducing scaffold leakage.
    """
    rng = np.random.RandomState(seed)
    by_scaffold: Dict[str, List[int]] = defaultdict(list)
    for i, sm in enumerate(smiles):
        by_scaffold[smiles_to_scaffold(sm)].append(i)

    groups = sorted(by_scaffold.values(), key=lambda x: len(x), reverse=True)

    N = len(smiles)
    train_cut = int(train_frac * N)
    val_cut = int((train_frac + val_frac) * N)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    total = 0
    for g in groups:
        rng.shuffle(g)
        if total < train_cut:
            train_idx.extend(g)
        elif total < val_cut:
            val_idx.extend(g)
        else:
            test_idx.extend(g)
        total += len(g)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)
