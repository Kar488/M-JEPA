from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _ensure_dc():
    try:
        import deepchem as dc  # noqa: F401

        return True
    except Exception:
        return False


def _write_df(df: pd.DataFrame, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if out_path.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


def download_moleculenet_to_parquet(
    name: str, out_dir: str = "data/moleculenet", fmt: str = "parquet"
) -> Dict[str, str]:
    """
    Download using DeepChem and save train/valid/test as tabular files with SMILES + labels.
    Returns dict with file paths.
    """
    if not _ensure_dc():
        raise RuntimeError(
            "DeepChem not installed. Please `pip install deepchem` (and rdkit)."
        )

    import deepchem as dc

    name_low = name.lower()
    loaders = {
        "esol": dc.molnet.load_delaney,
        "freesolv": dc.molnet.load_freesolv,
        "lipo": dc.molnet.load_lipo,
        "bace": dc.molnet.load_bace_classification,
        "bbbp": dc.molnet.load_bbbp,
        "tox21": dc.molnet.load_tox21,
    }
    if name_low not in loaders:
        raise ValueError(
            f"Unsupported dataset {name}. Supported: {list(loaders.keys())}"
        )

    tasks, datasets, transformers = loaders[name_low](featurizer="Raw")  # Raw SMILES
    train, valid, test = datasets

    def _to_df(dset) -> pd.DataFrame:
        # dset.ids holds SMILES; dset.y is (N, T)
        import numpy as np

        cols = ["smiles"] + tasks
        y = dset.y if dset.y.ndim > 1 else dset.y.reshape(-1, 1)
        return pd.DataFrame(
            data=np.concatenate([np.array(dset.ids).reshape(-1, 1), y], axis=1),
            columns=cols,
        )

    df_tr, df_va, df_te = map(_to_df, (train, valid, test))

    out = {}
    ext = ".parquet" if fmt == "parquet" else ".csv"
    out["train"] = _write_df(df_tr, f"{out_dir}/{name}_train{ext}")
    out["valid"] = _write_df(df_va, f"{out_dir}/{name}_valid{ext}")
    out["test"] = _write_df(df_te, f"{out_dir}/{name}_test{ext}")
    out["tasks"] = ",".join(tasks)
    return out
