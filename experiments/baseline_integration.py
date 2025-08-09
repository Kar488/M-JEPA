from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from adapters.cli_runner import BaselineCLI
try:
    from adapters.native_adapter import NativeBaseline  # optional
    _HAS_NATIVE = True
except Exception:
    _HAS_NATIVE = False


def concat_dir_to_csv(dirpath: str, smiles_col: str, label_col: Optional[str], out_csv: str) -> str:
    files = [f for f in os.listdir(dirpath) if f.lower().endswith((".parquet", ".csv"))]
    files.sort()
    frames = []
    for f in files:
        p = os.path.join(dirpath, f)
        df = pd.read_parquet(p) if f.lower().endswith(".parquet") else pd.read_csv(p)
        cols = [smiles_col] + ([label_col] if (label_col and label_col in df.columns) else [])
        frames.append(df[cols])
    if not frames:
        raise FileNotFoundError(f"No CSV/Parquet files in {dirpath}")
    full = pd.concat(frames, axis=0, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_csv, index=False)
    return out_csv


def baseline_pretrain_and_embed(
    method: str,
    unlabeled_file: str,
    smiles_eval_file: str,
    cfg_path: str = "adapters/config.yaml",
    ckpt_filename: str = "best.ckpt",
    force_embed: bool = False,
    use_native: bool = False,
    native_repo: Optional[str] = None,
    native_train: Optional[dict] = None,   # {"module": "...", "function": "..."}
    native_embed: Optional[dict] = None,   # {"module": "...", "function": "..."}
) -> Tuple[str, str]:
    """
    Returns (checkpoint_path, embeddings_file). Tries native adapter if requested and available,
    otherwise falls back to CLI wrapper.
    """
    # Native path (in‑process) if available
    if use_native and _HAS_NATIVE and native_repo and native_train and native_embed:
        nb = NativeBaseline(native_repo, native_train, native_embed)
        ckpt_path = nb.train(unlabeled=unlabeled_file, method=method)
        emb_out   = nb.embed(checkpoint=ckpt_path, smiles_file=smiles_eval_file)
        return ckpt_path, emb_out

    # Default: CLI path
    cli = BaselineCLI(cfg_path)
    out_dir = cli.outputs_dir(method)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ckpt_path = os.path.join(out_dir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        cli.train(method, unlabeled=unlabeled_file, out_dir=out_dir)

    emb_name = Path(smiles_eval_file).stem + ".npy"
    emb_out = str(Path(out_dir) / "embeddings" / emb_name)
    if not os.path.exists(emb_out) or force_embed:
        cli.embed(method, ckpt_path=ckpt_path, smiles_file=smiles_eval_file, emb_out=emb_out)
    return ckpt_path, emb_out
