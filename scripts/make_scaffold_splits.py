from __future__ import annotations

import argparse
import os

import pandas as pd

import logging

from data.scaffold_split import write_scaffold_splits

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Create scaffold splits from a single CSV/Parquet")
    ap.add_argument(
        "--input", required=True, help="Path to CSV/Parquet with a smiles column"
    )
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_parquet(args.input)

    write_scaffold_splits(
        df=df,
        smiles_col=args.smiles_col,
        out_dir=args.out_dir,
        fmt=args.format,
        train_frac=args.train,
        val_frac=args.val,
        seed=args.seed,
    )
    logger.info("Scaffold splits written to: %s", args.out_dir)
