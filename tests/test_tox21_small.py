"""
Tiny Tox21 case study runner.
- If you provide --csv, uses that; else uses samples/tox21_mini.csv
- Calls run_tox21_case_study if available; otherwise, runs a minimal fallback using GraphDataset.
"""

import argparse
from pathlib import Path

import numpy as np

from data.dataset import GraphDataset

from utils.logging import maybe_init_wandb

try:
    from experiments.case_study import run_tox21_case_study

    HAS_CASE = True
except Exception:
    HAS_CASE = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="samples/tox21_mini.csv")
    p.add_argument("--task", type=str, default="NR-AR")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--top_fraction", type=float, default=0.2)
    args = p.parse_args()
    wb = maybe_init_wandb(enable=False)

    csv = Path(args.csv)
    if not csv.exists():
        raise SystemExit(
            f"CSV not found: {csv}. Provide --csv path to your Tox21 file."
        )

    if HAS_CASE:
        wb.log({"mode": "case"})
        triple = run_tox21_case_study(
            tox21_csv=str(csv),
            task=args.task,
            add_3d=False,
            pretrain_epochs=args.epochs,
            finetune_epochs=args.epochs,
            device=args.device,
            top_fraction=args.top_fraction,
        )
        wb.log(
            {
                "tox21_mean_true": triple[0],
                "tox21_mean_random_after": triple[1],
                "tox21_mean_predicted_after": triple[2],
            }
        )
        return

    # Fallback: minimal pipeline just to validate IO
    wb.log({"mode": "fallback"})
    ds = GraphDataset.from_csv(
        str(csv), smiles_col="smiles", label_col=args.task, cache_dir="cache/tox21_tiny"
    )
    # Fake a 'predicted toxicity' score to verify ranking path
    rng = np.random.default_rng(42)
    scores = rng.random(len(ds.graphs))
    k = max(1, int(len(scores) * args.top_fraction))
    keep_mask = np.ones(len(scores), dtype=bool)
    keep_mask[np.argsort(scores)[-k:]] = False
    y = ds.labels if ds.labels is not None else rng.integers(0, 2, size=len(ds.graphs))
    mean_true = float(y.mean())
    mean_after = float(y[keep_mask].mean())
    wb.log(
        {
            "mean_true": mean_true,
            "mean_after_random": mean_true,
            "mean_after_pred": mean_after,
        }
    )


if __name__ == "__main__":
    main()
