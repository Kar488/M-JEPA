#!/usr/bin/env python3
"""
Step #2 helper:
- Read a W&B sweep (Phase-1 by default), pick best run with task-aware tie-breaks
- Write best config to $GRID_DIR/best_grid_config.json (for pretrain)
- Emit/overwrite $APP_DIR/sweeps/grid_sweep_phase2.yaml with narrowed Bayes spec
  (derived from top-K Phase-1 runs)

Notes:
- Input sweep defaults to $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_SWEEP_ID1
  but you can pass --sweep_id to point at any sweep (e.g., SWEEP_ID2).
- This script does NOT create a sweep; Step #3 will create/use WANDB_SWEEP_ID2.
"""

import argparse, json, math, os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import wandb
import yaml


# ---------- env helpers ----------

def need_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env: {name}")
    return v


# ---------- metric helpers ----------

def metric(run, name: str):
    v = run.summary.get(name, None)
    if v is None:
        v = run.config.get(name, None)
    return v


def detect_task(runs) -> str:
    have_auc = any(metric(r, "val_auc") is not None for r in runs)
    have_rmse = any(metric(r, "val_rmse") is not None for r in runs)
    if have_auc and not have_rmse:
        return "classification"
    if have_rmse and not have_auc:
        return "regression"
    return "classification" if have_auc else "regression"


def choose_best(runs, primary: str, maximize: bool, tie_eps: float,
                tiebreakers: List[Tuple[str, bool]]):
    cand = [r for r in runs if metric(r, primary) is not None]
    if not cand:
        raise RuntimeError(f"No runs have primary metric '{primary}'")

    keyf = (lambda r: -float(metric(r, primary, -math.inf))) if maximize \
           else (lambda r:  float(metric(r, primary,  math.inf)))
    cand.sort(key=keyf)

    best = cand[0]
    best_val = float(metric(best, primary))
    if maximize:
        thresh = best_val * (1.0 - tie_eps)
        tied = [r for r in cand if float(metric(r, primary)) >= thresh]
    else:
        thresh = best_val * (1.0 + tie_eps)
        tied = [r for r in cand if float(metric(r, primary)) <= thresh]

    for tb_name, tb_max in tiebreakers:
        tb = [r for r in tied if metric(r, tb_name) is not None]
        if len(tb) <= 1:
            continue
        tb.sort(key=(lambda r: -float(metric(r, tb_name, -math.inf))) if tb_max
                        else (lambda r:  float(metric(r, tb_name,  math.inf))))
        best = tb[0]
        best_val = float(metric(best, primary))
        if maximize:
            thresh = best_val * (1.0 - tie_eps)
            tied = [r for r in tb if float(metric(r, primary)) >= thresh]
        else:
            thresh = best_val * (1.0 + tie_eps)
            tied = [r for r in tb if float(metric(r, primary)) <= thresh]
    return best


def collect_topk(runs, primary: str, maximize: bool, k: int):
    have = [r for r in runs if metric(r, primary) is not None]
    have.sort(key=(lambda r: -float(metric(r, primary, -math.inf))) if maximize
                    else (lambda r:  float(metric(r, primary,  math.inf))))
    return have[:max(1, k)]


# ---------- bounds helpers ----------

def grab_cfg_vals(runs: List[Any], key: str) -> List[Any]:
    out = []
    for r in runs:
        v = r.config.get(key, None)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            out.append(v)
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(v)
    return out


def percentile_band(arr: List[Any]) -> Optional[Tuple[float, float]]:
    xs = [x for x in arr if isinstance(x, (int, float))]
    if not xs:
        return None
    lo, hi = np.percentile(xs, 10), np.percentile(xs, 90)
    return float(lo), float(hi)


def uniq_vals(arr: List[Any]) -> List[Any]:
    s = set()
    for v in arr:
        if isinstance(v, str):
            if v in ("0", "1"):
                s.add(int(v))
            else:
                try:
                    f = float(v)
                    s.add(int(f) if f.is_integer() else f)
                except Exception:
                    s.add(v)
        else:
            s.add(v)
    return sorted(s, key=lambda x: (str(type(x)), str(x)))


# ---------- main ----------

def main():
    APP_DIR  = need_env("APP_DIR")
    GRID_DIR = need_env("GRID_DIR")
    ENTITY   = need_env("WANDB_ENTITY")
    PROJECT  = need_env("WANDB_PROJECT")

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_id",
                    help="entity/project/sweepid to read from; "
                         "defaults to $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_SWEEP_ID1")
    ap.add_argument("--out", default=os.path.join(GRID_DIR, "best_grid_config.json"),
                    help="Write best run CONFIG to this JSON (default: $GRID_DIR/best_grid_config.json)")
    ap.add_argument("--phase2_yaml", default=os.path.join(APP_DIR, "sweeps", "grid_sweep_phase2.yaml"),
                    help="Emit/overwrite Phase-2 sweep YAML here (default: $APP_DIR/sweeps/grid_sweep_phase2.yaml)")

    # selection behavior
    ap.add_argument("--task", choices=["auto","regression","classification"], default="auto")
    ap.add_argument("--tie_eps", type=float, default=0.01)

    # metric names (override if logs differ)
    ap.add_argument("--reg_primary", default="val_rmse")
    ap.add_argument("--reg_tb1",     default="val_mae")
    ap.add_argument("--clf_primary", default="val_auc")
    ap.add_argument("--clf_tb1",     default="val_brier")
    ap.add_argument("--clf_tb2",     default="val_pr_auc")

    # Phase-2 derivation
    ap.add_argument("--emit_bounds", action="store_true",
                    help="Also derive top-K bounds and update Phase-2 YAML")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--phase2_method", default="bayes", choices=["bayes","random"])
    ap.add_argument("--phase2_metric", default=None,
                    help="Override Phase-2 metric name; defaults to task primary")

    args = ap.parse_args()

    sweep_id = args.sweep_id or f"{ENTITY}/{PROJECT}/{need_env('WANDB_SWEEP_ID1')}"
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    runs  = list(sweep.runs)
    if not runs:
        raise RuntimeError(f"No runs found in sweep {sweep_id}")

    # Task + metric plan
    task = args.task if args.task != "auto" else detect_task(runs)
    if task == "regression":
        primary, maximize = args.reg_primary, False
        tiebreakers: List[Tuple[str,bool]] = []
        if args.reg_tb1:
            tiebreakers.append((args.reg_tb1, False))   # mae: minimize
    else:
        primary, maximize = args.clf_primary, True
        tiebreakers = []
        if args.clf_tb1:
            tiebreakers.append((args.clf_tb1, False))   # brier: minimize
        if args.clf_tb2:
            tiebreakers.append((args.clf_tb2, True))    # pr_auc: maximize

    # Pick best
    best = choose_best(runs, primary, maximize, args.tie_eps, tiebreakers)

    # Log and write best config
    msg = f"[export_best] Best run={best.name} task={task} {primary}={metric(best, primary)}"
    for n,_mx in tiebreakers:
        v = metric(best, n)
        if v is not None:
            msg += f"  {n}={v}"
    print(msg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(best.config, f, indent=2)
    print(f"[export_best] Wrote best config to {args.out}")

    # Optionally derive narrowed Phase-2 ranges and write YAML for Step #3
    if args.emit_bounds:
        top = collect_topk(runs, primary, maximize, args.topk)

        # Numeric → p10–p90; discrete → unique sets
        numeric_keys = ["mask_ratio", "ema_decay", "learning_rate"]
        set_keys = [
            # model
            "hidden_dim","num_layers","gnn_type","contiguity","add_3d",
            # aug (contrastive only, harmless to include)
            "aug_rotate","aug_mask_angle","aug_dihedral",
            "aug_bond_deletion","aug_atom_masking","aug_subgraph_removal",
            # training length/scale
            "pretrain_batch_size","finetune_batch_size",
            "pretrain_epochs","finetune_epochs",
            # task/data
            "task_type","label_col",
            # dataset caps (if used in Phase-1)
            "sample_unlabeled","sample_labeled","n_rows_per_file",
        ]

        params: Dict[str, Any] = {}

        for k in numeric_keys:
            band = percentile_band(grab_cfg_vals(top, k))
            if band:
                lo, hi = band
                if lo == hi:
                    lo = float(lo) * 0.9
                    hi = float(hi) * 1.1 or 1e-6
                params[k] = {"min": float(lo), "max": float(hi)}

        for k in set_keys:
            arr = grab_cfg_vals(top, k)
            if not arr:
                continue
            vals = uniq_vals(arr)
            if len(vals) == 1:
                params[k] = {"value": vals[0]}
            else:
                params[k] = {"values": vals}

        phase2_metric_name = args.phase2_metric or primary
        phase2_goal        = "maximize" if maximize else "minimize"

        spec = {
            "program": "${env:APP_DIR}/scripts/train_jepa.py",
            "command": [
                "${interpreter}", "${program}", "sweep-run",
                "--unlabeled-dir", "${env:APP_DIR}/data/ZINC-canonicalized",
                "--labeled-dir",  "${env:APP_DIR}/data/katielinkmoleculenet_benchmark/train",
                "${args}"
            ],
            "method": args.phase2_method,
            "metric": {"name": phase2_metric_name, "goal": phase2_goal},
            "parameters": {
                "training_method": {"values": ["jepa","contrastive"]},
                **params
            }
        }

        os.makedirs(os.path.dirname(args.phase2_yaml) or ".", exist_ok=True)
        with open(args.phase2_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False)
        print(f"[export_best] Wrote Phase-2 sweep YAML to {args.phase2_yaml}")


if __name__ == "__main__":
    main()
