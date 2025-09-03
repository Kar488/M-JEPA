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
from utils.logging import maybe_init_wandb

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

def metric(run, name: str, default=None):
    v = run.summary.get(name, None)
    if v is None:
        v = run.config.get(name, None)
    return v if v is not None else default

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

# Task + metric plan (robust to missing val_rmse)
def pick_primary_metric(runs, task: str, args) -> Tuple[str, bool]:
    # (name, maximize?) ordered by preference
    if task == "regression":
        prefs = [("val_rmse", False), ("rmse", False), ("rmse_mean", False),
                    ("probe_rmse_mean", False), ("metric", False)]
    else:
        prefs = [("val_auc", True), ("pr_auc", True), ("roc_auc", True),
                    ("metric", True)]
    for name, mx in prefs:
        if any(metric(r, name) is not None for r in runs):
            return name, mx
    # fallback to the task default even if empty; choose_best will raise
    return (args.reg_primary if task=="regression" else args.clf_primary,
            False if task=="regression" else True)

# ---------- main ----------

def main():
    APP_DIR  = need_env("APP_DIR")
    GRID_DIR = need_env("GRID_DIR")
    ENTITY   = need_env("WANDB_ENTITY")
    PROJECT  = need_env("WANDB_PROJECT")

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_id", "--sweep-id", dest="sweep_id",
                    help="entity/project/sweepid to read from; "
                         "defaults to $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_SWEEP_ID1")
    ap.add_argument("--out", "--out-csv", dest="out",
                default=os.path.join(GRID_DIR, "best_grid_config.json"),
                help="Write best run CONFIG to this JSON")
    ap.add_argument("--phase2_yaml", "--phase2-yaml", dest="phase2_yaml",
                    default=os.path.join(APP_DIR, "sweeps", "grid_sweep_phase2.yaml"),
                    help="Emit/overwrite Phase-2 sweep YAML here (default: $APP_DIR/sweeps/grid_sweep_phase2.yaml)")
    # phase-2 data roots (externalizable via CI YAML/env)
    ap.add_argument("--phase2_unlabeled_dir", "--phase2-unlabeled-dir",
                    dest="phase2_unlabeled_dir",
                    default=os.path.join(APP_DIR, "data", "ZINC-canonicalized"))
    ap.add_argument("--phase2_labeled_dir", "--phase2-labeled-dir",
                    dest="phase2_labeled_dir",
                    default=os.path.join(APP_DIR, "data", "katielinkmoleculenet_benchmark", "train"))

    # selection behavior
    ap.add_argument("--task", "--task-type", dest="task", choices=["auto","regression","classification"], default="auto")
    ap.add_argument("--tie_eps", "--tie-eps", dest="tie_eps", type=float, default=0.01)

    # metric names (override if logs differ)
    ap.add_argument("--reg_primary", "--reg-primary", dest="reg_primary", default="val_rmse")
    ap.add_argument("--reg_tb1", "--reg-tb1",  dest="reg_tb1", default="val_mae")
    ap.add_argument("--clf_primary", "--clf-primary", dest="clf_primary", default="val_auc")
    ap.add_argument("--clf_tb1", "--clf-tb1", dest="clf_tb1", default="val_brier")
    ap.add_argument("--clf_tb2", "--clf-tb2", dest="clf_tb2", default="val_pr_auc")

    # Phase-2 derivation
    ap.add_argument("--emit_bounds", "--emit-bounds", dest="emit_bounds", action="store_true",
                    help="Also derive top-K bounds and update Phase-2 YAML")
    ap.add_argument("--topk", "--top-k", dest="topk", type=int, default=20)
    ap.add_argument("--phase2_method", "--phase2-method", dest="phase2_method", default="bayes", choices=["bayes","random"])
    ap.add_argument("--phase2_metric", "--phase2-metric", dest="phase2_metric", default=None,
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
    primary, maximize = pick_primary_metric(runs, task, args)
    tiebreakers: List[Tuple[str,bool]] = []
    if task == "regression":
        if args.reg_tb1: tiebreakers.append((args.reg_tb1, False))
    else:
        if args.clf_tb1: tiebreakers.append((args.clf_tb1, False))
        if args.clf_tb2: tiebreakers.append((args.clf_tb2, True))

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
                    hi = max(float(hi) * 1.1, 1e-6)
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

        params["-labeled_dir"]      = {"value": "${env:PHASE2_LABELED_DI}"}
        params["-unlabeled_dir"]    = {"value": "${env:PHASE2_UNLABELED_DIR}"}

        # safety in case Schnet is chosen and 3D is not set
        try:
            gnn_val = (params.get("gnn_type", {}) or {}).get("value")
            if gnn_val == "schnet3d":
                params["add_3d"] = {"value": 1}
        except Exception:
            pass
        

        phase2_metric_name = args.phase2_metric or primary
        phase2_goal        = "maximize" if maximize else "minimize"

        # Clean Phase-2 space to match JEPA-only Bayes with ranges

        # --- Phase-2 policy tweaks to match JEPA-only Bayes with ranges ---
        # --- Winner-aware Phase-2 policy (JEPA or Contrastive) with ranges ---
        # Winner comes from env (set by run-grid after paired-effect); default to JEPA.
        winner = os.environ.get("METHOD_WINNER", "jepa")

        # If JEPA won, drop contrastive-only knobs (aug_* and temperature).
        # If Contrastive won, keep them (and optionally drop JEPA-only knobs if you add any later).
        if winner == "jepa":
            for k in list(params.keys()):
                if k.startswith("aug_") or k in ("temperature", "seed", "seeds"):
                    params.pop(k, None)
        elif winner == "contrastive":
            for k in list(params.keys()):
                if k in ("mask_ratio", "ema_decay"):
                    params.pop(k, None)

        # Suggest distributions for continuous ranges so Bayes can interpolate well.
        if isinstance(params.get("mask_ratio"), dict) and "min" in params["mask_ratio"]:
            params["mask_ratio"].setdefault("distribution", "uniform")
        if isinstance(params.get("learning_rate"), dict) and "min" in params["learning_rate"]:
            params["learning_rate"].setdefault("distribution", "log_uniform_values")

        # JEPA or Contrastive per Phase-1 winner
        training_method_param = {"value": winner}

        # Early termination to speed up Bayes search
        early_terminate_cfg = {"type": "hyperband", "min_iter": 3}

        # Initialise optional W&B run for grid search
        wb = maybe_init_wandb(
            enable=True,
            project=os.environ.get("WANDB_PROJECT", PROJECT),
            tags=["export_best"],
            config={
                "sweep_id": sweep_id,
                "task": "auto",
                "tie_eps": args.tie_eps,
                "phase2_emit_bounds": bool(args.emit_bounds),
                "phase2_method": args.phase2_method,
                "phase2_metric": args.phase2_metric or "",
                # anything else we want as metadata…
            },
            api_key=os.environ.get("WANDB_API_KEY"),
        )

        spec = {
            "program": "${env:APP_DIR}/scripts/train_jepa.py",
            "command": [
                "${interpreter}", "${program}", "sweep-run",
                "--unlabeled-dir", args.phase2_unlabeled_dir,
                 "--labeled-dir",   args.phase2_labeled_dir,
                "${args}"
            ],
            "method": args.phase2_method,                 # expect "bayes"
            "metric": {"name": phase2_metric_name, "goal": phase2_goal},
            "early_terminate": early_terminate_cfg,       # optional (matches pic)
            "parameters": {
                "training_method": training_method_param, # JEPA or Contrastive (winner)
                **params
            }
        }

        # log the winner to the run’s summary (one-shot; no step noise)
        # one-shot summary on this export step's run
        if wb:
            s = {
                "best_run_name": best.name,
                "best_run_id":   best.id,
            }
            pv = metric(best, primary)
            if pv is not None:
                s[primary] = float(pv)
            for n, _mx in tiebreakers:
                v = metric(best, n)
                if v is not None:
                    # try float; fall back to str for non-numeric types
                    try: s[n] = float(v)
                    except Exception: s[n] = str(v)

            wb.summary.update(s)
            wb.finish()

        os.makedirs(os.path.dirname(args.phase2_yaml) or ".", exist_ok=True)
        with open(args.phase2_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False)
            
        print(f"[export_best] Wrote Phase-2 sweep YAML to {args.phase2_yaml}")


if __name__ == "__main__":
    main()
