#!/usr/bin/env python3
"""
Pull top-k configs from a Phase-2 sweep, re-run each on extra seeds,
and save a summary JSON with averaged metrics and 95% CI per config.

Usage:
  python recheck_topk_from_wandb.py \
    --sweep "entity/project/abcdef" \
    --project mjepa --group "${WANDB_RUN_GROUP}" \
    --metric val_rmse --direction min \
    --topk 5 --extra_seeds 3 \
    --program "${env:APP_DIR}/scripts/train_jepa.py" \
    --subcmd "sweep-run" \
    --unlabeled-dir "${env:APP_DIR}/data/ZINC-canonicalized" \
    --labeled-dir   "${env:APP_DIR}/data/katielinkmoleculenet_benchmark/train" \
    --out "${GRID_DIR}/recheck_summary.json"
"""
import argparse, json, os, math, time, tempfile, pathlib
from typing import List, Dict, Any, Tuple
import numpy as np
import wandb
import subprocess
import shlex

def metric_of(run, name, default=None):
    v = run.summary.get(name, None)
    if v is None: v = run.config.get(name, None)
    return v if v is not None else default

def pick_topk(sweep, metric: str, maximize: bool, k: int):
    runs = list(sweep.runs)
    have = [r for r in runs if metric_of(r, metric) is not None]
    have.sort(key=(lambda r: -float(metric_of(r, metric, -math.inf))) if maximize
                    else (lambda r:  float(metric_of(r, metric,  math.inf))))
    return have[:max(1, k)]

def run_once(mm, program, subcmd, cfg: Dict[str, Any], seed: int,
             unlabeled: str, labeled: str, log_dir: str,
             project: str, group: str) -> int:
    import os, shlex, subprocess

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if "WANDB_PROJECT" not in env and project:
        env["WANDB_PROJECT"] = project
    if "WANDB_RUN_GROUP" not in env and group:
        env["WANDB_RUN_GROUP"] = group
    env.setdefault("WANDB_JOB_TYPE", "recheck")

    # Base command
    args = [
        mm, "run", "-n", "mjepa", "env", "PYTHONUNBUFFERED=1",
        "python", "-u", program, subcmd,
        "--unlabeled-dir", unlabeled,
        "--labeled-dir",   labeled,
    ]

    # Force training method explicitly (default to jepa)
    method = str(cfg.get("training_method", "jepa")).lower()
    args += ["--training-method", method]

    # sweep-run supported flags (hyphen form)
    ALLOWED = {
        "task-type", "label-col",
        "gnn-type", "hidden-dim", "num-layers", "contiguity", "add-3d",
        "learning-rate", "pretrain-batch-size", "finetune-batch-size",
        "pretrain-epochs", "finetune-epochs",
        "max-pretrain-batches", "max-finetune-batches",
        "sample-unlabeled", "sample-labeled",
        "time-budget-mins",
        "aug-bond-deletion", "aug-atom-masking", "aug-subgraph-removal",
        "aug-rotate", "aug-mask-angle", "aug-dihedral",
        "prefetch-factor", "pin-memory", "persistent-workers",
        "bf16", "num-workers", "devices", "device",
        "mask-ratio", "ema-decay",      # JEPA-only
        "temperature",                  # Contrastive-only
        "cache-dir", "use-wandb", "wandb-project", "wandb-tags",
        "target-pretrain-samples",
    }
    MAP = {
        "pretrain_bs":  "pretrain-batch-size",
        "finetune_bs":  "finetune-batch-size",
        "lr":           "learning-rate",
        "label_col":    "label-col",
        "gnn_type":     "gnn-type",
        "add_3d":       "add-3d",
        "contiguous":   "contiguity",   # CLI expects --contiguity
        # long names fall through
    }
    DROP = {
        "augmentations", "pair_id", "pair_key", "_wandb", "wandb_version",
        "seed", "name", "id", "group",
    }

    forwarded = []  # for debug print below

    for k, v in cfg.items():
        if k in DROP or v is None:
            continue
        if isinstance(v, (dict, list, tuple)):
            continue
        if k == "training_method":
            continue

        cli_key = MAP.get(k, k).replace("_", "-")
        if cli_key not in ALLOWED:
            continue

        flag = f"--{cli_key}"
        if isinstance(v, bool):
            args += [flag, "1" if v else "0"]
        else:
            args += [flag, str(v)]
        forwarded.append((flag, v))

    # Seed per recheck
    args += ["--seed", str(seed)]

    # Make sure log dir exists; write the command at the top of the log
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"recheck_{method}_seed{seed}.log")

    # stdout: show what we're about to run
    print("[recheck] cmd:", " ".join(map(shlex.quote, args)), flush=True)
    if forwarded:
        print("[recheck] forwarded flags:", forwarded, flush=True)

    with open(log, "w", encoding="utf-8") as f:
        f.write("[recheck] cmd: " + " ".join(map(shlex.quote, args)) + "\n")
        if forwarded:
            f.write("[recheck] forwarded flags: " + repr(forwarded) + "\n")
        p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    return p.wait()

def ci95(xs: List[float]) -> Tuple[float, float]:
    bs = [np.mean(np.random.choice(xs, size=len(xs), replace=True)) for _ in range(4000)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--project", default=os.getenv("WANDB_PROJECT"))
    ap.add_argument("--group",   default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--metric", default=os.getenv("PHASE2_METRIC","val_rmse"))
    ap.add_argument("--direction", choices=["min","max"], default="min")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--extra_seeds", type=int, default=3)
    ap.add_argument("--program", required=True)
    ap.add_argument("--subcmd", default="sweep-run")
    ap.add_argument("--unlabeled-dir", "--unlabeled_dir", "--unlabeled",
                dest="unlabeled_dir", required=True)
    ap.add_argument("--labeled-dir", "--labeled_dir", "--labeled",
                    dest="labeled_dir", required=True)
    ap.add_argument("--mm", default=os.environ.get("MMBIN","micromamba"))
    ap.add_argument("--log_dir", default=os.environ.get("LOG_DIR","./logs"))
    # Pick a writable default at runtime (cwd or temp) if GRID_DIR is missing or not writable
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("[recheck] args parsed:", flush=True)
    print("  sweep      =", args.sweep, flush=True)
    print("  project    =", args.project, flush=True)
    print("  group      =", args.group, flush=True)
    print("  metric     =", args.metric, flush=True)
    print("  direction  =", args.direction, flush=True)
    print("  topk       =", args.topk, flush=True)
    print("  extra_seeds=", args.extra_seeds, flush=True)
    print("  program    =", args.program, flush=True)
    print("  subcmd     =", args.subcmd, flush=True)
    print("  unlabeled  =", args.unlabeled, flush=True)
    print("  labeled    =", args.labeled, flush=True)
    print("  mm         =", args.mm, flush=True)
    print("  log_dir    =", args.log_dir, flush=True)
    print("  out        =", args.out, flush=True)

    # --- resolve a safe, writable output path and ensure parent exists ---
    def _safe_out(user_out: str | None) -> str:
        if user_out:
            out_path = pathlib.Path(user_out)
        else:
            base = os.environ.get("GRID_DIR") or os.getcwd()
            out_path = pathlib.Path(base) / "recheck_summary.json"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # also check writability
            testfile = out_path.parent / ".writetest"
            with open(testfile, "w", encoding="utf-8") as tf:
                tf.write("ok")
            testfile.unlink(missing_ok=True)
        except Exception:
            # fall back to temp dir on permission/creation error
            out_path = pathlib.Path(tempfile.gettempdir()) / (out_path.name or "recheck_summary.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        return str(out_path)

    args.out = _safe_out(args.out)

    # --- resolve a safe, writable log_dir too ---
    def _safe_dir(d: str | None, fallback_name: str = "recheck_logs") -> str:
        base = pathlib.Path(d or "./logs")
        try:
            base.mkdir(parents=True, exist_ok=True)
            test = base / ".writetest"
            with open(test, "w", encoding="utf-8") as tf:
                tf.write("ok")
            test.unlink(missing_ok=True)
            return str(base)
        except Exception:
            tmp = pathlib.Path(tempfile.gettempdir()) / fallback_name
            tmp.mkdir(parents=True, exist_ok=True)
            return str(tmp)

    args.log_dir = _safe_dir(args.log_dir)

    api = wandb.Api()
    sweep = api.sweep(args.sweep)
    maximize = (args.direction == "max")
    top = pick_topk(sweep, args.metric, maximize, args.topk)

    print(f"[recheck] picked top {len(top)} from sweep: {[r.id for r in top]}", flush=True)

    def _kv(d, keys=("training_method","gnn_type","hidden_dim","num_layers","contiguity","mask_ratio","ema_decay","temperature","learning_rate")):
        out = {}
        for k in keys:
            v = d.get(k)
            if v is not None:
                out[k] = v
        return out

    print("[recheck] top configs (key fields):", flush=True)
    for r in top:
        print(" ", r.id, _kv(r.config), flush=True)

    # Extract the raw config dict for each top run
    tops: List[Dict[str, Any]] = []
    for r in top:
        cfg = {}
        for k, v in r.config.items():
            if k.startswith("_"):   # skip internal
                continue
            cfg[k] = v
        # Drop seed so we can override
        cfg.pop("seed", None)
        tops.append(cfg)

    #os.makedirs(args.log_dir, exist_ok=True)

    results = []
    for i, cfg in enumerate(tops):
        seeds = [1000 + i*10 + s for s in range(args.extra_seeds)]
        vals = []
        print(f"[recheck] launching seed {s} for config index {i}", flush=True)
        for s in seeds:
            rc = run_once(args.mm, args.program, args.subcmd, cfg, s,
              args.unlabeled_dir, args.labeled_dir, args.log_dir,
              args.project, args.group)
            if rc != 0:
                # show last ~120 lines of the log for quick diagnosis
                log_path = os.path.join(args.log_dir, f"recheck_{cfg.get('training_method','jepa')}_seed{s}.log")
                try:
                    with open(log_path, "r", encoding="utf-8") as lf:
                        lines = lf.readlines()[-120:]
                    print(f"[recheck][seed {s}] log tail:", "".join(lines), flush=True)
                except Exception as _e:
                    print(f"[recheck][seed {s}] could not read log {log_path}: {_e}", flush=True)
                raise RuntimeError(f"recheck run failed with exit code {rc} (seed {s})")
            # after the run, we could read the metric from W&B; here we assume train_jepa.py writes best val to a file or W&B;
            # to keep this self-contained, we re-fetch the last matching run (same group + tags would be best).
            # Minimal: refetch on metric for runs in the same project/group with matching cfg+seed.
            time.sleep(1)

        # Fetch metrics again
        # fetch from project+group (includes recheck runs; sweeps would miss them)
        entity = os.getenv("WANDB_ENTITY")
        proj_path = f"{entity}/{args.project}" if entity and args.project else None
        runs = []
        if proj_path:
            runs = list(api.runs(proj_path, filters={"group": args.group}))
        for r in runs:
            ok = True
            for k,v in cfg.items():
                if r.config.get(k) != v:
                    ok=False; break
            if ok and (r.config.get("seed") in seeds):
                mv = metric_of(r, args.metric)
                if mv is not None:
                    vals.append(float(mv))

        if vals:
            mu = float(np.mean(vals)); lo, hi = ci95(vals)
            results.append({"index": i, "mean": mu, "ci95": [lo,hi], "n": len(vals), "config": cfg})
        else:
            results.append({"index": i, "mean": None, "ci95": [None,None], "n": 0, "config": cfg})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"metric": args.metric, "direction": args.direction,
                   "topk": args.topk, "extra_seeds": args.extra_seeds,
                   "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
