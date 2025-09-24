from collections import defaultdict
import numpy as np, wandb, os, argparse, json, re, sys

def _coerce_to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.getenv("WANDB_PROJECT","mjepa"))
    ap.add_argument("--group",   default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--metric",  default="val_rmse")
    ap.add_argument("--direction", choices=["min","max"], default=None,
                    help="Optimization direction for metric; if omitted, infer from metric name.")
    ap.add_argument("--out", default=os.path.join(os.getenv("GRID_DIR", "."), "paired_effect.json"))
    ap.add_argument("--seed", type=int, default=None, help="Seed for bootstrap reproducibility")
    ap.add_argument("--aggregate", choices=["pair-seed","mean","median","best"], default="pair-seed",
        help="How to aggregate multiple runs per (pair_id, method). "
        "'pair-seed' = compute deltas only on shared seeds; "
        "fallback to mean per method if no shared seeds.")
    ap.add_argument("--strict", action="store_true",
        help="Exit non-zero when no runs or no matched pairs are found.")
    ap.add_argument("--min_pretrain_epochs", type=int, default=None,
        help="Minimum pretraining epochs (config: pretrain_epochs, units: epochs) required to include a run.")
    ap.add_argument("--min_finetune_epochs", type=int, default=None,
        help="Minimum fine-tuning epochs (config: finetune_epochs, units: epochs) required to include a run.")
    ap.add_argument("--min_pretrain_batches", type=int, default=None,
        help="Minimum pretraining batches (config: max_pretrain_batches, units: batches) required to include a run.")
    args = ap.parse_args()

    api = wandb.Api()
    filters = {"group": args.group} if args.group else None
    runs = api.runs(f"{os.getenv('WANDB_ENTITY')}/{args.project}", filters=filters)
    if not runs:
        # Do not write output when empty; only fail hard if --strict
        if args.strict:
            import sys; print("No runs found.", flush=True); sys.exit(2)
        return

    # Collect values
    # For general aggregation:        by_pair_vals[pid][method] -> list[float]
    # For seed-wise paired deltas:    by_pair_seed[pid][seed][method] -> list[float]
    by_pair_vals = defaultdict(lambda: defaultdict(list))
    by_pair_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in runs:
        mid = r.config.get("training_method")
        pid = r.config.get("pair_id")
        v   = r.summary.get(args.metric)
        if not pid or mid not in ("jepa","contrastive") or v is None:
            continue

        thresholds = (
            ("pretrain_epochs", args.min_pretrain_epochs),
            ("finetune_epochs", args.min_finetune_epochs),
            ("max_pretrain_batches", args.min_pretrain_batches),
        )
        skip = False
        for key, minimum in thresholds:
            if minimum is None:
                continue
            conf_val = _coerce_to_float(r.config.get(key))
            # Phase-1 sweeps often terminate early while phase-2 sweeps rely on mature metrics;
            # filtering prevents these under-trained runs from biasing the phase-1/phase-2 workflow.
            if conf_val is None or conf_val < minimum:
                skip = True
                break
        if skip:
            continue
        
        try:
            val = float(v)
        except Exception:
            continue

        pid = str(pid)
        by_pair_vals[pid][mid].append(val)
        # capture seed if available
        seed = r.config.get("seed", None)
        if seed is not None:
            try:
                seed = int(seed)
            except Exception:
                pass
            by_pair_seed[pid][seed][mid].append(val)

    # infer direction (or use CLI) BEFORE reduction
    direction = args.direction
    if direction is None:
        m = args.metric.lower()
        direction = "max" if re.search(r'(auc|acc|f1|pr[_-]?auc|roc)', m) else "min"
    choose_best = max if direction == "max" else min
    choose_mean = np.mean
    choose_median = np.median

    deltas = []
    used_pairs = 0
    if args.aggregate == "pair-seed":
        # Compute deltas only on seeds present for BOTH methods. If none exist,
        # fall back to mean per method for that pair (unpaired).
        for pid, seeds in by_pair_seed.items():
            # find common seeds
            common = [sd for sd, mm in seeds.items() if "jepa" in mm and "contrastive" in mm]
            pair_deltas = []
            for sd in common:
                # Reduce within-seed (mean across repeats of same seed/method)
                jv = float(choose_mean(seeds[sd]["jepa"]))
                cv = float(choose_mean(seeds[sd]["contrastive"]))
                pair_deltas.append(cv - jv)
            if pair_deltas:
                deltas.extend(pair_deltas)
                used_pairs += 1
            else:
                # fallback: mean per method for the whole pair if both present at all
                methods = by_pair_vals.get(pid, {})
                if "jepa" in methods and "contrastive" in methods and methods["jepa"] and methods["contrastive"]:
                    jv = float(choose_mean(methods["jepa"]))
                    cv = float(choose_mean(methods["contrastive"]))
                    deltas.append(cv - jv)
                    used_pairs += 1
        # Global fallback: if no seeds were present at all, reduce across ALL pairs
        if not deltas and by_pair_vals:
            for pid, methods in by_pair_vals.items():
                if "jepa" in methods and "contrastive" in methods and methods["jepa"] and methods["contrastive"]:
                    jv = float(np.mean(methods["jepa"]))
                    cv = float(np.mean(methods["contrastive"]))
                    deltas.append(cv - jv)
                    used_pairs += 1
    else:
        # Non-paired reducers collapse per method first, then compute one delta per pair
        reducer = {"mean": choose_mean, "median": choose_median, "best": choose_best}[args.aggregate]
        for pid, methods in by_pair_vals.items():
            if "jepa" in methods and "contrastive" in methods and methods["jepa"] and methods["contrastive"]:
                jv = float(reducer(methods["jepa"]))
                cv = float(reducer(methods["contrastive"]))
                deltas.append(cv - jv)
                used_pairs += 1


    if not deltas:
        # As a last resort, attempt a global comparison between methods even if no
        # per-pair matches were recorded.  This allows tiny sweeps (e.g. WANDB_COUNT
        # of just a few runs) to still report a winner instead of failing with
        # "No matched pairs" once filters remove some runs.  The fallback preserves
        # the aggregate direction semantics while clearly signalling that no actual
        # pairs were used in the comparison via `used_pairs`.
        global_methods = defaultdict(list)
        for methods in by_pair_vals.values():
            for method, values in methods.items():
                global_methods[method].extend(values)

        if global_methods.get("jepa") and global_methods.get("contrastive"):
            # Treat the aggregate as a single pseudo-pair delta so downstream code
            # can continue to operate without special casing.  Record that no real
            # pairs were consumed to keep reporting transparent.
            jv = float(np.mean(global_methods["jepa"]))
            cv = float(np.mean(global_methods["contrastive"]))
            deltas = [cv - jv]
            used_pairs = 0
            print("[paired-effect] falling back to global mean delta across methods", flush=True)

    if not deltas:
        # Do not write output when empty; only fail hard if --strict
        if args.strict:
            import sys; print("No matched pairs found.", flush=True); sys.exit(2)
        return
    

    mu = float(np.mean(deltas))
    # bootstrap 95% CI
    if args.seed is not None:
        np.random.seed(args.seed)
    bs = [np.mean(np.random.choice(deltas, size=len(deltas), replace=True)) for _ in range(5000)]

    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    
    if direction == "min":
        win = 100.0 * sum(d < 0 for d in deltas) / len(deltas)
    else:
        win = 100.0 * sum(d > 0 for d in deltas) / len(deltas)

    print(f"Pairs: {len(deltas)}  meanΔ(ctr-JEPA)={mu:.4f}  95%CI[{lo:.4f},{hi:.4f}]  win%={win:.1f}")

    # delta = contrastive - jepa
    # min: lower is better → contrastive wins if mu < 0; max: higher is better → contrastive wins if mu > 0
    if direction == "min":
        winner = "contrastive" if mu < 0 else "jepa"
    else:
        winner = "contrastive" if mu > 0 else "jepa"

    task = "classification" if direction == "max" else "regression"

    # machine-readable artifact
    payload = {
        "metric": args.metric, "direction": direction,
        "pairs": len(deltas), "mean_delta_contrastive_minus_jepa": mu,
        "ci95": [lo, hi], "win_pct_contrastive_over_jepa": win,
        "winner": winner, "task": task,
        "pairs_used": used_pairs,
        "aggregate": args.aggregate,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # grep-friendly marker line:
    print(f"::winner::{winner} ::task::{task} ::metric::{args.metric} ::direction::{direction}")
if __name__ == "__main__":
    main()
