from collections import defaultdict
import numpy as np, wandb, os, argparse, json, re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.getenv("WANDB_PROJECT","mjepa"))
    ap.add_argument("--group",   default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--metric",  default="val_rmse")
    ap.add_argument("--direction", choices=["min","max"], default=None,
                    help="Optimization direction for metric; if omitted, infer from metric name.")
    ap.add_argument("--out", default=os.path.join(os.getenv("GRID_DIR", "."), "paired_effect.json"))
    args = ap.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{os.getenv('WANDB_ENTITY')}/{args.project}",
                    filters={"group": args.group})

    by_pair = defaultdict(dict)
    for r in runs:
        mid = r.config.get("training_method")
        pid = r.config.get("pair_id")
        v   = r.summary.get(args.metric)
        if pid and mid in ("jepa","contrastive") and v is not None:
            by_pair[pid][mid] = float(v)

    deltas = [pair["contrastive"] - pair["jepa"]
              for pair in by_pair.values()
              if "jepa" in pair and "contrastive" in pair]

    if not deltas:
        print("No matched pairs found.")
        return
    mu = float(np.mean(deltas))
    # bootstrap 95% CI
    bs = [np.mean(np.random.choice(deltas, size=len(deltas), replace=True))
          for _ in range(5000)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    win = 100.0 * sum(d>0 for d in deltas)/len(deltas)

    print(f"Pairs: {len(deltas)}  meanΔ(ctr-JEPA)={mu:.4f}  95%CI[{lo:.4f},{hi:.4f}]  win%={win:.1f}")

    # ---- decide direction & task, then winner
    direction = args.direction
    if direction is None:
        m = args.metric.lower()
        # crude but robust default: AUC/ACC/F1 → max, RMSE/MAE/LOSS → min
        if re.search(r'(auc|acc|f1|pr_auc|roc)', m):
            direction = "max"
        else:
            direction = "min"

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
        "winner": winner, "task": task
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # grep-friendly marker line:
    print(f"::winner::{winner} ::task::{task} ::metric::{args.metric} ::direction::{direction}")
if __name__ == "__main__":
    main()
