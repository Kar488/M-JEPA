#!/usr/bin/env python3
import argparse, json, wandb,os


def main():

    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="val_loss",
                        help="Metric name to minimize (default=val_loss)")
    parser.add_argument("--out", default="grid/best_grid_config.json",
                        help="Path to write best config JSON")
    parser.add_argument("--sweep_id", help="W&B sweep id (entity/project/sweepid)")
    args = parser.parse_args()

    

    sweep_id = args.sweep_id or f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{os.environ['WANDB_SWEEP_ID2']}"

    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    if not sweep.runs:
        raise RuntimeError(f"No runs found in sweep {args.sweep_id}")

    # Pick run with lowest metric
    best_run = min(
        sweep.runs,
        key=lambda r: r.summary.get(args.metric, float("inf"))
    )

    print(f"[export_best] Best run = {best_run.name}, "
          f"{args.metric}={best_run.summary.get(args.metric)}")

    with open(args.out, "w") as f:
        json.dump(best_run.config, f, indent=2)

    print(f"[export_best] Wrote best config to {args.out}")

if __name__ == "__main__":
    main()
