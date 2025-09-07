from __future__ import annotations

import argparse
import sys
def cmd_tox21(args: argparse.Namespace) -> None:
    """Run the Tox21 ranking case study."""
    logger.info("Starting Tox21 case study with args: %s", args)
    if run_tox21_case_study is None:
        logger.error("Case study module is unavailable.")
        sys.exit(5)

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "csv": args.csv,
            "task": args.task,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "num_top_exclude": args.num_top_exclude,
        },
    )

    try:
        wb.log({"phase": "tox21", "status": "start"})
        mean_true, mean_random, mean_jepa, baseline_means = run_tox21_case_study(
            csv_path=args.csv,
            task_name=args.task,
            pretrain_epochs=getattr(args, "pretrain_epochs", 5),
            finetune_epochs=getattr(args, "finetune_epochs", 20),
            num_top_exclude=getattr(args, "num_top_exclude", 10),
            device=resolve_device(args.device),
        )
        # Assemble a single metrics dictionary so all values appear on the same
        # W&B step.  We prefix baseline keys for clarity.  This allows
        # convenient visualisation of all outputs together in the W&B UI.
        metrics = {
            "phase": "tox21",
            "status": "success",
            "mean_true": mean_true,
            "mean_random_after": mean_random,
            "mean_jepa_after": mean_jepa,
        }
        for name, val in baseline_means.items():
            metrics[f"baseline/{name}"] = val
        wb.log(metrics)

        import os, json, csv
        report_dir = getattr(args, "tox21_dir", os.path.join(os.path.dirname(args.csv), "reports"))
        os.makedirs(report_dir, exist_ok=True)
        stem = f"tox21_{args.task}"
        with open(os.path.join(report_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "mean_true": float(mean_true),
                "mean_rand": float(mean_random),
                "mean_pred": float(mean_jepa),
                "baselines": {k: float(v) for k, v in (baseline_means or {}).items()},
            }, f, indent=2, sort_keys=True)

        with open(os.path.join(report_dir, f"{stem}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["mean_true", float(mean_true)])
            w.writerow(["mean_rand", float(mean_random)])
            w.writerow(["mean_pred", float(mean_jepa)])
            for k, v in (baseline_means or {}).items():
                w.writerow([f"baseline/{k}", float(v)])

    except Exception as e:
        # Log full traceback and surface the message to W&B
        logger.exception("Tox21 case study failed")
        wb.log({"phase": "tox21", "status": "error", "error": str(e)})
        sys.exit(5)
    finally:
        wb.finish()

