from __future__ import annotations

import argparse
import sys
def cmd_tox21(args: argparse.Namespace) -> None:
    """Run the Tox21 ranking case study."""
    logger.info("Starting Tox21 case study with args: %s", args)
    if run_tox21_case_study is None:
        logger.error("Case study module is unavailable.")
        sys.exit(5)

    import os, json, csv
    
    triage_pct = getattr(args, "triage_pct", 0.10)
    calibrate  = not getattr(args, "no_calibrate", False)

    # choose a writable report dir (arg → env → <csv_dir>/reports)
    report_dir = getattr(args, "tox21_dir", None) \
                 or getattr(args, "report_dir", None) \
                 or os.environ.get("TOX21_DIR")

    if not report_dir:
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        report_dir = os.path.join(csv_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    wb = maybe_init_wandb(
        getattr(args, "use_wandb", False),
        project=getattr(args, "wandb_project", "m-jepa"),
        tags=getattr(args, "wandb_tags", []),
        config={
            "csv": args.csv,
            "task": args.task,
            "pretrain_epochs": getattr(args, "pretrain_epochs", 5),
            "finetune_epochs": getattr(args, "finetune_epochs", 20),
            "triage_pct": triage_pct,
            "calibrate": calibrate,
        },
    )

    try:
        wb.log({"phase": "tox21", "status": "start"})
        mean_true, mean_random, mean_jepa, baseline_means = run_tox21_case_study(
            csv_path=getattr(args, "csv"),
            task_name=getattr(args, "task"),
            pretrain_epochs=getattr(args, "pretrain_epochs", 5),
            finetune_epochs=getattr(args, "finetune_epochs", 20),
            lr=getattr(args, "lr", 1e-3),
            # model knobs may be missing if caller bypassed argparse
            hidden_dim=getattr(args, "hidden_dim", 128),
            num_layers=getattr(args, "num_layers", 2),
            gnn_type=getattr(args, "gnn_type", "edge_mpnn"),
            # drop contiguity for now (case_study doesn’t accept it yet)
            #contiguity=getattr(args, "contiguity", getattr(args, "contiguous", False)),
            contrastive=getattr(args, "contrastive", False),
            triage_pct=getattr(args, "triage_pct", 0.10),
            calibrate=not getattr(args, "no_calibrate", False),
            device=resolve_device(getattr(args, "device", "cpu")),
        )
        
        # Assemble a single metrics dictionary so all values appear on the same
        # W&B step.  We prefix baseline keys for clarity.  This allows
        # convenient visualisation of all outputs together in the W&B UI.

        wb.log({
             "phase": "tox21",
             "status": "success",
             "mean_true": float(mean_true),
             "mean_rand": float(mean_random),
             "mean_pred": float(mean_jepa),
            "task": args.task,
            "triage_pct": triage_pct,
            "calibrate": calibrate,
         })
        
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

