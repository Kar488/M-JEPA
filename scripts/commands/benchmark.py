from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List
def cmd_benchmark(args: argparse.Namespace) -> None:
    """Compare JEPA and contrastive encoders on the same labelled dataset  with flexible loading + report.

    Runs training across seeds and reports which method yields better
    performance based on ROC‑AUC (classification) or RMSE (regression).
    """

    logger.info("Starting benchmark with args: %s", args)
    if (
        load_directory_dataset is None
        or build_encoder is None
        or train_linear_head is None
    ):
        logger.warning("Benchmark modules are unavailable.")
        sys.exit(6)

    seeds: List[int]
    arg_seeds = getattr(args, "seeds", None)
    if arg_seeds is not None and len(arg_seeds) > 0:
        seeds = arg_seeds
    else:
        seeds = CONFIG.get("benchmark", {}).get("seeds", [0])  # type: ignore[assignment]

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "labeled_dir": args.labeled_dir,
            "test_dir": getattr(args, "test_dir", None),
            "task_type": args.task_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seeds": seeds,
        },
    )

    import json
    import os
    import time

    import numpy as np
    import torch

    from utils.checkpoint import load_checkpoint  # for fine-tuned ckpt (encoder+head)

    # --- paths / report ---
    args.report_dir = getattr(args, "report_dir", "reports")
    os.makedirs(args.report_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_stem = getattr(args, "report_stem", f"benchmark_{timestamp}")
    report_json = os.path.join(args.report_dir, report_stem + ".json")
    report_csv = os.path.join(args.report_dir, report_stem + ".csv")

    data_dir = getattr(args, "test_dir", None) or args.labeled_dir

    # safe W&B helpers
    # safe W&B helpers: prefer wb.log / wb.finish if present; else try wb.run.*
    def _wb_log(payload):
        if wb is None:
            return
        try:
            if hasattr(wb, "log"):
                wb.log(payload)
            elif hasattr(wb, "run") and hasattr(wb.run, "log"):
                wb.run.log(payload)
        except Exception:
            pass
    def _wb_finish():
        if wb is None:
            return
        try:
            if hasattr(wb, "finish"):
                wb.finish()
            elif hasattr(wb, "run") and hasattr(wb.run, "finish"):
                wb.run.finish()
        except Exception:
            pass

    try:
        labeled = load_directory_dataset(
            data_dir,
            label_col=args.label_col,
            add_3d=args.add_3d,
            num_workers=getattr(args, "num_workers", 0),
            cache_dir=getattr(args, "cache_dir", None),
        )  # type: ignore[arg-type]
        _wb_log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset for benchmarking")
        _wb_log({"phase": "data_load", "status": "error"})
        sys.exit(1)

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = (
        None
        if labeled.graphs[0].edge_attr is None
        else labeled.graphs[0].edge_attr.shape[1]
    )
    device = resolve_device(args.device)

    # Prepare results dict
    all_results: Dict[str, Dict[str, float]] = {}
    from typing import Any, Dict

    # If a separate test directory is provided, run in eval-only mode using the
    # fine-tuned checkpoint and return early.
    if getattr(args, "test_dir", None):
        _wb_log({"phase": "benchmark", "status": "start"})
        agg_ft = evaluate_finetuned_head(args.ft_ckpt, labeled, args, device)
        if agg_ft:
            all_results["finetuned"] = agg_ft
            for k, v in agg_ft.items():
                wb.log({f"finetuned/{k}": v})
        verdict = "finetuned"
        _wb_log({"phase": "benchmark", "status": "success", "best_method": verdict})
        logger.info(f"Benchmark completed. Best method: {verdict}")

        try:
            payload = {"results": all_results, "best_method": verdict}
            with open(report_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            import csv

            with open(report_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["method", "metric", "value"])
                for k, v in agg_ft.items():
                    w.writerow(["finetuned", k, v])
            logger.info("Wrote reports: %s , %s", report_json, report_csv)
        except Exception:
            logger.warning("Failed to write reports", exc_info=True)
        finally:
            _wb_finish()
        return

    def evaluate_state(
        state_obj: Dict[str, Any] | Any, method_name: str
    ) -> Dict[str, float]:
        """
        Evaluate an already-loaded state object (either a raw encoder state_dict or a
        dict with key 'encoder'). Always trains a fresh linear head for fairness.
        """
        metrics_runs: List[Dict[str, float]] = []
        prev_det = None
        try:
            prev_det = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
        except Exception:
            prev_det = None

        for seed in seeds:
            # Repro
            torch.manual_seed(seed)
            np.random.seed(seed)
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            try:
                if prev_det is not None:
                    torch.use_deterministic_algorithms(True)
            except Exception:
                pass

            # Build & load encoder
            enc = build_encoder(
                gnn_type=args.gnn_type,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                edge_dim=edge_dim,
            )
            if isinstance(state_obj, dict) and "encoder" in state_obj:
                _load_state_dict_forgiving(enc, state_obj["encoder"])
            else:
                _load_state_dict_forgiving(enc, state_obj)
            _maybe_to(enc, device)

            # Train fresh head and log metrics
            mets = train_linear_head(
                dataset=labeled,
                encoder=enc,
                task_type=args.task_type,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
                patience=args.patience,
                devices=args.devices,
            )
            metrics_runs.append({k: v for k, v in mets.items() if k != "head"})

        agg = aggregate_metrics(metrics_runs)
        for k, v in agg.items():
            _wb_log({f"{method_name}/{k}": v})
        return agg

    # Thin wrappers that load, then call evaluate_state
    def evaluate_encoder(ckpt_path: str, method_name: str) -> Dict[str, float]:
        state = _safe_load_checkpoint(ckpt_path, device)
        return evaluate_state(state, method_name)

    def evaluate_finetuned(ft_ckpt_path: str) -> Dict[str, float]:
        try:
            state = load_checkpoint(ft_ckpt_path)
        except Exception:
            logger.exception("Failed to load fine-tuned checkpoint: %s", ft_ckpt_path)
            return {}
        return evaluate_state(state, "finetuned")

    _wb_log({"phase": "benchmark", "status": "start"})
    # Evaluate JEPA
    agg_jepa = evaluate_encoder(args.jepa_encoder, "jepa")
    all_results["jepa"] = agg_jepa

    # Evaluate contrastive
    agg_cont: Dict[str, float] = {}
    if args.contrastive_encoder:
        agg_cont = evaluate_encoder(args.contrastive_encoder, "contrastive")
        all_results["contrastive"] = agg_cont

    # Optional: evaluate a fine-tuned checkpoint that already has a head
    agg_ft: Dict[str, float] = {}
    if getattr(args, "ft_ckpt", None):
        agg_ft = evaluate_finetuned(args.ft_ckpt)
        if agg_ft:
            all_results["finetuned"] = agg_ft

    # Decide which is better
    verdict = "jepa"
    if agg_cont:
        # Choose metric based on task
        if args.task_type == "classification":
            # Higher AUC/ACC is better
            key = (
                "roc_auc_mean"
                if "roc_auc_mean" in agg_jepa
                else ("acc_mean" if "acc_mean" in agg_jepa else None)
            )
            if key and agg_cont.get(key, float("-inf")) > agg_jepa.get(
                key, float("-inf")
            ):
                verdict = "contrastive"
        else:
            # Lower RMSE/MAE is better
            key = (
                "rmse_mean"
                if "rmse_mean" in agg_jepa
                else ("mae_mean" if "mae_mean" in agg_jepa else None)
            )
            if key and agg_cont.get(key, float("inf")) < agg_jepa.get(
                key, float("inf")
            ):
                verdict = "contrastive"

    # If finetuned was evaluated, compare it too
    if "finetuned" in all_results:
        if args.task_type == "classification":
            key = (
                "roc_auc_mean"
                if "roc_auc_mean" in agg_jepa
                else ("acc_mean" if "acc_mean" in agg_jepa else None)
            )
            if key and all_results["finetuned"].get(
                key, float("-inf")
            ) > all_results.get(verdict, {}).get(key, float("-inf")):
                verdict = "finetuned"
        else:
            key = (
                "rmse_mean"
                if "rmse_mean" in agg_jepa
                else ("mae_mean" if "mae_mean" in agg_jepa else None)
            )
            if key and all_results["finetuned"].get(
                key, float("inf")
            ) < all_results.get(verdict, {}).get(key, float("inf")):
                verdict = "finetuned"

    _wb_log({"phase": "benchmark", "status": "success", "best_method": verdict})
    logger.info(f"Benchmark completed. Best method: {verdict}")

    # --- Write JSON/CSV report with all results + verdict ---
    try:
        payload = {"results": all_results, "best_method": verdict}
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        # CSV: method,metric,value
        import csv

        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "metric", "value"])
            for method, mets in all_results.items():
                for k, v in mets.items():
                    w.writerow([method, k, v])
        logger.info("Wrote reports: %s , %s", report_json, report_csv)
    except Exception:
        logger.warning("Failed to write reports", exc_info=True)
    finally:
        _wb_finish()

