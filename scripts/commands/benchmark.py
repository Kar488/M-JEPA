from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from . import log_effective_gnn

try:  # pragma: no cover - optional relative import depending on entry point
    from ..bench import BenchmarkRule, resolve_metric_threshold
except ImportError:  # pragma: no cover - fallback when executed as a script
    from scripts.bench import BenchmarkRule, resolve_metric_threshold


HIGHER_IS_BETTER = {"roc_auc", "pr_auc", "acc", "accuracy"}
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

    dataset_name = getattr(args, "dataset", None)
    task_name = getattr(args, "task", None)
    threshold_rule: Optional[BenchmarkRule] = None
    if dataset_name:
        try:
            threshold_rule = resolve_metric_threshold(dataset_name, task_name)
        except KeyError:
            threshold_rule = None

    threshold_payload: Dict[str, Any] = {}
    if threshold_rule is not None:
        threshold_payload = {
            "benchmark_metric": threshold_rule.metric,
            "benchmark_threshold": threshold_rule.threshold,
        }

    seeds: List[int]
    arg_seeds = getattr(args, "seeds", None)
    if arg_seeds is not None and len(arg_seeds) > 0:
        seeds = arg_seeds
    else:
        seeds = CONFIG.get("benchmark", {}).get("seeds", [0])  # type: ignore[assignment]

    config_payload: Dict[str, Any] = {
        "labeled_dir": args.labeled_dir,
        "test_dir": getattr(args, "test_dir", None),
        "task_type": args.task_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seeds": seeds,
        "gnn_type": getattr(args, "gnn_type", None),
        "hidden_dim": getattr(args, "hidden_dim", None),
        "num_layers": getattr(args, "num_layers", None),
        "add_3d": bool(getattr(args, "add_3d", False)),
        "num_workers": getattr(args, "num_workers", None),
        "pin_memory": getattr(args, "pin_memory", None),
        "persistent_workers": getattr(args, "persistent_workers", None),
        "prefetch_factor": getattr(args, "prefetch_factor", None),
    }
    if dataset_name:
        config_payload["dataset"] = dataset_name
    if task_name:
        config_payload["task"] = task_name
    config_payload.update(threshold_payload)

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config=config_payload,
    )
    log_effective_gnn(args, logger, wb)

    import json
    import os
    import time

    import numpy as np
    import torch

    from utils.checkpoint import load_checkpoint  # for fine-tuned ckpt (encoder+head)

    try:
        from ..utils.checkpoint  import safe_load_checkpoint as _safe_load_checkpoint        # type: ignore[import-not-found]
        from ..utils.checkpoint  import load_state_dict_forgiving as _load_state_dict_forgiving      # type: ignore[import-not-found]
        from ..utils.checkpoint  import resolve_ckpt_path   # type: ignore[import-not-found]
    except ImportError:
        # Fallback: absolute imports when run from repo root with PYTHONPATH set
        from utils.checkpoint import safe_load_checkpoint  as _safe_load_checkpoint        # type: ignore[import-not-found]
        from utils.checkpoint import load_state_dict_forgiving as _load_state_dict_forgiving        # type: ignore[import-not-found]
        from utils.checkpoint import resolve_ckpt_path  # type: ignore[import-not-found]

    from pathlib import Path

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
            num_workers=getattr(args, "num_workers", -1),
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

    def _resolve_ft_checkpoint(raw: str) -> str:
        def _sanitize(label: str) -> str:
            return label.replace("/", "_").replace(" ", "_")

        base = Path(raw)
        search_root = base
        if base.suffix == ".pt" and not base.exists():
            search_root = base.parent

        def _find_in_dir(directory: Path) -> Optional[Path]:
            for fname in ("ft_best.pt", "head.pt"):
                candidate = directory / fname
                if candidate.is_file():
                    return candidate
            return None

        if base.is_file():
            logger.info("Resolved finetune head at %s", base)
            return str(base)

        label_candidates: List[str] = []
        arg_label = getattr(args, "label_col", None)
        if arg_label:
            label_candidates.append(str(arg_label))

        search_dirs = [search_root]
        for label in label_candidates:
            search_dirs.append(search_root / label)
            search_dirs.append(search_root / _sanitize(label))

        for candidate_dir in search_dirs:
            resolved = _find_in_dir(candidate_dir)
            if resolved:
                logger.info("Resolved finetune head at %s", resolved)
                return str(resolved)

            try:
                for subdir in sorted(p for p in candidate_dir.iterdir() if p.is_dir()):
                    resolved = _find_in_dir(subdir)
                    if resolved:
                        logger.info("Resolved finetune head at %s", resolved)
                        return str(resolved)
            except FileNotFoundError:
                continue

        logger.error(
            "Checkpoint path '%s' not found on disk (labels tried: %s)",
            raw,
            ", ".join(label_candidates) if label_candidates else "<none>",
        )
        raise FileNotFoundError(raw)

    resolved_ft_ckpt: Optional[str] = None
    if getattr(args, "ft_ckpt", None):
        try:
            resolved_ft_ckpt = _resolve_ft_checkpoint(args.ft_ckpt)
        except FileNotFoundError:
            _wb_log({"phase": "benchmark", "status": "error", "error": "missing_ft_ckpt"})
            _wb_finish()
            logger.exception("Fine-tuned checkpoint not found")
            sys.exit(1)

    # Prepare results dict
    all_results: Dict[str, Dict[str, float]] = {}
    from typing import Any, Dict

    # If a separate test directory is provided, run in eval-only mode using the
    # fine-tuned checkpoint and return early.
    if getattr(args, "test_dir", None):
        start_payload = {"phase": "benchmark", "status": "start"}
        start_payload.update(threshold_payload)
        _wb_log(start_payload)
        # Don’t resolve here—tests monkey-patch evaluate_finetuned_head().
        # Pass through what the CLI provided.
        ft_target = resolved_ft_ckpt or args.ft_ckpt
        try:
            agg_ft = evaluate_finetuned_head(ft_target, labeled, args, device)
        except FileNotFoundError:
            _wb_log({"phase": "benchmark", "status": "error", "error": "missing_ft_ckpt"})
            _wb_finish()
            logger.exception("Failed to resolve fine-tuned checkpoint")
            raise SystemExit(1)
        if agg_ft:
            all_results["finetuned"] = agg_ft
            for k, v in agg_ft.items():
                _wb_log({f"finetuned/{k}": v})
        verdict = "finetuned"
        success_payload = {"phase": "benchmark", "status": "success", "best_method": verdict}
        success_payload.update(threshold_payload)
        _wb_log(success_payload)
        logger.info(f"Benchmark completed. Best method: {verdict}")

        try:
            payload = {"results": all_results, "best_method": verdict}
            if threshold_rule is not None:
                payload["threshold"] = {
                    "dataset": dataset_name,
                    "task": task_name,
                    **threshold_payload,
                }
            with open(report_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            import csv

            with open(report_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["method", "metric", "value"])
                for k, v in agg_ft.items():
                    w.writerow(["finetuned", k, v])
                if threshold_rule is not None:
                    w.writerow(["threshold/metric", threshold_rule.metric])
                    w.writerow(["threshold/value", float(threshold_rule.threshold)])
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
                num_workers=getattr(args, "num_workers", -1),
                pin_memory=getattr(args, "pin_memory", True),
                persistent_workers=getattr(args, "persistent_workers", True),
                prefetch_factor=getattr(args, "prefetch_factor", 4),
                bf16=getattr(args, "bf16", False),
            )
            metrics_runs.append({k: v for k, v in mets.items() if k != "head"})

        agg = aggregate_metrics(metrics_runs)
        for k, v in agg.items():
            _wb_log({f"{method_name}/{k}": v})
        return agg

    # Thin wrappers that load, then call evaluate_state
    def evaluate_encoder(ckpt_path: str, method_name: str) -> Dict[str, float]:
        state, loaded_path = _safe_load_checkpoint(
            primary=ckpt_path,
            ckpt_dir=None,
            default_name="encoder.pt",
            map_location=device,
            allow_missing=True,
        )
        if not isinstance(state, dict):
            state = {}
        if "encoder" not in state or not state["encoder"]:
            logger.warning("No encoder weights; using random init (path=%r).", loaded_path or getattr(args, "ft_ckpt", None))
            
        return evaluate_state(state, method_name)

    def evaluate_finetuned(ft_ckpt_path: str) -> Dict[str, float]:
        try:
            state = load_checkpoint(ft_ckpt_path)
        except Exception:
            logger.exception("Failed to load fine-tuned checkpoint: %s", ft_ckpt_path)
            _wb_log({"phase": "benchmark", "status": "error", "error": "missing_ft_ckpt"})
            _wb_finish()
            raise SystemExit(1)
        return evaluate_state(state, "finetuned")

    main_start_payload = {"phase": "benchmark", "status": "start"}
    main_start_payload.update(threshold_payload)
    _wb_log(main_start_payload)
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
    if resolved_ft_ckpt:
        agg_ft = evaluate_finetuned(resolved_ft_ckpt)
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

    metric_value_key: Optional[str] = None
    threshold_report: Optional[Dict[str, Any]] = None
    metric_pass: Optional[bool] = None
    best_metric_value: Optional[float] = None
    if threshold_rule is not None:
        for candidate in (f"{threshold_rule.metric}_mean", threshold_rule.metric):
            if any(candidate in mets for mets in all_results.values()):
                metric_value_key = candidate
                break
        if metric_value_key is not None:
            higher_is_better = threshold_rule.metric in HIGHER_IS_BETTER
            threshold_report = {
                "dataset": dataset_name,
                "task": task_name,
                "metric": threshold_rule.metric,
                "threshold": threshold_rule.threshold,
                "metric_key": metric_value_key,
                "orientation": "higher" if higher_is_better else "lower",
                "results": {},
            }
            for method, metrics in all_results.items():
                if metric_value_key in metrics:
                    value = metrics[metric_value_key]
                    passed = bool(
                        value >= threshold_rule.threshold
                        if higher_is_better
                        else value <= threshold_rule.threshold
                    )
                    threshold_report["results"][method] = {
                        "value": value,
                        "passed": passed,
                    }
                    if method == verdict:
                        best_metric_value = value
                        metric_pass = passed

    success_payload = {"phase": "benchmark", "status": "success", "best_method": verdict}
    success_payload.update(threshold_payload)
    if best_metric_value is not None:
        success_payload["benchmark_metric_value"] = best_metric_value
    if metric_pass is not None:
        success_payload["benchmark_pass"] = metric_pass
    logger.info(f"Benchmark completed. Best method: {verdict}")
    _wb_log(success_payload)

    # --- Write JSON/CSV report with all results + verdict ---
    try:
        payload = {"results": all_results, "best_method": verdict}
        if threshold_report is not None:
            payload["threshold"] = threshold_report
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
            if threshold_rule is not None:
                w.writerow(["threshold/info", "metric", threshold_rule.metric])
                w.writerow(["threshold/info", "threshold", float(threshold_rule.threshold)])
                if metric_value_key is not None:
                    for method, info in (threshold_report or {}).get("results", {}).items():
                        value = info.get("value")
                        if value is not None:
                            w.writerow([f"threshold/{method}", metric_value_key, float(value)])
                        w.writerow([f"threshold/{method}", "passed", info.get("passed")])
                if best_metric_value is not None:
                    w.writerow(["threshold/best_method", "name", verdict])
                    w.writerow(["threshold/best_method", metric_value_key or "value", float(best_metric_value)])
                    if metric_pass is not None:
                        w.writerow(["threshold/best_method", "passed", metric_pass])
        logger.info("Wrote reports: %s , %s", report_json, report_csv)
    except Exception:
        logger.warning("Failed to write reports", exc_info=True)
    finally:
        _wb_finish()
