from __future__ import annotations

import argparse
import logging
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from . import log_effective_gnn

try:  # pragma: no cover - optional relative import depending on entry point
    from ..bench import BenchmarkRule, resolve_metric_threshold
except ImportError:  # pragma: no cover - fallback when executed as a script
    from scripts.bench import BenchmarkRule, resolve_metric_threshold


logger = logging.getLogger(__name__)


def _wandb_log_safe(wb: Any, payload: Dict[str, Any]) -> None:
    if wb is None:
        return
    try:
        if hasattr(wb, "log"):
            wb.log(payload)
        elif hasattr(wb, "run") and hasattr(wb.run, "log"):
            wb.run.log(payload)
    except Exception:
        pass


def _wandb_save_safe(wb: Any, path: str) -> None:
    if wb is None or not path:
        return
    try:
        if hasattr(wb, "save"):
            wb.save(path)
        elif hasattr(wb, "run") and hasattr(wb.run, "save"):
            wb.run.save(path)
    except Exception:
        pass


def _coerce_case_study_result(result: Any) -> Tuple[List[Any], Any]:
    """Normalise legacy return types from ``run_tox21_case_study``.

    Historically the case-study entry point returned a simple tuple of
    ``(mean_true, mean_random, mean_pred)`` (optionally followed by baseline
    dictionaries).  Modern implementations return a dataclass with an
    ``evaluations`` attribute.  Tests and scripted entry points may monkeypatch
    the function with either shape, so this helper converts the response into a
    uniform ``List`` of evaluation objects and propagates the optional
    ``threshold_rule`` when available.
    """

    rule_from_result = getattr(result, "threshold_rule", None)

    evaluations = list(getattr(result, "evaluations", []) or [])
    if evaluations:
        return evaluations, rule_from_result

    def _build_eval(mean_true: Any, mean_rand: Any, mean_pred: Any, baselines: Dict[str, Any], metrics: Dict[str, Any]):
        return SimpleNamespace(
            name="evaluation",
            encoder_source=getattr(result, "encoder_source", "unknown"),
            mean_true=float(mean_true),
            mean_random=float(mean_rand),
            mean_pred=float(mean_pred),
            baseline_means={str(k): float(v) for k, v in (baselines or {}).items()},
            metrics={str(k): float(v) for k, v in (metrics or {}).items()},
            benchmark_metric=getattr(result, "benchmark_metric", None),
            benchmark_threshold=getattr(result, "benchmark_threshold", None),
            met_benchmark=getattr(result, "met_benchmark", None),
            manifest_path=getattr(result, "manifest_path", None),
        )

    if isinstance(result, (list, tuple)) and len(result) >= 3:
        baselines = result[3] if len(result) >= 4 and isinstance(result[3], dict) else {}
        metrics = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
        evaluation = _build_eval(result[0], result[1], result[2], baselines, metrics)
        return [evaluation], rule_from_result

    if isinstance(result, dict):
        mean_true = result.get("mean_true")
        mean_rand = result.get("mean_random", result.get("mean_rand"))
        mean_pred = result.get("mean_pred")
        if mean_true is not None and mean_rand is not None and mean_pred is not None:
            baselines_dict = result.get("baseline_means", result.get("baselines", {}))
            metrics_dict = result.get("metrics", {})
            evaluation = _build_eval(mean_true, mean_rand, mean_pred, baselines_dict, metrics_dict)
            return [evaluation], rule_from_result

    return [], rule_from_result
def cmd_tox21(args: argparse.Namespace) -> None:
    """Run the Tox21 ranking case study."""
    logger.info("Starting Tox21 case study with args: %s", args)
    if run_tox21_case_study is None:
        logger.error("Case study module is unavailable.")
        sys.exit(5)

    import csv
    import json
    import os

    triage_pct = getattr(args, "triage_pct", 0.10)
    calibrate = not getattr(args, "no_calibrate", False)

    dataset_name = getattr(args, "dataset", "tox21") or "tox21"
    task_name = getattr(args, "task", None)
    threshold_rule: BenchmarkRule | None = None
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

    report_dir = (
        getattr(args, "tox21_dir", None)
        or getattr(args, "report_dir", None)
        or os.environ.get("TOX21_DIR")
    )
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
            "dataset": dataset_name,
            "pretrain_epochs": getattr(args, "pretrain_epochs", 5),
            "finetune_epochs": getattr(args, "finetune_epochs", 20),
            "triage_pct": triage_pct,
            "calibrate": calibrate,
            "pretrain_time_budget_mins": getattr(args, "pretrain_time_budget_mins", 0),
            "finetune_time_budget_mins": getattr(args, "finetune_time_budget_mins", 0),
            **threshold_payload,
        },
    )
    log_effective_gnn(args, logger, wb)

    try:
        start_log = {"phase": "tox21", "status": "start"}
        start_log.update(threshold_payload)
        _wandb_log_safe(wb, start_log)

        result = run_tox21_case_study(
            csv_path=getattr(args, "csv"),
            task_name=getattr(args, "task"),
            dataset_name=dataset_name,
            pretrain_epochs=getattr(args, "pretrain_epochs", 5),
            finetune_epochs=getattr(args, "finetune_epochs", 20),
            lr=getattr(args, "lr", 1e-3),
            hidden_dim=getattr(args, "hidden_dim", 128),
            num_layers=getattr(args, "num_layers", 2),
            gnn_type=getattr(args, "gnn_type", "edge_mpnn"),
            contrastive=getattr(args, "contrastive", False),
            triage_pct=triage_pct,
            calibrate=calibrate,
            device=resolve_device(getattr(args, "device", "cpu")),
            num_workers=getattr(args, "num_workers", -1),
            pin_memory=getattr(args, "pin_memory", True),
            persistent_workers=getattr(args, "persistent_workers", True),
            prefetch_factor=getattr(args, "prefetch_factor", 4),
            bf16=getattr(args, "bf16", False),
            bf16_head=getattr(args, "bf16_head", None),
            pretrain_time_budget_mins=getattr(args, "pretrain_time_budget_mins", 0),
            finetune_time_budget_mins=getattr(args, "finetune_time_budget_mins", 0),
            encoder_checkpoint=getattr(args, "encoder_checkpoint", None),
            encoder_manifest=getattr(args, "encoder_manifest", None),
            strict_encoder_config=getattr(args, "strict_encoder_config", False),
        )

        evaluations, rule_from_result = _coerce_case_study_result(result)
        if not evaluations:
            raise RuntimeError("run_tox21_case_study returned no evaluation results")

        if rule_from_result is not None:
            threshold_rule = rule_from_result
            threshold_payload = {
                "benchmark_metric": threshold_rule.metric,
                "benchmark_threshold": threshold_rule.threshold,
            }

        primary = evaluations[0]
        summary_payload = {
            "phase": "tox21",
            "status": "success",
            "mean_true": float(getattr(primary, "mean_true", 0.0)),
            "mean_rand": float(getattr(primary, "mean_random", 0.0)),
            "mean_pred": float(getattr(primary, "mean_pred", 0.0)),
            "task": args.task,
            "triage_pct": triage_pct,
            "calibrate": calibrate,
            **threshold_payload,
        }
        _wandb_log_safe(wb, summary_payload)

        multi_eval = len(evaluations) > 1
        for eval_res in evaluations:
            prefix = f"{getattr(eval_res, 'name', 'evaluation')}/" if multi_eval else ""
            payload = {"phase": "tox21", "status": "success"}
            payload.update(threshold_payload)
            payload[f"{prefix}mean_true"] = float(getattr(eval_res, "mean_true", 0.0))
            payload[f"{prefix}mean_rand"] = float(getattr(eval_res, "mean_random", 0.0))
            payload[f"{prefix}mean_pred"] = float(getattr(eval_res, "mean_pred", 0.0))
            payload[f"{prefix}encoder_source"] = getattr(eval_res, "encoder_source", "unknown")

            benchmark_metric = getattr(eval_res, "benchmark_metric", None)
            benchmark_threshold = getattr(eval_res, "benchmark_threshold", None)
            met_benchmark = getattr(eval_res, "met_benchmark", None)
            if benchmark_metric is not None:
                payload[f"{prefix}benchmark_metric"] = benchmark_metric
            if benchmark_threshold is not None:
                payload[f"{prefix}benchmark_threshold"] = float(benchmark_threshold)
            if met_benchmark is not None:
                payload[f"{prefix}met_benchmark"] = bool(met_benchmark)

            metrics_block: Dict[str, Any] = getattr(eval_res, "metrics", {}) or {}
            for name, value in metrics_block.items():
                payload[f"{prefix}metrics/{name}"] = float(value)

            baseline_block: Dict[str, Any] = getattr(eval_res, "baseline_means", {}) or {}
            for name, value in baseline_block.items():
                payload[f"{prefix}baseline/{name}"] = float(value)

            _wandb_log_safe(wb, payload)

            manifest_path = getattr(eval_res, "manifest_path", None)
            if manifest_path:
                _wandb_save_safe(wb, manifest_path)
                _wandb_log_safe(wb, {f"{prefix}encoder_manifest": manifest_path})

        stem = f"tox21_{args.task}"
        json_path = os.path.join(report_dir, f"{stem}.json")
        csv_path = os.path.join(report_dir, f"{stem}.csv")

        json_payload: Dict[str, Any] = {
            "mean_true": float(getattr(primary, "mean_true", 0.0)),
            "mean_rand": float(getattr(primary, "mean_random", 0.0)),
            "mean_pred": float(getattr(primary, "mean_pred", 0.0)),
            "baselines": {
                k: float(v)
                for k, v in (getattr(primary, "baseline_means", {}) or {}).items()
            },
            "threshold": {
                "dataset": dataset_name,
                "task": task_name,
                **threshold_payload,
            },
            "evaluations": [],
        }

        for eval_res in evaluations:
            json_payload["evaluations"].append(
                {
                    "name": getattr(eval_res, "name", "evaluation"),
                    "encoder_source": getattr(eval_res, "encoder_source", "unknown"),
                    "mean_true": float(getattr(eval_res, "mean_true", 0.0)),
                    "mean_rand": float(getattr(eval_res, "mean_random", 0.0)),
                    "mean_pred": float(getattr(eval_res, "mean_pred", 0.0)),
                    "baseline_means": {
                        k: float(v)
                        for k, v in (getattr(eval_res, "baseline_means", {}) or {}).items()
                    },
                    "metrics": {
                        k: float(v)
                        for k, v in (getattr(eval_res, "metrics", {}) or {}).items()
                    },
                    "benchmark_metric": getattr(eval_res, "benchmark_metric", None),
                    "benchmark_threshold": getattr(eval_res, "benchmark_threshold", None),
                    "met_benchmark": getattr(eval_res, "met_benchmark", None),
                    "encoder_manifest": getattr(eval_res, "manifest_path", None),
                }
            )

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(json_payload, fh, indent=2, sort_keys=True)

        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["evaluation", "metric", "value"])
            for eval_res in evaluations:
                name = getattr(eval_res, "name", "evaluation")
                writer.writerow([name, "mean_true", float(getattr(eval_res, "mean_true", 0.0))])
                writer.writerow([name, "mean_rand", float(getattr(eval_res, "mean_random", 0.0))])
                writer.writerow([name, "mean_pred", float(getattr(eval_res, "mean_pred", 0.0))])
                for key, value in (getattr(eval_res, "baseline_means", {}) or {}).items():
                    writer.writerow([name, f"baseline/{key}", float(value)])
                for key, value in (getattr(eval_res, "metrics", {}) or {}).items():
                    writer.writerow([name, f"metrics/{key}", float(value)])
                bm_metric = getattr(eval_res, "benchmark_metric", None)
                if bm_metric is not None:
                    writer.writerow([name, "benchmark_metric", bm_metric])
                bm_thresh = getattr(eval_res, "benchmark_threshold", None)
                if bm_thresh is not None:
                    writer.writerow([name, "benchmark_threshold", float(bm_thresh)])
                bm_met = getattr(eval_res, "met_benchmark", None)
                if bm_met is not None:
                    writer.writerow([name, "met_benchmark", int(bool(bm_met))])
                manifest_path = getattr(eval_res, "manifest_path", None)
                if manifest_path:
                    writer.writerow([name, "encoder_manifest", manifest_path])

    except Exception as exc:
        logger.exception("Tox21 case study failed")
        error_log = {"phase": "tox21", "status": "error", "error": str(exc)}
        error_log.update(threshold_payload)
        _wandb_log_safe(wb, error_log)
        sys.exit(5)
    finally:
        try:
            if wb is not None and hasattr(wb, "finish"):
                wb.finish()
            elif wb is not None and hasattr(wb, "run") and hasattr(wb.run, "finish"):
                wb.run.finish()
        except Exception:
            pass

