from __future__ import annotations

import argparse
import logging
import math
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
    eval_mode = str(
        getattr(
            args,
            "evaluation_mode",
            getattr(args, "encoder_source", "pretrain_frozen"),
        )
        or "pretrain_frozen"
    ).lower()
    if getattr(args, "encoder_source", None) is None:
        setattr(args, "encoder_source", eval_mode)
    threshold_rule: BenchmarkRule | None = None
    try:
        threshold_rule = resolve_metric_threshold(dataset_name, task_name)
    except KeyError:
        threshold_rule = None

    threshold_payload: Dict[str, Any] = {}
    target_baseline = 0.86
    if threshold_rule is not None:
        metric_name = str(getattr(threshold_rule, "metric", "")).lower()
        if metric_name == "roc_auc":
            try:
                target_baseline = float(threshold_rule.threshold)
            except Exception:
                target_baseline = 0.86
        threshold_payload = {
            "benchmark_metric": threshold_rule.metric,
            "benchmark_threshold": threshold_rule.threshold,
        }

    target_payload = {"target_baseline_roc_auc": float(target_baseline)}

    report_dir = (
        getattr(args, "tox21_dir", None)
        or getattr(args, "report_dir", None)
        or os.environ.get("TOX21_DIR")
    )
    if not report_dir:
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        report_dir = os.path.join(csv_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    gate_passed_flag = False

    def _export_gate_env(passed: bool) -> None:
        env_value = "true" if passed else "false"
        os.environ["TOX21_MET_GATE"] = env_value
        env_path = os.environ.get("GITHUB_ENV")
        if env_path:
            try:
                with open(env_path, "a", encoding="utf-8") as fh:
                    fh.write(f"TOX21_MET_GATE={env_value}\n")
            except Exception:
                logger.debug("Failed to write TOX21_MET_GATE to %s", env_path, exc_info=True)

    wandb_tags = list(getattr(args, "wandb_tags", []) or [])
    if "target_baseline_roc_auc" not in {str(t) for t in wandb_tags}:
        wandb_tags.append("target_baseline_roc_auc")

    wb = maybe_init_wandb(
        getattr(args, "use_wandb", False),
        project=getattr(args, "wandb_project", "m-jepa"),
        tags=wandb_tags,
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
            **target_payload,
            "evaluation_mode": eval_mode,
        },
    )
    log_effective_gnn(args, logger, wb)

    try:
        start_log = {"phase": "tox21", "status": "start"}
        start_log.update(threshold_payload)
        start_log.update(target_payload)
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
            encoder_source_override=getattr(args, "encoder_source", None),
            evaluation_mode=eval_mode,
        )

        diagnostics = getattr(result, "diagnostics", {}) or {}

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
        auc_summary: Dict[str, float] = {}
        benchmark_flags: Dict[str, bool] = {}
        manifest_lookup: Dict[str, str] = {}
        for eval_res in evaluations:
            source = getattr(eval_res, "encoder_source", getattr(eval_res, "name", "unknown"))
            metrics_block = getattr(eval_res, "metrics", {}) or {}
            roc_auc = metrics_block.get("roc_auc")
            if roc_auc is not None and not math.isnan(roc_auc):
                auc_summary[source] = float(roc_auc)
            met_flag = getattr(eval_res, "met_benchmark", None)
            if met_flag is not None:
                benchmark_flags[source] = bool(met_flag)
            manifest_path = getattr(eval_res, "manifest_path", None)
            if manifest_path:
                manifest_lookup[source] = manifest_path

        def _lookup_auc(*keys: str) -> float | None:
            for key in keys:
                if key in auc_summary:
                    return auc_summary[key]
            return None

        selected_source = None
        if auc_summary:
            selected_source = max(auc_summary.items(), key=lambda kv: kv[1])[0]
        selected_benchmark = (
            benchmark_flags[selected_source]
            if selected_source in benchmark_flags
            else None
        )

        primary_metrics = getattr(primary, "metrics", {}) or {}
        gate_metric_name = getattr(primary, "benchmark_metric", None)
        gate_threshold = getattr(primary, "benchmark_threshold", None)
        gate_metric_value = None
        if gate_metric_name is not None:
            gate_metric_value = primary_metrics.get(str(gate_metric_name))
        gate_flag_attr = getattr(primary, "met_benchmark", None)
        gate_passed_flag = bool(gate_flag_attr) if gate_flag_attr is not None else False
        source_for_gate = getattr(primary, "encoder_source", getattr(args, "encoder_source", "unknown"))

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
        summary_payload["encoder_source"] = source_for_gate
        summary_payload["evaluation_mode"] = eval_mode
        summary_payload["encoder_checkpoint"] = getattr(args, "encoder_checkpoint", None)
        summary_payload["tox21_gate_passed"] = bool(gate_passed_flag)
        summary_payload.update(target_payload)
        if gate_metric_name is not None and "benchmark_metric" not in summary_payload:
            summary_payload["benchmark_metric"] = gate_metric_name
        if gate_threshold is not None and "benchmark_threshold" not in summary_payload:
            summary_payload["benchmark_threshold"] = float(gate_threshold)
        if gate_metric_value is not None:
            try:
                metric_val = float(gate_metric_value)
                if not math.isnan(metric_val):
                    summary_payload["benchmark_metric_value"] = metric_val
            except Exception:
                pass
        frozen_auc = _lookup_auc("pretrain_frozen", "frozen", "checkpoint")
        fine_tuned_auc = _lookup_auc("fine_tuned", "fine-tuned", "scratch")
        if frozen_auc is not None:
            summary_payload["auc_pretrain_frozen"] = frozen_auc
        if fine_tuned_auc is not None:
            summary_payload["auc_fine_tuned"] = fine_tuned_auc
        if selected_source is not None:
            summary_payload["selected_path"] = selected_source
        if selected_benchmark is not None:
            summary_payload["met_benchmark_selected"] = bool(selected_benchmark)
        _export_gate_env(gate_passed_flag)
        _wandb_log_safe(wb, summary_payload)

        multi_eval = len(evaluations) > 1
        for eval_res in evaluations:
            prefix = f"{getattr(eval_res, 'name', 'evaluation')}/" if multi_eval else ""
            payload = {"phase": "tox21", "status": "success"}
            payload.update(threshold_payload)
            payload.update(target_payload)
            payload[f"{prefix}mean_true"] = float(getattr(eval_res, "mean_true", 0.0))
            payload[f"{prefix}mean_rand"] = float(getattr(eval_res, "mean_random", 0.0))
            payload[f"{prefix}mean_pred"] = float(getattr(eval_res, "mean_pred", 0.0))
            payload[f"{prefix}encoder_source"] = getattr(eval_res, "encoder_source", "unknown")
            payload[f"{prefix}evaluation_mode"] = eval_mode

            benchmark_metric = getattr(eval_res, "benchmark_metric", None)
            benchmark_threshold = getattr(eval_res, "benchmark_threshold", None)
            met_benchmark = getattr(eval_res, "met_benchmark", None)
            if benchmark_metric is not None:
                payload[f"{prefix}benchmark_metric"] = benchmark_metric
            if benchmark_threshold is not None:
                payload[f"{prefix}benchmark_threshold"] = float(benchmark_threshold)
            if met_benchmark is not None:
                payload[f"{prefix}met_benchmark"] = bool(met_benchmark)
            payload[f"{prefix}tox21_gate_passed"] = bool(met_benchmark) if met_benchmark is not None else False

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
            **target_payload,
            "auc_summary": auc_summary,
            "selected_path": selected_source,
            "met_benchmark_selected": selected_benchmark,
            "tox21_gate_passed": bool(gate_passed_flag),
            "benchmark_metric_value": summary_payload.get("benchmark_metric_value"),
            "evaluations": [],
            "diagnostics": diagnostics,
            "evaluation_mode": eval_mode,
            "encoder_checkpoint": getattr(args, "encoder_checkpoint", None),
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
                    "tox21_gate_passed": (
                        bool(getattr(eval_res, "met_benchmark", None))
                        if getattr(eval_res, "met_benchmark", None) is not None
                        else None
                    ),
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

        try:
            stage_dir = os.path.join(report_dir, "stage-outputs")
            os.makedirs(stage_dir, exist_ok=True)
            stage_name = getattr(args, "encoder_source", None) or getattr(primary, "encoder_source", "run")
            stage_path = os.path.join(stage_dir, f"tox21_{stage_name}.json")
            stage_payload = {
                "encoder_source": getattr(args, "encoder_source", None),
                "evaluation_mode": eval_mode,
                "selected_path": selected_source,
                "selected_auc": auc_summary.get(selected_source) if selected_source else None,
                "met_benchmark": selected_benchmark,
                "tox21_gate_passed": bool(gate_passed_flag),
                **target_payload,
                "auc_summary": auc_summary,
                "evaluations": [
                    {
                        "encoder_source": getattr(ev, "encoder_source", getattr(ev, "name", "unknown")),
                        "roc_auc": auc_summary.get(
                            getattr(ev, "encoder_source", getattr(ev, "name", "unknown"))
                        ),
                        "met_benchmark": getattr(ev, "met_benchmark", None),
                        "tox21_gate_passed": (
                            bool(getattr(ev, "met_benchmark", None))
                            if getattr(ev, "met_benchmark", None) is not None
                            else None
                        ),
                        "manifest_path": manifest_lookup.get(
                            getattr(ev, "encoder_source", getattr(ev, "name", "unknown"))
                        ),
                    }
                    for ev in evaluations
                ],
                "diagnostics": diagnostics,
            }
            with open(stage_path, "w", encoding="utf-8") as fh:
                json.dump(stage_payload, fh, indent=2, sort_keys=True)
        except Exception:
            logger.debug("Failed to write tox21 stage outputs", exc_info=True)

    except Exception as exc:
        logger.exception("Tox21 case study failed")
        error_log = {"phase": "tox21", "status": "error", "error": str(exc)}
        error_log.update(threshold_payload)
        _wandb_log_safe(wb, error_log)
        try:
            _export_gate_env(False)
        except Exception:
            logger.debug("Failed to export failure gate status", exc_info=True)
        sys.exit(5)
    finally:
        try:
            if wb is not None and hasattr(wb, "finish"):
                wb.finish()
            elif wb is not None and hasattr(wb, "run") and hasattr(wb.run, "finish"):
                wb.run.finish()
        except Exception:
            pass

