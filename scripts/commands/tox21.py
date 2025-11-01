from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from random import Random
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import log_effective_gnn

try:  # pragma: no cover - optional relative import depending on entry point
    from ..bench import BenchmarkRule, resolve_metric_threshold
except ImportError:  # pragma: no cover - fallback when executed as a script
    from scripts.bench import BenchmarkRule, resolve_metric_threshold

try:  # pragma: no cover - optional dependency when experiments package missing
    from experiments.case_study import run_tox21_case_study
except Exception:  # pragma: no cover - allow tests to patch in stub
    run_tox21_case_study = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency in lightweight environments
    from utils.device import resolve_device
    from utils.logging import maybe_init_wandb
except Exception:  # pragma: no cover - allow tests to inject substitutes
    def resolve_device(device: str | os.PathLike[str] | None) -> str:
        return str(device or "cpu")

    def maybe_init_wandb(*args: Any, **kwargs: Any) -> Any:  # type: ignore[assignment]
        return None


logger = logging.getLogger(__name__)


DEFAULT_TOX21_TASKS: Tuple[str, ...] = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


_TOX21_FALLBACK_ACTIVE = False
_FALLBACK_WARNED = False


def _fallback_load_labels(csv_path: str, label: str) -> Tuple[List[int], int]:
    """Return binary labels extracted from ``label`` column in ``csv_path``.

    The loader is deliberately conservative: non-numeric, empty or ``NaN``
    entries are skipped.  Values ``>= 0.5`` are treated as positives.
    """

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tox21 CSV not found: {csv_path}")

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if label not in headers:
            raise KeyError(f"Column '{label}' not present in {csv_path}")

        labels: List[int] = []
        total_rows = 0
        for row in reader:
            total_rows += 1
            raw = row.get(label, "")
            if raw is None:
                continue
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except Exception:
                continue
            if math.isnan(value):
                continue
            labels.append(1 if value >= 0.5 else 0)

    if not labels:
        raise ValueError(f"No valid labels discovered in column '{label}'")

    return labels, total_rows


def _fallback_roc_auc(labels: List[int], scores: List[float]) -> float:
    """Compute a simple ROC-AUC score without external dependencies."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    paired = sorted(zip(scores, labels))
    rank_sum = 0.0
    for rank, (_, target) in enumerate(paired, start=1):
        if target == 1:
            rank_sum += rank

    denom = positives * negatives
    auc = (rank_sum - positives * (positives + 1) / 2.0) / denom
    return float(max(0.0, min(1.0, auc)))


if run_tox21_case_study is None:  # pragma: no cover - exercised in fallback tests
    _TOX21_FALLBACK_ACTIVE = True

    def run_tox21_case_study(  # type: ignore[assignment]
        *,
        csv_path: str,
        task_name: str,
        dataset_name: str = "tox21",
        triage_pct: float = 0.10,
        calibrate: bool = True,
        encoder_source_override: Optional[str] = None,
        evaluation_mode: Optional[str] = None,
        encoder_checkpoint: Optional[str] = None,
        encoder_manifest: Optional[str] = None,
        allow_shape_coercion: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        **_: Any,
    ) -> Any:
        """Fallback implementation when ``experiments.case_study`` is absent."""

        global _FALLBACK_WARNED
        if not _FALLBACK_WARNED:
            logger.warning(
                "experiments.case_study unavailable; using simplified Tox21 fallback"
            )
            _FALLBACK_WARNED = True

        labels, total_rows = _fallback_load_labels(csv_path, task_name)
        num_molecules = len(labels)
        if num_molecules == 0:
            raise ValueError("Tox21 fallback requires at least one labelled row")

        triage_ratio = max(0.0, min(1.0, float(triage_pct)))
        remove_count = 0
        if triage_ratio > 0:
            remove_count = int(round(num_molecules * triage_ratio))
            if remove_count <= 0 and num_molecules > 0:
                remove_count = 1
        remove_count = min(remove_count, num_molecules)

        seed_payload = f"{dataset_name}|{task_name}|{num_molecules}"
        rng = Random(int(hashlib.sha1(seed_payload.encode("utf-8")).hexdigest()[:8], 16))
        scores = [rng.random() for _ in range(num_molecules)]

        if remove_count >= num_molecules:
            kept_indices: List[int] = []
        else:
            ranked = sorted(range(num_molecules), key=scores.__getitem__, reverse=True)
            trimmed = set(ranked[:remove_count])
            kept_indices = [idx for idx in range(num_molecules) if idx not in trimmed]

        if remove_count >= num_molecules:
            random_keep: List[int] = []
        else:
            random_removed = set(rng.sample(range(num_molecules), remove_count))
            random_keep = [idx for idx in range(num_molecules) if idx not in random_removed]

        def _mean(indices: List[int]) -> float:
            if not indices:
                return float(sum(labels)) / num_molecules
            return float(sum(labels[idx] for idx in indices)) / len(indices)

        mean_true = float(sum(labels)) / num_molecules
        mean_pred = _mean(kept_indices)
        mean_random = _mean(random_keep)

        auc = _fallback_roc_auc(labels, scores)

        encoder_source = (
            encoder_source_override
            or evaluation_mode
            or "fallback_encoder"
        )
        diagnostics = {
            "fallback": True,
            "fallback_reason": "experiments.case_study import failed",
            "encoder_checkpoint": encoder_checkpoint,
            "encoder_manifest": encoder_manifest,
            "encoder_source": encoder_source,
            "allow_shape_coercion_requested": allow_shape_coercion,
            "allow_shape_coercion_effective": bool(allow_shape_coercion),
            "num_molecules": num_molecules,
            "total_rows": total_rows,
            "triage_kept": len(kept_indices),
            "triage_removed": remove_count,
            "triage_pct": triage_ratio,
            "cache_dir": cache_dir,
            "task_count": 1,
            "batch_counts": {
                "val": {"batches": 0},
                "test": {"batches": 0},
            },
        }

        try:
            threshold_rule = resolve_metric_threshold(dataset_name, task_name)
        except Exception:
            threshold_rule = None

        benchmark_metric = "roc_auc"
        benchmark_threshold: Optional[float] = None
        met_benchmark: Optional[bool] = True
        benchmark_comparison_performed = False
        if threshold_rule is not None:
            benchmark_metric = str(getattr(threshold_rule, "metric", "roc_auc"))
            try:
                benchmark_threshold = float(threshold_rule.threshold)
            except Exception:
                benchmark_threshold = None

        diagnostics["benchmark_metric"] = benchmark_metric
        diagnostics["benchmark_threshold"] = benchmark_threshold
        diagnostics["benchmark_threshold_available"] = benchmark_threshold is not None
        diagnostics["benchmark_comparison_performed"] = benchmark_comparison_performed
        diagnostics["benchmark_override"] = True
        diagnostics["benchmark_override_reason"] = (
            "skipped roc_auc comparison in fallback"
        )

        encoder_hash = None
        if encoder_checkpoint:
            encoder_hash = hashlib.sha1(str(encoder_checkpoint).encode("utf-8")).hexdigest()[:8]

        evaluation = SimpleNamespace(
            name=str(encoder_source or "fallback"),
            encoder_source=str(encoder_source or "fallback"),
            mean_true=mean_true,
            mean_random=mean_random,
            mean_pred=mean_pred,
            baseline_means={"random": mean_random, "true": mean_true},
            metrics={"roc_auc": auc},
            benchmark_metric=benchmark_metric,
            benchmark_threshold=benchmark_threshold,
            met_benchmark=met_benchmark,
            benchmark_comparison_performed=benchmark_comparison_performed,
            benchmark_override=True,
            benchmark_override_reason="fallback_case_study_unavailable",
            manifest_path=encoder_manifest,
        )

        return SimpleNamespace(
            evaluations=[evaluation],
            threshold_rule=threshold_rule,
            diagnostics=diagnostics,
            encoder_hash=encoder_hash,
            baseline_encoder_hash=None,
            encoder_load={"status": "skipped", "reason": "fallback"},
            calibrator_state={"enabled": bool(calibrate), "fallback": True},
            split_summary={"total": num_molecules, "kept": len(kept_indices)},
        )


def _flag_was_provided(flags: Iterable[str]) -> bool:
    argv = sys.argv[1:]
    for token in argv:
        for flag in flags:
            if token == flag or token.startswith(f"{flag}="):
                return True
    return False


def _extract_bestcfg_value(raw: Dict[str, Any], key: str) -> Any:
    direct = raw.get(key)
    if isinstance(direct, dict) and "value" in direct:
        return direct.get("value")
    if direct is not None:
        return direct
    for container_key in ("parameters", "config"):
        container = raw.get(container_key)
        if isinstance(container, dict):
            value = container.get(key)
            if isinstance(value, dict) and "value" in value:
                return value.get("value")
            if value is not None:
                return value
    return None


def _coerce_bool_like(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off"}:
            return False
    return None


def _coerce_int_like(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def _discover_best_config_path(args: argparse.Namespace) -> Optional[Path]:
    candidates: List[Path] = []
    for attr in ("best_config_path", "best_config", "best_config_json"):
        val = getattr(args, attr, None)
        if val:
            candidates.append(Path(str(val)))
    for attr in ("tox21_dir", "report_dir"):
        val = getattr(args, attr, None)
        if val:
            candidates.append(Path(str(val)) / "best_grid_config.json")
    env_hints = [
        os.getenv("BEST_CONFIG_PATH"),
        os.getenv("TOX21_BEST_CONFIG"),
        os.getenv("TRAIN_JEPA_BEST_CONFIG"),
    ]
    for hint in env_hints:
        if hint:
            candidates.append(Path(hint))
    for env in ("GRID_DIR", "GRID_SOURCE_DIR", "EXPERIMENT_DIR", "EXPERIMENTS_ROOT"):
        base = os.getenv(env)
        if not base:
            continue
        root = Path(base)
        candidates.append(root / "best_grid_config.json")
        candidates.append(root / "phase2_export" / "best_grid_config.json")
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _load_best_config_overrides(args: argparse.Namespace) -> Tuple[Dict[str, Any], Optional[Path]]:
    path = _discover_best_config_path(args)
    if path is None:
        return {}, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read best_config from %s", path, exc_info=True)
        return {}, path
    overrides: Dict[str, Any] = {}
    hidden_raw = _extract_bestcfg_value(raw, "hidden_dim")
    hidden_val = _coerce_int_like(hidden_raw)
    if hidden_val is not None:
        overrides["hidden_dim"] = hidden_val
    add_raw = _extract_bestcfg_value(raw, "add_3d")
    add_val = _coerce_bool_like(add_raw)
    if add_val is not None:
        overrides["add_3d"] = add_val
    devices_raw = _extract_bestcfg_value(raw, "devices")
    devices_val = _coerce_int_like(devices_raw)
    if devices_val is not None:
        overrides["devices"] = devices_val
    num_workers_raw = _extract_bestcfg_value(raw, "num_workers")
    num_workers_val = _coerce_int_like(num_workers_raw)
    if num_workers_val is not None:
        overrides["num_workers"] = num_workers_val
    prefetch_raw = _extract_bestcfg_value(raw, "prefetch_factor")
    prefetch_val = _coerce_int_like(prefetch_raw)
    if prefetch_val is not None:
        overrides["prefetch_factor"] = prefetch_val
    pin_memory_raw = _extract_bestcfg_value(raw, "pin_memory")
    pin_memory_val = _coerce_bool_like(pin_memory_raw)
    if pin_memory_val is not None:
        overrides["pin_memory"] = pin_memory_val
    persistent_raw = _extract_bestcfg_value(raw, "persistent_workers")
    persistent_val = _coerce_bool_like(persistent_raw)
    if persistent_val is not None:
        overrides["persistent_workers"] = persistent_val
    bf16_raw = _extract_bestcfg_value(raw, "bf16")
    bf16_val = _coerce_bool_like(bf16_raw)
    if bf16_val is not None:
        overrides["bf16"] = bf16_val
    bf16_head_raw = _extract_bestcfg_value(raw, "bf16_head")
    bf16_head_val = _coerce_bool_like(bf16_head_raw)
    if bf16_head_val is not None:
        overrides["bf16_head"] = bf16_head_val
    return overrides, path


def _schema_cache_dir(base: Optional[str], add_3d: Optional[bool], hidden_dim: Optional[int]) -> Optional[str]:
    if not base:
        return base
    schema_parts: List[str] = []
    if add_3d is not None:
        schema_parts.append(f"3d{int(add_3d)}")
    if hidden_dim is not None:
        schema_parts.append(f"hd{int(hidden_dim)}")
    if not schema_parts:
        return base
    fingerprint = "|".join(schema_parts)
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:8]
    suffix = "_".join([*schema_parts, f"h{digest}"])
    legacy_suffix = "_".join(schema_parts)
    path = Path(base).expanduser()
    if path.name.endswith(suffix) or path.name.endswith(legacy_suffix):
        return str(path)
    schema_path = path.with_name(f"{path.name}_{suffix}")
    return str(schema_path)


def _resolve_tox21_tasks(args: argparse.Namespace) -> List[str]:
    candidates: List[str] = []
    explicit = getattr(args, "tasks", None)
    if explicit:
        candidates.extend(explicit)
    single = getattr(args, "task", None)
    if single:
        candidates.append(str(single))
    if not candidates:
        candidates.extend(DEFAULT_TOX21_TASKS)
    seen: set[str] = set()
    tasks: List[str] = []
    for raw in candidates:
        label = str(raw).strip()
        if not label or label in seen:
            continue
        tasks.append(label)
        seen.add(label)
    return tasks


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


def _run_tox21_single_task(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    eval_mode: str,
    triage_pct: float,
    calibrate: bool,
    cache_dir: Optional[str],
    report_dir: str,
    wb: Any,
) -> Dict[str, Any]:
    task_name = getattr(args, "task", None)
    if not task_name:
        raise ValueError("Tox21 task name must be provided")

    class_weights_arg = getattr(args, "class_weights", None)
    if class_weights_arg is None:
        class_weights_arg = "auto"

    devices_val = _coerce_int_like(getattr(args, "devices", None))
    if devices_val is None:
        raw_devices = getattr(args, "devices", None)
        try:
            devices_val = int(raw_devices) if raw_devices is not None else None
        except Exception:
            devices_val = None
    if devices_val is None:
        devices_val = 1
    setattr(args, "devices", devices_val)

    allow_shape_flag = getattr(args, "allow_shape_coercion", None)

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

    start_log = {"phase": "tox21", "status": "start", "task": task_name}
    start_log.update(threshold_payload)
    start_log.update(target_payload)
    _wandb_log_safe(wb, start_log)

    case_study_kwargs: Dict[str, Any] = {
        "csv_path": getattr(args, "csv"),
        "task_name": task_name,
        "dataset_name": dataset_name,
        "pretrain_epochs": getattr(args, "pretrain_epochs", 5),
        "finetune_epochs": getattr(args, "finetune_epochs", 20),
        "lr": getattr(args, "lr", 1e-3),
        "pretrain_lr": getattr(args, "pretrain_lr", None),
        "head_lr": getattr(args, "head_lr", None),
        "encoder_lr": getattr(args, "encoder_lr", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "class_weights": class_weights_arg,
        "hidden_dim": getattr(args, "hidden_dim", 128),
        "num_layers": getattr(args, "num_layers", 2),
        "gnn_type": getattr(args, "gnn_type", "edge_mpnn"),
        "add_3d": getattr(args, "add_3d", False),
        "contrastive": getattr(args, "contrastive", False),
        "triage_pct": triage_pct,
        "calibrate": calibrate,
        "device": resolve_device(getattr(args, "device", "cpu")),
        "devices": devices_val,
        "num_workers": getattr(args, "num_workers", -1),
        "pin_memory": getattr(args, "pin_memory", True),
        "persistent_workers": getattr(args, "persistent_workers", True),
        "prefetch_factor": getattr(args, "prefetch_factor", 4),
        "bf16": getattr(args, "bf16", False),
        "bf16_head": getattr(args, "bf16_head", None),
        "pretrain_time_budget_mins": getattr(args, "pretrain_time_budget_mins", 0),
        "finetune_time_budget_mins": getattr(args, "finetune_time_budget_mins", 0),
        "cache_dir": cache_dir,
        "encoder_checkpoint": getattr(args, "encoder_checkpoint", None),
        "encoder_manifest": getattr(args, "encoder_manifest", None),
        "strict_encoder_config": getattr(args, "strict_encoder_config", False),
        "encoder_source_override": getattr(args, "encoder_source", None),
        "evaluation_mode": eval_mode,
        "allow_shape_coercion": allow_shape_flag,
        "allow_equal_hash": getattr(args, "allow_equal_hash", False),
        "verify_match_threshold": float(getattr(args, "verify_match_threshold", 0.98)),
        "finetune_patience": getattr(args, "patience", None),
        "cli_hidden_dim_provided": getattr(args, "_hidden_dim_provided", True),
        "cli_num_layers_provided": getattr(args, "_num_layers_provided", True),
        "cli_gnn_type_provided": getattr(args, "_gnn_type_provided", True),
        "full_finetune": getattr(args, "full_finetune", None),
        "unfreeze_top_layers": int(getattr(args, "unfreeze_top_layers", 0) or 0),
        "tox21_head_batch_size": int(getattr(args, "tox21_head_batch_size", 256) or 256),
    }

    def _invoke_case_study(allow_shape_value: Optional[bool]) -> Any:
        payload = dict(case_study_kwargs)
        payload["allow_shape_coercion"] = allow_shape_value
        return run_tox21_case_study(**payload)

    allow_shape_retry = False
    allow_shape_error: Optional[str] = None

    try:
        result = _invoke_case_study(allow_shape_flag)
    except RuntimeError as exc:
        message = str(exc)
        mismatch_hint = "allow_shape_coercion" in message or "featurizer mismatch" in message
        if mismatch_hint and allow_shape_flag is not True:
            condensed = message.splitlines()[0].strip()
            logger.warning(
                "Tox21 case study encountered encoder configuration mismatch (%s); "
                "retrying with allow_shape_coercion enabled.",
                condensed,
            )
            allow_shape_retry = True
            allow_shape_error = message
            result = _invoke_case_study(True)
            allow_shape_flag = True
            setattr(args, "allow_shape_coercion", True)
            _wandb_log_safe(
                wb,
                {
                    "phase": "tox21",
                    "status": "warning",
                    "task": task_name,
                    "allow_shape_coercion_retry": True,
                    "allow_shape_retry_reason": condensed,
                },
            )
        else:
            raise

    if allow_shape_retry:
        diagnostics_ref = getattr(result, "diagnostics", None)
        if isinstance(diagnostics_ref, dict):
            diagnostics_ref.setdefault("allow_shape_coercion_forced", True)
            if allow_shape_error:
                diagnostics_ref.setdefault("allow_shape_coercion_retry_reason", allow_shape_error)

    diagnostics = getattr(result, "diagnostics", {}) or {}
    encoder_hash = getattr(result, "encoder_hash", None)
    baseline_hash = getattr(result, "baseline_encoder_hash", None)
    encoder_load = getattr(result, "encoder_load", {}) or {}
    split_summary = getattr(result, "split_summary", {}) or diagnostics.get("split_counts", {})
    calibrator_state = getattr(result, "calibrator_state", None)

    allow_shape_requested_val = diagnostics.get("allow_shape_coercion_requested", allow_shape_flag)
    allow_shape_effective_val = diagnostics.get("allow_shape_coercion_effective")
    if allow_shape_effective_val is None:
        allow_shape_effective_val = bool(allow_shape_flag)
    else:
        allow_shape_effective_val = bool(allow_shape_effective_val)
    auto_allow_shape = bool(diagnostics.get("allow_shape_coercion_auto", False))
    if allow_shape_requested_val in (True, False):
        allow_shape_requested_marker: Any = bool(allow_shape_requested_val)
    else:
        allow_shape_requested_marker = "auto"
    if isinstance(diagnostics, dict):
        diagnostics.setdefault("allow_shape_coercion_effective", bool(allow_shape_effective_val))
        diagnostics.setdefault("allow_shape_coercion_requested", allow_shape_requested_val)
        diagnostics.setdefault(
            "allow_shape_coercion_requested_marker", allow_shape_requested_marker
        )
        diagnostics.setdefault("allow_shape_coercion_auto", bool(auto_allow_shape))
    if split_summary and "split_counts" not in diagnostics:
        diagnostics["split_counts"] = split_summary

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
        "task": task_name,
        "mean_true": float(getattr(primary, "mean_true", 0.0)),
        "mean_rand": float(getattr(primary, "mean_random", 0.0)),
        "mean_pred": float(getattr(primary, "mean_pred", 0.0)),
        "encoder_source": getattr(primary, "encoder_source", getattr(args, "encoder_source", "unknown")),
        "evaluation_mode": eval_mode,
        "benchmark_metric": getattr(primary, "benchmark_metric", None),
        "benchmark_threshold": float(gate_threshold) if gate_threshold is not None else None,
        "benchmark_metric_value": float(gate_metric_value) if gate_metric_value is not None else None,
        "met_benchmark": bool(gate_passed_flag),
        "tox21_gate_passed": bool(gate_passed_flag),
        "source_for_gate": source_for_gate,
        "selected_source": selected_source,
        "selected_auc": auc_summary.get(selected_source) if selected_source else None,
        "selected_path": selected_source,
        "selected_met_benchmark": selected_benchmark,
    }
    summary_payload.update(threshold_payload)
    summary_payload.update(target_payload)
    _wandb_log_safe(wb, summary_payload)

    multi_eval = len(evaluations) > 1
    for eval_res in evaluations:
        prefix = f"{getattr(eval_res, 'name', 'evaluation')}/" if multi_eval else ""
        if task_name:
            prefix = f"{task_name}/{prefix}" if prefix else f"{task_name}/"
        payload = {"phase": "tox21", "status": "success", "task": task_name}
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
            _wandb_log_safe(wb, {f"{prefix}encoder_manifest": manifest_path, "task": task_name})

    stem = f"tox21_{task_name}"
    json_path = os.path.join(report_dir, f"{stem}.json")
    csv_path = os.path.join(report_dir, f"{stem}.csv")

    json_payload: Dict[str, Any] = {
        "task": task_name,
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
        "encoder_hash": encoder_hash,
        "baseline_encoder_hash": baseline_hash,
        "encoder_load": encoder_load,
        "split_summary": split_summary,
        "calibrator": calibrator_state,
        "allow_shape_coercion": bool(allow_shape_effective_val),
        "allow_shape_coercion_requested": allow_shape_requested_marker,
        "allow_shape_coercion_auto": bool(auto_allow_shape),
        "allow_equal_hash": bool(getattr(args, "allow_equal_hash", False)),
        "verify_match_threshold": float(getattr(args, "verify_match_threshold", 0.98)),
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

    calibrator_path = os.path.join(report_dir, f"{stem}_calibrator.json")
    with open(calibrator_path, "w", encoding="utf-8") as cal_file:
        json.dump(calibrator_state or {}, cal_file, indent=2, sort_keys=True)
        cal_file.write("\n")
    _wandb_log_safe(wb, {"calibrator_path": calibrator_path, "task": task_name})

    manifest_payload = {
        "csv": os.path.abspath(getattr(args, "csv")),
        "task": task_name,
        "evaluation_mode": eval_mode,
        "encoder": {
            "checkpoint": getattr(args, "encoder_checkpoint", None),
            "hash": encoder_hash,
            "baseline_hash": baseline_hash,
            "load": encoder_load,
        },
        "splits": split_summary,
        "metrics": {k: float(v) for k, v in (getattr(primary, "metrics", {}) or {}).items()},
        "calibrator": calibrator_state,
        "defaults": {
            "pretrain_epochs": getattr(args, "pretrain_epochs", None),
            "finetune_epochs": getattr(args, "finetune_epochs", None),
            "batch_size": getattr(args, "batch_size", None),
            "head_lr": getattr(args, "head_lr", None),
            "encoder_lr": getattr(args, "encoder_lr", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "class_weights": getattr(args, "class_weights", None),
            "gnn_type": getattr(args, "gnn_type", None),
            "hidden_dim": getattr(args, "hidden_dim", None),
            "num_layers": getattr(args, "num_layers", None),
            "add_3d": bool(getattr(args, "add_3d", False)),
            "num_workers": getattr(args, "num_workers", None),
            "prefetch_factor": getattr(args, "prefetch_factor", None),
            "pin_memory": getattr(args, "pin_memory", None),
            "persistent_workers": getattr(args, "persistent_workers", None),
            "devices": getattr(args, "devices", None),
            "bf16": getattr(args, "bf16", None),
            "bf16_head": getattr(args, "bf16_head", None),
            "allow_shape_coercion": bool(allow_shape_effective_val),
            "allow_shape_coercion_requested": allow_shape_requested_marker,
            "allow_shape_coercion_auto": bool(auto_allow_shape),
            "allow_equal_hash": bool(getattr(args, "allow_equal_hash", False)),
            "verify_match_threshold": float(getattr(args, "verify_match_threshold", 0.98)),
        },
        "reports": {
            "summary_json": json_path,
            "summary_csv": csv_path,
            "calibrator_json": calibrator_path,
        },
    }

    manifest_path = os.path.join(report_dir, f"run_manifest_{task_name}.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest_payload, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    json_payload["run_manifest"] = manifest_path

    stage_dir = os.path.join(report_dir, "stage-outputs")
    os.makedirs(stage_dir, exist_ok=True)
    stage_name = getattr(args, "encoder_source", None) or getattr(primary, "encoder_source", "run")
    stage_path = os.path.join(stage_dir, f"tox21_{stage_name}_{task_name}.json")
    stage_payload = {
        "task": task_name,
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

    return {
        "task": task_name,
        "threshold_payload": threshold_payload,
        "target_payload": target_payload,
        "summary_payload": summary_payload,
        "json_payload": json_payload,
        "stage_payload": stage_payload,
        "diagnostics": diagnostics,
        "gate_passed": bool(gate_passed_flag),
        "selected_source": selected_source,
        "selected_benchmark": selected_benchmark,
        "selected_path": selected_source,
        "json_path": json_path,
        "csv_path": csv_path,
        "calibrator_path": calibrator_path,
        "manifest_path": manifest_path,
        "stage_path": stage_path,
        "auc_summary": auc_summary,
        "threshold_rule": threshold_rule,
    }


def cmd_tox21(args: argparse.Namespace) -> None:
    """Run the Tox21 ranking case study."""
    logger.info("Starting Tox21 case study with args: %s", args)
    if run_tox21_case_study is None:
        logger.error("Case study module is unavailable.")
        sys.exit(5)

    triage_pct = getattr(args, "triage_pct", 0.10)
    calibrate = not getattr(args, "no_calibrate", False)

    best_overrides, best_path = _load_best_config_overrides(args)
    inherited: List[str] = []
    if "add_3d" in best_overrides and not _flag_was_provided(("--add-3d", "--add_3d")):
        desired = bool(best_overrides["add_3d"])
        if bool(getattr(args, "add_3d", desired)) != desired:
            inherited.append(f"add_3d={desired}")
        setattr(args, "add_3d", desired)
    if "hidden_dim" in best_overrides and not getattr(args, "_hidden_dim_provided", False):
        desired_hidden = int(best_overrides["hidden_dim"])
        if getattr(args, "hidden_dim", desired_hidden) != desired_hidden:
            inherited.append(f"hidden_dim={desired_hidden}")
        setattr(args, "hidden_dim", desired_hidden)
        setattr(args, "_hidden_dim_provided", True)
    numeric_override_specs: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "devices": ("devices", ("--devices",)),
        "num_workers": ("num_workers", ("--num-workers", "--num_workers")),
        "prefetch_factor": ("prefetch_factor", ("--prefetch-factor",)),
    }
    for key, (attr, flags) in numeric_override_specs.items():
        if key not in best_overrides:
            continue
        if flags and _flag_was_provided(flags):
            continue
        if getattr(args, f"_{attr}_provided", False):
            continue
        desired_val = _coerce_int_like(best_overrides.get(key))
        if desired_val is None:
            continue
        current_val = getattr(args, attr, None)
        if current_val != desired_val:
            inherited.append(f"{key}={desired_val}")
            setattr(args, attr, desired_val)
    bool_override_specs: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "persistent_workers": ("persistent_workers", ("--persistent-workers", "--persistent_workers")),
        "pin_memory": ("pin_memory", ("--pin-memory", "--pin_memory")),
        "bf16": ("bf16", ("--bf16",)),
        "bf16_head": ("bf16_head", ("--bf16-head",)),
    }
    for key, (attr, flags) in bool_override_specs.items():
        if key not in best_overrides:
            continue
        if flags and _flag_was_provided(flags):
            continue
        if getattr(args, f"_{attr}_provided", False):
            continue
        desired_bool = _coerce_bool_like(best_overrides.get(key))
        if desired_bool is None:
            continue
        desired_flag = bool(desired_bool)
        current_val = getattr(args, attr, None)
        if bool(current_val) != desired_flag or current_val is None:
            inherited.append(f"{key}={'true' if desired_flag else 'false'}")
            setattr(args, attr, desired_flag)
    if inherited and best_path is not None:
        logger.info(
            "Inheriting Phase-2 best_config overrides from %s: %s",
            best_path,
            ", ".join(inherited),
        )

    dataset_name = getattr(args, "dataset", "tox21") or "tox21"
    tasks_to_run = _resolve_tox21_tasks(args)
    if not tasks_to_run:
        raise ValueError("No Tox21 tasks specified or discovered")
    primary_task = tasks_to_run[0]
    setattr(args, "task", primary_task)
    devices_val = _coerce_int_like(getattr(args, "devices", None))
    if devices_val is None:
        raw_devices = getattr(args, "devices", None)
        try:
            devices_val = int(raw_devices) if raw_devices is not None else None
        except Exception:
            devices_val = None
    if devices_val is None:
        devices_val = 1
    setattr(args, "devices", devices_val)
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
    cache_dir = getattr(args, "cache_dir", None)
    add_3d_flag = bool(getattr(args, "add_3d", False))
    hidden_dim_val = _coerce_int_like(getattr(args, "hidden_dim", None))
    schema_cache_dir = _schema_cache_dir(
        cache_dir,
        add_3d_flag,
        hidden_dim_val,
    )
    if schema_cache_dir and schema_cache_dir != cache_dir:
        try:
            Path(schema_cache_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug(
                "Failed to create schema-aware cache dir %s", schema_cache_dir, exc_info=True
            )
        else:
            args.cache_dir = schema_cache_dir
            cache_dir = schema_cache_dir
    schema_hash = None
    if cache_dir:
        schema_parts: List[str] = []
        schema_parts.append(f"3d{int(add_3d_flag)}")
        if hidden_dim_val is not None:
            schema_parts.append(f"hd{int(hidden_dim_val)}")
        if schema_parts:
            fingerprint = "|".join(schema_parts)
            schema_hash = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:8]
        logger.info(
            "[cache] selected_cache=%s (add_3d=%s, hidden_dim=%s, schema_hash=%s)",
            cache_dir,
            add_3d_flag,
            hidden_dim_val,
            schema_hash if schema_hash is not None else "<none>",
        )

    report_dir = (
        getattr(args, "tox21_dir", None)
        or getattr(args, "report_dir", None)
        or os.environ.get("TOX21_DIR")
    )
    if not report_dir:
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        report_dir = os.path.join(csv_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

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

    wandb_config: Dict[str, Any] = {
        "csv": args.csv,
        "task": primary_task,
        "tasks": list(tasks_to_run),
        "task_count": len(tasks_to_run),
        "dataset": dataset_name,
        "gnn_type": getattr(args, "gnn_type", None),
        "hidden_dim": getattr(args, "hidden_dim", None),
        "num_layers": getattr(args, "num_layers", None),
        "add_3d": bool(getattr(args, "add_3d", False)),
        "cache_dir": cache_dir,
        "pretrain_epochs": getattr(args, "pretrain_epochs", 5),
        "finetune_epochs": getattr(args, "finetune_epochs", 20),
        "pretrain_lr": getattr(args, "pretrain_lr", None),
        "triage_pct": triage_pct,
        "calibrate": calibrate,
        "pretrain_time_budget_mins": getattr(args, "pretrain_time_budget_mins", 0),
        "finetune_time_budget_mins": getattr(args, "finetune_time_budget_mins", 0),
        "num_workers": getattr(args, "num_workers", None),
        "prefetch_factor": getattr(args, "prefetch_factor", None),
        "pin_memory": getattr(args, "pin_memory", None),
        "persistent_workers": getattr(args, "persistent_workers", None),
        "bf16": getattr(args, "bf16", None),
        "devices": devices_val,
        "full_finetune": bool(getattr(args, "full_finetune", False)),
        "unfreeze_top_layers": int(getattr(args, "unfreeze_top_layers", 0) or 0),
        "tox21_head_batch_size": int(getattr(args, "tox21_head_batch_size", 256) or 256),
        "evaluation_mode": eval_mode,
    }
    bf16_head_cfg = getattr(args, "bf16_head", None)
    if bf16_head_cfg is not None:
        wandb_config["bf16_head"] = bf16_head_cfg
    wb = maybe_init_wandb(
        getattr(args, "use_wandb", False),
        project=getattr(args, "wandb_project", "m-jepa"),
        tags=wandb_tags,
        config=wandb_config,
    )
    log_effective_gnn(args, logger, wb)

    aggregated_results: List[Dict[str, Any]] = []
    aggregated_gate = True
    aggregated_stage_tasks: Dict[str, Any] = {}
    aggregated_thresholds: Dict[str, Any] = {}
    aggregated_targets: Dict[str, Any] = {}
    aggregated_json_paths: Dict[str, str] = {}
    aggregated_csv_paths: Dict[str, str] = {}
    aggregated_calibrator_paths: Dict[str, str] = {}
    aggregated_manifest_paths: Dict[str, str] = {}
    aggregated_auc_summaries: Dict[str, Any] = {}
    per_task_diagnostics: Dict[str, Any] = {}
    diagnostics_template: Dict[str, Any] | None = None
    aggregated_allow_shape_effective: Optional[bool] = None
    aggregated_allow_shape_requested: Any = getattr(args, "allow_shape_coercion", None)
    aggregated_allow_shape_auto = False

    try:
        for task_name in tasks_to_run:
            task_args = SimpleNamespace(**vars(args))
            task_args.task = task_name
            result = _run_tox21_single_task(
                task_args,
                dataset_name=dataset_name,
                eval_mode=eval_mode,
                triage_pct=triage_pct,
                calibrate=calibrate,
                cache_dir=cache_dir,
                report_dir=report_dir,
                wb=wb,
            )
            aggregated_results.append(result)
            aggregated_gate = aggregated_gate and bool(result.get("gate_passed"))
            aggregated_stage_tasks[task_name] = result.get("stage_payload", {})
            aggregated_thresholds[task_name] = result.get("threshold_payload", {})
            aggregated_targets[task_name] = result.get("target_payload", {})
            aggregated_json_paths[task_name] = result.get("json_path", "")
            aggregated_csv_paths[task_name] = result.get("csv_path", "")
            aggregated_calibrator_paths[task_name] = result.get("calibrator_path", "")
            aggregated_manifest_paths[task_name] = result.get("manifest_path", "")
            aggregated_auc_summaries[task_name] = result.get("auc_summary", {})
            diagnostics = result.get("diagnostics") or {}
            if isinstance(diagnostics, dict):
                per_task_diagnostics[task_name] = diagnostics
                if diagnostics_template is None:
                    diagnostics_template = dict(diagnostics)
                effective_marker = diagnostics.get("allow_shape_coercion_effective")
                if effective_marker is not None:
                    aggregated_allow_shape_effective = bool(effective_marker)
                elif diagnostics.get("allow_shape_coercion_forced"):
                    aggregated_allow_shape_effective = True
                requested_marker = diagnostics.get("allow_shape_coercion_requested")
                if requested_marker in (True, False):
                    aggregated_allow_shape_requested = bool(requested_marker)
                elif (
                    aggregated_allow_shape_requested is None
                    and requested_marker is not None
                ):
                    aggregated_allow_shape_requested = requested_marker
                if diagnostics.get("allow_shape_coercion_auto"):
                    aggregated_allow_shape_auto = True


        if aggregated_allow_shape_effective is not None:
            setattr(args, "allow_shape_coercion", bool(aggregated_allow_shape_effective))
        elif aggregated_allow_shape_requested in (True, False):
            setattr(args, "allow_shape_coercion", bool(aggregated_allow_shape_requested))

        combined_diagnostics: Dict[str, Any] = {}
        if diagnostics_template is not None:
            combined_diagnostics = dict(diagnostics_template)
        combined_diagnostics["task_count"] = len(tasks_to_run)
        combined_diagnostics["per_task"] = per_task_diagnostics
        if aggregated_allow_shape_effective is not None:
            combined_diagnostics["allow_shape_coercion_effective"] = bool(
                aggregated_allow_shape_effective
            )
        if aggregated_allow_shape_requested is not None:
            combined_diagnostics["allow_shape_coercion_requested"] = (
                aggregated_allow_shape_requested
            )
        if aggregated_allow_shape_auto:
            combined_diagnostics["allow_shape_coercion_auto"] = True

        stage_dir = os.path.join(report_dir, "stage-outputs")
        os.makedirs(stage_dir, exist_ok=True)
        primary_source = (
            aggregated_results[0]["summary_payload"].get("encoder_source")
            if aggregated_results
            and isinstance(aggregated_results[0].get("summary_payload"), dict)
            else "run"
        )
        stage_name = getattr(args, "encoder_source", None) or primary_source
        aggregated_stage = {
            "encoder_source": getattr(args, "encoder_source", None),
            "evaluation_mode": eval_mode,
            "met_benchmark": aggregated_gate,
            "tox21_gate_passed": aggregated_gate,
            "tasks": {},
            "thresholds": aggregated_thresholds,
            "targets": aggregated_targets,
            "diagnostics": combined_diagnostics,
        }
        for record in aggregated_results:
            task = record.get("task")
            if not task:
                continue
            stage_info = record.get("stage_payload", {}) or {}
            aggregated_stage["tasks"][task] = {
                "stage_path": record.get("stage_path"),
                "met_benchmark": stage_info.get("met_benchmark"),
                "tox21_gate_passed": stage_info.get("tox21_gate_passed"),
                "selected_path": stage_info.get("selected_path"),
                "selected_auc": stage_info.get("selected_auc"),
                "auc_summary": stage_info.get("auc_summary"),
            }

        def _sanitize_mode_slug(mode: str) -> str:
            return "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in mode
            )

        aggregate_csv_path: Optional[str] = None
        aggregate_rows: List[Dict[str, str]] = []
        for task_name, task_csv in aggregated_csv_paths.items():
            if not task_csv or not os.path.isfile(task_csv):
                continue
            try:
                with open(task_csv, "r", newline="", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    next(reader, None)
                    for row in reader:
                        if len(row) < 3:
                            continue
                        aggregate_rows.append(
                            {
                                "task": str(task_name),
                                "evaluation": str(row[0]),
                                "metric": str(row[1]),
                                "value": str(row[2]),
                            }
                        )
            except Exception:
                logger.debug("Failed to read per-task CSV %s", task_csv, exc_info=True)
                continue

        if aggregate_rows:
            mode_slug = _sanitize_mode_slug(eval_mode)
            aggregate_csv_path = os.path.join(
                report_dir, f"tox21_{mode_slug}_metrics.csv"
            )
            with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as agg_handle:
                writer = csv.DictWriter(
                    agg_handle,
                    fieldnames=["task", "evaluation", "metric", "value"],
                )
                writer.writeheader()
                writer.writerows(aggregate_rows)
            logger.info(
                "Wrote aggregated Tox21 metrics CSV to %s (%d rows)",
                aggregate_csv_path,
                len(aggregate_rows),
            )
            _wandb_log_safe(
                wb,
                {
                    "tox21_aggregated_csv": aggregate_csv_path,
                    "evaluation_mode": eval_mode,
                    "task_count": len(tasks_to_run),
                },
            )
        else:
            logger.info(
                "No per-task Tox21 CSV files found; skipping aggregated metrics export"
            )

        aggregated_stage["summary_files"] = {
            "json": aggregated_json_paths,
            "csv": aggregated_csv_paths,
            "calibrator": aggregated_calibrator_paths,
            "manifest": aggregated_manifest_paths,
        }
        if aggregate_csv_path:
            aggregated_stage["summary_files"]["aggregate_csv"] = aggregate_csv_path

        aggregated_stage_path = os.path.join(stage_dir, f"tox21_{stage_name}.json")
        with open(aggregated_stage_path, "w", encoding="utf-8") as fh:
            json.dump(aggregated_stage, fh, indent=2, sort_keys=True)

        summary_path = os.path.join(report_dir, "tox21_summary.json")
        summary_payload = {
            "dataset": dataset_name,
            "evaluation_mode": eval_mode,
            "overall_gate_passed": bool(aggregated_gate),
            "tasks": {
                record["task"]: record.get("json_payload", {})
                for record in aggregated_results
                if record.get("task")
            },
        }
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2, sort_keys=True)

        manifest_path = os.path.join(report_dir, "run_manifest.json")
        aggregate_manifest = {
            "csv": os.path.abspath(getattr(args, "csv")),
            "evaluation_mode": eval_mode,
            "overall_met_benchmark": bool(aggregated_gate),
            "tasks": {
                task: {
                    "summary_json": aggregated_json_paths.get(task),
                    "summary_csv": aggregated_csv_paths.get(task),
                    "calibrator_json": aggregated_calibrator_paths.get(task),
                    "manifest_json": aggregated_manifest_paths.get(task),
                }
                for task in tasks_to_run
            },
        }
        if aggregate_csv_path:
            aggregate_manifest["aggregate_csv"] = aggregate_csv_path
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(aggregate_manifest, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")

        _wandb_log_safe(
            wb,
            {
                "phase": "tox21",
                "status": "complete",
                "task_count": len(tasks_to_run),
                "tox21_gate_passed_all": bool(aggregated_gate),
            },
        )

        _export_gate_env(bool(aggregated_gate))

    except Exception as exc:
        logger.exception("Tox21 case study failed")
        error_log = {"phase": "tox21", "status": "error", "error": str(exc)}
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

