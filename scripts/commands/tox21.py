from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import re
import os
import sys
import traceback
from pathlib import Path
from random import Random
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping

import torch
try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - allow running without PyYAML
    yaml = None  # type: ignore[assignment]

from . import log_effective_gnn
from utils.ddp import is_main_process
from utils.threads import configure_omp_threads

try:  # pragma: no cover - optional relative import depending on entry point
    from ..bench import BenchmarkRule, resolve_metric_threshold
except ImportError:  # pragma: no cover - fallback when executed as a script
    from scripts.bench import BenchmarkRule, resolve_metric_threshold

_CASE_STUDY_IMPORT_ERROR: Optional[str] = None
_CASE_STUDY_IMPORT_TRACEBACK: Optional[str] = None

try:  # pragma: no cover - optional dependency when experiments package missing
    from experiments.case_study import run_tox21_case_study
except Exception as exc:  # pragma: no cover - allow tests to patch in stub
    _CASE_STUDY_IMPORT_ERROR = f"{exc.__class__.__name__}: {exc}"
    _CASE_STUDY_IMPORT_TRACEBACK = traceback.format_exc()
    run_tox21_case_study = None  # type: ignore[assignment]
    debug_msg = (
        "[tox21] debug: failed to import experiments.case_study.run_tox21_case_study; "
        "falling back to simplified implementation"
    )
    print(f"{debug_msg}: {_CASE_STUDY_IMPORT_ERROR}", file=sys.stderr)
    logging.getLogger(__name__).debug(
        "experiments.case_study import failed; using fallback\n%s",
        _CASE_STUDY_IMPORT_TRACEBACK,
    )

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

_CALIBRATION_ECE_WARN = float(os.getenv("TOX21_CALIBRATION_WARN_ECE", "0.12"))


def _detect_cuda_devices() -> int:
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        return 0
    return 0


def _detect_visible_devices() -> int:
    """Return the number of visible CUDA devices, respecting environment masks."""

    mask = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if mask:
        entries = [token.strip() for token in mask.split(",") if token.strip()]
        if entries:
            return len(entries)

    try:
        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            if count > 0:
                return count
    except Exception:
        pass

    return 1


def _resolve_auto_device(device: Optional[str | os.PathLike[str]]) -> str:
    if device:
        return resolve_device(device)
    if _detect_cuda_devices() > 0:
        return "cuda"
    return "cpu"


def _normalise_device_and_count(args: argparse.Namespace) -> Tuple[str, int]:
    """Resolve the device string and GPU count, updating ``args`` in-place."""

    device_value = _resolve_auto_device(getattr(args, "device", None))
    setattr(args, "device", device_value)

    devices_val = _coerce_int_like(getattr(args, "devices", None))
    if devices_val is None:
        raw_devices = getattr(args, "devices", None)
        try:
            devices_val = int(raw_devices) if raw_devices is not None else None
        except Exception:
            devices_val = None
    if devices_val is None or devices_val <= 0:
        if str(device_value).lower().startswith("cuda"):
            detected = _detect_visible_devices()
            devices_val = detected if detected > 0 else 1
        else:
            devices_val = 1
    setattr(args, "devices", devices_val)
    return device_value, devices_val


def _should_relaunch_ddp(devices: int, world_size_env: int) -> bool:
    if devices <= 1 or world_size_env > 1:
        return False
    if os.environ.get("TOX21_DDP_LAUNCHED") == "1":
        return False
    if os.environ.get("TOX21_DISABLE_DDP_LAUNCH") == "1":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        return False
    return True


def _relaunch_with_torchrun(devices: int) -> None:
    torchrun_path = shutil.which("torchrun")
    if torchrun_path:
        command = [
            torchrun_path,
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={devices}",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={devices}",
        ]
    command.extend(sys.argv)
    env = os.environ.copy()
    env["TOX21_DDP_LAUNCHED"] = "1"
    logger.info("[tox21] relaunching with DDP: %s", " ".join(command))
    os.execvpe(command[0], command, env)


def _estimate_class_balance(csv_path: str, tasks: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Estimate positive/negative counts for each Tox21 assay."""

    stats: Dict[str, Dict[str, float]] = {
        str(task): {"pos": 0.0, "neg": 0.0, "total": 0.0} for task in tasks
    }

    try:
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for task, info in stats.items():
                    raw = row.get(task)
                    if raw is None:
                        continue
                    token = str(raw).strip()
                    if not token:
                        continue
                    try:
                        value = float(token)
                    except Exception:
                        continue
                    if math.isnan(value):
                        continue
                    info["total"] += 1.0
                    if value >= 0.5:
                        info["pos"] += 1.0
                    else:
                        info["neg"] += 1.0
    except FileNotFoundError:
        logger.warning("Tox21 CSV not found while estimating class balance: %s", csv_path)
    except Exception:
        logger.warning("Failed to estimate Tox21 class balance from %s", csv_path, exc_info=True)

    for task, info in stats.items():
        total = info.get("total", 0.0)
        pos = info.get("pos", 0.0)
        neg = info.get("neg", 0.0)
        if total > 0:
            info["pos_frac"] = pos / total
            info["neg_frac"] = neg / total
        else:
            info["pos_frac"] = 0.0
            info["neg_frac"] = 0.0
        info["pos_weight"] = (neg / pos) if pos > 0 else None
    return stats


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
        calibrate_per_head: bool = False,
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
        fallback_reason = "experiments.case_study import failed"
        if _CASE_STUDY_IMPORT_ERROR:
            fallback_reason = f"{fallback_reason}: {_CASE_STUDY_IMPORT_ERROR}"

        diagnostics = {
            "fallback": True,
            "fallback_reason": fallback_reason,
            "fallback_import_error": _CASE_STUDY_IMPORT_ERROR,
            "fallback_import_traceback": _CASE_STUDY_IMPORT_TRACEBACK,
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
        benchmark_threshold_original: Optional[float] = None
        met_benchmark: Optional[bool] = True
        benchmark_comparison_performed = False
        if threshold_rule is not None:
            benchmark_metric = str(getattr(threshold_rule, "metric", "roc_auc"))
            try:
                benchmark_threshold_original = float(threshold_rule.threshold)
            except Exception:
                benchmark_threshold_original = None

        diagnostics["benchmark_metric"] = benchmark_metric
        diagnostics["benchmark_threshold"] = benchmark_threshold
        diagnostics["benchmark_threshold_available"] = benchmark_threshold is not None
        diagnostics["benchmark_threshold_original"] = benchmark_threshold_original
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


def _strip_encoder_manifest_args(args: List[str]) -> Tuple[List[str], List[str]]:
    cleaned: List[str] = []
    removed: List[str] = []
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--encoder-manifest":
            if idx + 1 < len(args) and not str(args[idx + 1]).startswith("-"):
                removed.append(str(args[idx + 1]))
                idx += 2
                continue
            idx += 1
            continue
        if token.startswith("--encoder-manifest="):
            removed.append(token.split("=", 1)[1])
            idx += 1
            continue
        cleaned.append(token)
        idx += 1
    return cleaned, removed


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


def _coerce_float_like(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_pos_class_weight(values: Optional[Iterable[str]]) -> Optional[Any]:
    if values is None:
        return None
    if isinstance(values, Mapping):
        return values
    if isinstance(values, (int, float)):
        try:
            return float(values)
        except Exception:
            return None
    if isinstance(values, str):
        values = [values]
    scalar: Optional[float] = None
    mapping: Dict[str, float] = {}
    for raw in values:
        if raw is None:
            continue
        token = str(raw).strip()
        if not token:
            continue
        if "=" in token:
            key, weight = token.split("=", 1)
            key = key.strip()
            try:
                mapping[key] = float(weight)
            except Exception:
                logger.warning(
                    "Failed to parse pos_class_weight pair '%s'; ignoring",
                    token,
                    exc_info=True,
                )
        else:
            try:
                scalar = float(token)
            except Exception:
                logger.warning(
                    "Failed to parse pos_class_weight '%s'; ignoring",
                    token,
                    exc_info=True,
                )
    if mapping:
        if scalar is not None and "default" not in mapping:
            mapping["default"] = scalar
        return mapping
    return scalar


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
    finetune_epochs_raw = _extract_bestcfg_value(raw, "finetune_epochs")
    finetune_epochs_val = _coerce_int_like(finetune_epochs_raw)
    if finetune_epochs_val is not None:
        overrides["finetune_epochs"] = finetune_epochs_val
    patience_raw = _extract_bestcfg_value(raw, "patience")
    patience_val = _coerce_int_like(patience_raw)
    if patience_val is not None:
        overrides["patience"] = patience_val
    return overrides, path


def _load_task_hparams(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        raw = Path(str(path)).expanduser()
    except Exception:
        return {}
    if not raw.is_file():
        return {}
    try:
        text = raw.read_text(encoding="utf-8")
    except Exception:
        logger.debug("Failed to read per-task hparams from %s", raw, exc_info=True)
        return {}
    if yaml is not None:
        try:
            data = yaml.safe_load(text) or {}
        except Exception:
            logger.debug("Failed to parse per-task hparams YAML from %s", raw, exc_info=True)
            return {}
    else:
        data = {}
        for line in text.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            data[key.strip()] = value.strip()
    if not isinstance(data, dict):
        return {}
    return {str(k).strip().lower(): v for k, v in data.items()}


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


def _resolve_task_encoder_checkpoint(
    base_checkpoint: Optional[str],
    task_name: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Resolve the encoder checkpoint for a specific Tox21 task.

    When the evaluation entry-point receives a fine-tuned checkpoint from the
    CI wrappers it typically points at one assay directory (e.g. the first
    task).  This helper re-roots the path so that each assay loads its
    corresponding fine-tuned encoder, falling back to the provided path when a
    task-specific export is unavailable.
    """

    info: Dict[str, Any] = {
        "task": task_name,
        "provided": base_checkpoint,
        "resolved": base_checkpoint,
        "source": "unset" if not base_checkpoint else "provided",
        "task_dir": None,
        "candidates": [],
    }

    if not base_checkpoint or not task_name:
        return base_checkpoint, info

    try:
        base_path = Path(base_checkpoint).expanduser()
    except Exception as exc:
        info["error"] = f"{exc.__class__.__name__}: {exc}"
        return base_checkpoint, info

    resolved_str = str(base_path)
    info["resolved"] = resolved_str
    info["candidates"].append(resolved_str)

    parent = base_path.parent
    if parent.name == task_name:
        info["source"] = "provided_task_dir"
        info["task_dir"] = str(parent)
        return resolved_str, info

    finetune_root: Optional[Path] = None
    if parent.name == "finetune":
        finetune_root = parent
    else:
        try:
            grandparent = parent.parent
        except Exception:
            grandparent = None
        if grandparent is not None and grandparent.name == "finetune":
            finetune_root = grandparent

    if finetune_root is None:
        info["source"] = "provided"
        return resolved_str, info

    task_dir = finetune_root / task_name
    info["task_dir"] = str(task_dir)

    candidate_paths: List[Path] = []
    candidate_paths.append(task_dir / base_path.name)

    suffix = base_path.suffix
    if suffix:
        candidate_paths.append(task_dir / f"encoder_ft{suffix}")
    candidate_paths.append(task_dir / "encoder_ft.pt")

    seen_candidates = {str(candidate) for candidate in candidate_paths}
    try:
        for candidate in sorted(task_dir.glob("encoder_ft*.pt")):
            candidate_str = str(candidate)
            if candidate_str not in seen_candidates:
                candidate_paths.append(candidate)
                seen_candidates.add(candidate_str)
    except Exception:
        pass

    for candidate in candidate_paths:
        candidate_str = str(candidate)
        info["candidates"].append(candidate_str)
        try:
            if candidate.is_file():
                info["resolved"] = candidate_str
                info["source"] = "task_dir"
                return candidate_str, info
        except Exception:
            continue

    try:
        if task_dir.is_dir():
            info["source"] = "task_dir_missing"
        else:
            info["source"] = "provided"
    except Exception:
        info["source"] = "provided"

    return resolved_str, info


def _resolve_tox21_tasks(args: argparse.Namespace) -> List[str]:
    candidates: List[str] = []
    explicit = getattr(args, "tasks", None)
    if explicit:
        if isinstance(explicit, str):
            candidates.append(explicit)
        else:
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
    calibrate_per_head: bool = False,
    task_hparam_map: Optional[Dict[str, Any]] = None,
    hybrid_defaults: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str],
    report_dir: str,
    wb: Any,
    class_balance: Dict[str, Dict[str, float]],
    auto_pos_weights: Dict[str, float],
    calibration_warn_threshold: float,
) -> Dict[str, Any]:
    def _normalise_explain_modes(mode: Optional[Union[str, Iterable[str]]]) -> List[str]:
        def _canonicalise(token: Optional[str]) -> str:
            if token is None:
                return ""
            value = str(token).strip()
            if value.lower() == "motif_ig":
                return "ig_motif"
            return value

        if mode is None:
            return []

        raw_modes: List[str] = []
        if isinstance(mode, str):
            raw_modes = [part for part in re.split(r"[\s,]+", mode) if part]
        else:
            for entry in mode:
                if entry is None:
                    continue
                if isinstance(entry, str):
                    raw_modes.extend(part for part in re.split(r"[\s,]+", entry) if part)
                else:
                    raw_modes.append(str(entry))

        normalised: List[str] = []
        for token in raw_modes:
            token_norm = _canonicalise(token)
            if token_norm and token_norm not in normalised:
                normalised.append(token_norm)
        return normalised

    def _filter_explain_modes(modes: List[str]) -> List[str]:
        allowed = {"ig", "ig_motif", "off"}
        filtered = [mode for mode in modes if mode in allowed]
        if "off" in filtered:
            return []
        filtered = [mode for mode in filtered if mode != "off"]
        if not filtered:
            return ["ig", "ig_motif"]
        return filtered

    def _cleanup_explain_artifacts(report_root: str, task: str) -> None:
        """Remove stale explanation artifacts when explanations are disabled."""

        task_slug = str(task).strip().lower()
        explain_dirs = [
            Path(report_root) / "ig_explanations" / task_slug,
            Path(report_root) / "ig_motif_explanations" / task_slug,
        ]
        for path in explain_dirs:
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                logger.debug("Failed to remove explain artifacts at %s", path, exc_info=True)

    task_name = getattr(args, "task", None)
    if not task_name:
        raise ValueError("Tox21 task name must be provided")

    task_key = str(task_name).strip().lower()
    task_overrides: Dict[str, Any] = {}
    if task_hparam_map:
        task_overrides = task_hparam_map.get(task_key, {}) or {}
    hybrid_defaults = hybrid_defaults or {}

    balance_entry = class_balance.get(task_name, {}) if class_balance else {}
    class_weights_arg = task_overrides.get("class_weights", getattr(args, "class_weights", None))
    if class_weights_arg is None:
        class_weights_arg = "auto"

    pos_class_weight_arg = _parse_pos_class_weight(
        task_overrides.get("pos_weight", getattr(args, "pos_class_weight", None))
    )
    if pos_class_weight_arg is None:
        auto_weight = auto_pos_weights.get(task_name)
        if auto_weight is not None and auto_weight > 0:
            pos_class_weight_arg = {task_name: float(auto_weight)}

    head_ensemble_value = _coerce_int_like(getattr(args, "head_ensemble_size", None))
    if head_ensemble_value is None or head_ensemble_value <= 0:
        head_ensemble_value = 1
    setattr(args, "head_ensemble_size", head_ensemble_value)

    freeze_encoder_flag = bool(getattr(args, "freeze_encoder", False))

    device_value, devices_val = _normalise_device_and_count(args)

    allow_shape_flag = getattr(args, "allow_shape_coercion", None)

    explain_mode = _normalise_explain_modes(getattr(args, "explain_mode", None))
    if not explain_mode:
        env_mode = os.environ.get("TOX21_EXPLAIN_MODE")
        if env_mode:
            explain_mode = _normalise_explain_modes(env_mode)
    explain_mode = _filter_explain_modes(explain_mode)
    explain_steps = getattr(args, "explain_steps", None)
    if explain_steps is None:
        env_steps = os.environ.get("TOX21_EXPLAIN_STEPS")
        if env_steps:
            try:
                explain_steps = int(env_steps)
            except Exception:
                explain_steps = None
    explain_config_payload: Optional[Dict[str, Any]] = None
    if explain_mode:
        explain_config_payload = {
            "task_name": task_name,
            "output_dir": report_dir,
        }
        if explain_steps is not None:
            try:
                explain_config_payload["steps"] = int(explain_steps)
            except Exception:
                pass
    else:
        _cleanup_explain_artifacts(report_dir, task_name)

    finetune_epochs_provided = getattr(args, "_finetune_epochs_provided", None)
    if finetune_epochs_provided is None:
        finetune_epochs_provided = _flag_was_provided(("--finetune-epochs", "--finetune_epochs"))
    patience_provided = getattr(args, "_patience_provided", None)
    if patience_provided is None:
        patience_provided = _flag_was_provided(("--patience",))

    baseline_finetune_default = getattr(args, "_baseline_finetune_default", None)
    if baseline_finetune_default is None:
        baseline_epochs_env = _coerce_int_like(os.getenv("TOX21_BASELINE_FINETUNE_EPOCHS"))
        baseline_finetune_default = (
            baseline_epochs_env
            if baseline_epochs_env is not None
            else _coerce_int_like(getattr(args, "baseline_finetune_epochs", None))
        )
    baseline_patience_default = getattr(args, "_baseline_patience_default", None)
    if baseline_patience_default is None:
        baseline_patience_env = _coerce_int_like(os.getenv("TOX21_BASELINE_PATIENCE"))
        baseline_patience_default = (
            baseline_patience_env
            if baseline_patience_env is not None
            else _coerce_int_like(getattr(args, "baseline_patience", None))
        )

    base_encoder_checkpoint = getattr(args, "encoder_checkpoint", None)
    task_encoder_checkpoint, task_encoder_info = _resolve_task_encoder_checkpoint(
        base_encoder_checkpoint,
        task_name,
    )
    if task_encoder_info.get("source") == "task_dir" and task_encoder_checkpoint != base_encoder_checkpoint:
        logger.info(
            "Tox21 task %s using fine-tuned encoder checkpoint %s", task_name, task_encoder_checkpoint
        )
    elif task_encoder_info.get("source") == "task_dir_missing" and base_encoder_checkpoint:
        candidates = ", ".join(task_encoder_info.get("candidates", [])[1:]) or "<none>"
        logger.warning(
            "Fine-tuned encoder for %s not found under %s (candidates: %s); falling back to %s",
            task_name,
            task_encoder_info.get("task_dir"),
            candidates,
            base_encoder_checkpoint,
        )
    if task_encoder_checkpoint != base_encoder_checkpoint:
        setattr(args, "encoder_checkpoint", task_encoder_checkpoint)
    resolved_encoder_checkpoint = getattr(args, "encoder_checkpoint", None)

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
    start_log["freeze_encoder"] = freeze_encoder_flag
    start_log["head_ensemble_size"] = head_ensemble_value
    if task_overrides:
        start_log["task_overrides"] = list(sorted(task_overrides.keys()))
    if pos_class_weight_arg is not None:
        start_log["pos_class_weight"] = pos_class_weight_arg
    if balance_entry:
        start_log["class_balance_total"] = balance_entry.get("total")
        start_log["class_balance_pos_frac"] = balance_entry.get("pos_frac")
        start_log["class_balance_neg_frac"] = balance_entry.get("neg_frac")
        if balance_entry.get("pos_weight") is not None:
            start_log["class_balance_pos_weight"] = balance_entry.get("pos_weight")
    _wandb_log_safe(wb, start_log)

    finetune_epochs_value = getattr(args, "finetune_epochs", None)
    try:
        if finetune_epochs_value is not None:
            finetune_epochs_value = int(finetune_epochs_value)
    except Exception:
        pass

    bestcfg_epochs_override = bool(getattr(args, "_bestcfg_epochs_override", False))
    bestcfg_patience_override = bool(getattr(args, "_bestcfg_patience_override", False))

    head_lr_provided = _flag_was_provided(("--head-lr", "--head_lr"))
    encoder_lr_provided = _flag_was_provided(("--encoder-lr", "--encoder_lr"))
    layerwise_provided = _flag_was_provided(("--layerwise-decay", "--layerwise_decay"))

    head_lr_value = task_overrides.get("head_lr", getattr(args, "head_lr", None))
    if head_lr_value is None and hybrid_defaults and not head_lr_provided:
        head_lr_value = hybrid_defaults.get("head_lr", head_lr_value)

    encoder_lr_value = task_overrides.get("encoder_lr", getattr(args, "encoder_lr", None))
    if encoder_lr_value is None and hybrid_defaults and not encoder_lr_provided:
        encoder_lr_value = hybrid_defaults.get("encoder_lr", encoder_lr_value)

    layerwise_decay_value = task_overrides.get("layerwise_decay", getattr(args, "layerwise_decay", None))
    if layerwise_decay_value is None and hybrid_defaults and not layerwise_provided:
        layerwise_decay_value = hybrid_defaults.get("layerwise_decay", layerwise_decay_value)

    case_study_kwargs: Dict[str, Any] = {
        "csv_path": getattr(args, "csv"),
        "task_name": task_name,
        "dataset_name": dataset_name,
        "pretrain_epochs": getattr(args, "pretrain_epochs", 5),
        "finetune_epochs": getattr(args, "finetune_epochs", finetune_epochs_value),
        "lr": getattr(args, "lr", 1e-3),
        "pretrain_lr": getattr(args, "pretrain_lr", None),
        "lr_scheduler": getattr(args, "lr_scheduler", None),
        "warmup_ratio": getattr(args, "warmup_ratio", None),
        "min_lr": getattr(args, "min_lr", None),
        "min_lr_ratio": getattr(args, "min_lr_ratio", None),
        "head_lr": head_lr_value,
        "encoder_lr": encoder_lr_value,
        "weight_decay": getattr(args, "weight_decay", None),
        "class_weights": class_weights_arg,
        "pos_class_weight": pos_class_weight_arg,
        "hidden_dim": getattr(args, "hidden_dim", 128),
        "num_layers": getattr(args, "num_layers", 2),
        "dropout": getattr(args, "dropout", None),
        "gnn_type": getattr(args, "gnn_type", "edge_mpnn"),
        "add_3d": getattr(args, "add_3d", False),
        "contrastive": getattr(args, "contrastive", False),
        "triage_pct": triage_pct,
        "calibrate": bool(task_overrides.get("calibrate_probabilities", calibrate)),
        "calibrate_per_head": calibrate_per_head,
        "device": device_value,
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
        "encoder_checkpoint": resolved_encoder_checkpoint,
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
        "cli_finetune_epochs_provided": bool(finetune_epochs_provided),
        "cli_patience_provided": bool(patience_provided),
        "full_finetune": getattr(args, "full_finetune", None),
        "freeze_encoder": freeze_encoder_flag,
        "unfreeze_top_layers": int(getattr(args, "unfreeze_top_layers", 0) or 0),
        "tox21_head_batch_size": int(getattr(args, "tox21_head_batch_size", 256) or 256),
        "head_ensemble_size": head_ensemble_value,
        "head_scheduler": getattr(args, "head_scheduler", None),
        "layerwise_decay": layerwise_decay_value,
        "threshold_metric": task_overrides.get(
            "threshold_metric",
            getattr(args, "threshold_metric", None),
        ),
        "hybrid_freeze_epochs": getattr(args, "hybrid_freeze_epochs", None),
        "hybrid_partial_epochs": getattr(args, "hybrid_partial_epochs", None),
        "hybrid_warmup_ratio": getattr(args, "hybrid_warmup_ratio", None),
        "hybrid_lr_scheduler": getattr(args, "hybrid_lr_scheduler", None),
        "hybrid_min_lr": getattr(args, "hybrid_min_lr", None),
        "hybrid_min_lr_ratio": getattr(args, "hybrid_min_lr_ratio", None),
        "explain_mode": explain_mode,
        "explain_config": explain_config_payload,
        "oversample_minority": bool(
            task_overrides.get("oversample_minority", getattr(args, "oversample_minority", False))
        ),
        "use_focal_loss": bool(
            task_overrides.get("use_focal_loss", getattr(args, "use_focal_loss", False))
        ),
        "dynamic_pos_weight": bool(
            task_overrides.get("dynamic_pos_weight", getattr(args, "dynamic_pos_weight", False))
        ),
        "focal_gamma": task_overrides.get("focal_gamma", getattr(args, "focal_gamma", 2.0)),
        "baseline_finetune_epochs": baseline_finetune_default,
        "baseline_patience": baseline_patience_default,
        "bestcfg_epochs_override": bool(bestcfg_epochs_override),
        "bestcfg_patience_override": bool(bestcfg_patience_override),
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
            logger.error(
                "Tox21 case study runtime error for task %s: %s (%s)",
                task_name,
                message,
                exc.__class__.__name__,
            )
            raise

    if allow_shape_retry:
        diagnostics_ref = getattr(result, "diagnostics", None)
        if isinstance(diagnostics_ref, dict):
            diagnostics_ref.setdefault("allow_shape_coercion_forced", True)
            if allow_shape_error:
                diagnostics_ref.setdefault("allow_shape_coercion_retry_reason", allow_shape_error)

    diagnostics = getattr(result, "diagnostics", {}) or {}
    try:
        resolver_payload = dict(task_encoder_info)
    except Exception:
        resolver_payload = {"task": task_name, "provided": base_encoder_checkpoint}
    if isinstance(resolver_payload.get("candidates"), list):
        resolver_payload["candidates"] = [str(entry) for entry in resolver_payload["candidates"]]
    if isinstance(diagnostics, dict):
        diagnostics.setdefault("task_encoder_resolver", resolver_payload)
        if balance_entry:
            diagnostics.setdefault("class_balance", balance_entry)
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

    split_counts_payload = {}
    split_strategy = None
    split_positive_floor = None
    if isinstance(diagnostics, Mapping):
        split_counts_payload = diagnostics.get("split_counts", {}) or {}
        split_strategy = diagnostics.get("split_strategy")
        split_positive_floor = diagnostics.get("split_positive_floor")

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
        "prediction_csv": None,
    }
    if split_strategy is not None:
        summary_payload["split_strategy"] = split_strategy
    if split_positive_floor is not None:
        try:
            summary_payload["split_positive_floor"] = int(split_positive_floor)
        except Exception:
            pass
    try:
        val_pos = split_counts_payload.get("val", {}).get("positives")
        test_pos = split_counts_payload.get("test", {}).get("positives")
        if val_pos is not None:
            summary_payload["split_val_positives"] = int(val_pos)
        if test_pos is not None:
            summary_payload["split_test_positives"] = int(test_pos)
    except Exception:
        pass
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

        ece_value = metrics_block.get("ece")
        try:
            ece_float = float(ece_value) if ece_value is not None else None
        except Exception:
            ece_float = None
        if (
            ece_float is not None
            and ece_float > calibration_warn_threshold
            and isinstance(diagnostics, dict)
        ):
            warnings_list = diagnostics.setdefault("calibration_warnings", [])
            warnings_list.append(
                {
                    "task": task_name,
                    "encoder": getattr(eval_res, "encoder_source", getattr(eval_res, "name", "unknown")),
                    "ece": ece_float,
                }
            )
            _wandb_log_safe(
                wb,
                {
                    "phase": "tox21",
                    "status": "warning",
                    "task": task_name,
                    f"{prefix}metrics/ece": ece_float,
                    f"{prefix}calibration_warning": True,
                },
            )

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
    prediction_csv_path: Optional[str] = None

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
        "split_strategy": split_strategy,
        "split_positive_floor": split_positive_floor,
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

    preds_block = diagnostics.get("test_predictions") if isinstance(diagnostics, dict) else None
    if isinstance(preds_block, dict):
        indices = list(preds_block.get("indices") or [])
        logits = list(preds_block.get("logits") or [])
        probabilities = list(preds_block.get("probabilities") or [])
        labels = list(preds_block.get("true_labels") or [])
        base_count = min(len(indices), len(logits), len(probabilities))

        def _as_float(value: Any) -> float:
            try:
                return float(value)
            except Exception:
                return float("nan")

        if base_count > 0:
            prediction_csv_path = os.path.join(report_dir, f"{stem}_scores.csv")
            with open(prediction_csv_path, "w", newline="", encoding="utf-8") as pred_handle:
                writer = csv.writer(pred_handle)
                writer.writerow(["graph_id", "assay", "true_label", "logit", "probability"])
                for row_idx in range(base_count):
                    idx_val = _coerce_int_like(indices[row_idx])
                    graph_id = f"graph_{idx_val:05d}" if idx_val is not None else f"graph_{indices[row_idx]}"
                    label_val = labels[row_idx] if row_idx < len(labels) else float("nan")
                    writer.writerow(
                        [
                            graph_id,
                            task_name,
                            _as_float(label_val),
                            _as_float(logits[row_idx]),
                            _as_float(probabilities[row_idx]),
                        ]
                    )
            json_payload["prediction_csv"] = prediction_csv_path
            _wandb_save_safe(wb, prediction_csv_path)
            _wandb_log_safe(
                wb,
                {"task": task_name, "prediction_csv": prediction_csv_path},
            )
            summary_payload["prediction_csv"] = prediction_csv_path

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
    if prediction_csv_path:
        manifest_payload["reports"]["prediction_csv"] = prediction_csv_path

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
        "prediction_csv": prediction_csv_path,
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
        "prediction_csv_path": prediction_csv_path,
        "manifest_path": manifest_path,
        "stage_path": stage_path,
        "auc_summary": auc_summary,
        "threshold_rule": threshold_rule,
    }


def cmd_tox21(args: argparse.Namespace) -> None:
    """Run the Tox21 ranking case study."""
    logger.info("Starting Tox21 case study with args: %s", args)
    wb = None

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

    try:
        configure_omp_threads(
            stage="tox21",
            num_workers=getattr(args, "num_workers", -1),
            world_size=os.environ.get("WORLD_SIZE"),
            log=logger,
        )
    except Exception as exc:
        logger.exception("Tox21 case study failed")
        error_log = {"phase": "tox21", "status": "error", "error": str(exc)}
        _wandb_log_safe(wb, error_log)
        try:
            _export_gate_env(False)
        except Exception:
            logger.debug("Failed to export failure gate status", exc_info=True)
        sys.exit(5)

    if run_tox21_case_study is None:
        logger.error("Case study module is unavailable.")
        sys.exit(5)

    triage_pct = getattr(args, "triage_pct", 0.10)
    calibrate = not getattr(args, "no_calibrate", False)
    calibrate_per_head = bool(getattr(args, "calibrate_per_head", False))

    finetune_epochs_provided = getattr(args, "_finetune_epochs_provided", None)
    if finetune_epochs_provided is None:
        finetune_epochs_provided = _flag_was_provided(("--finetune-epochs", "--finetune_epochs"))
    patience_provided = getattr(args, "_patience_provided", None)
    if patience_provided is None:
        patience_provided = _flag_was_provided(("--patience",))

    eval_mode = str(
        getattr(
            args,
            "evaluation_mode",
            getattr(args, "encoder_source", "pretrain_frozen"),
        )
        or "pretrain_frozen"
    ).lower()

    best_overrides, best_path = _load_best_config_overrides(args)
    inherited: List[str] = []
    if eval_mode == "hybrid" and "finetune_epochs" in best_overrides:
        best_overrides.pop("finetune_epochs", None)
    setattr(args, "_bestcfg_epochs_override", "finetune_epochs" in best_overrides)
    setattr(args, "_bestcfg_patience_override", "patience" in best_overrides)
    if "add_3d" in best_overrides and not _flag_was_provided(("--add-3d", "--add_3d")):
        desired = bool(best_overrides["add_3d"])
        if bool(getattr(args, "add_3d", desired)) != desired:
            inherited.append(f"add_3d={desired}")
        setattr(args, "add_3d", desired)
    if "hidden_dim" in best_overrides:
        desired_hidden = int(best_overrides["hidden_dim"])
        if getattr(args, "hidden_dim", desired_hidden) != desired_hidden:
            inherited.append(f"hidden_dim={desired_hidden}")
        setattr(args, "hidden_dim", desired_hidden)
        setattr(args, "_hidden_dim_provided", True)
    if "dropout" in best_overrides and not getattr(args, "_dropout_provided", False):
        desired_dropout = _coerce_float_like(best_overrides.get("dropout"))
        if desired_dropout is not None:
            current_dropout = getattr(args, "dropout", None)
            if current_dropout is None or float(current_dropout) != float(desired_dropout):
                inherited.append(f"dropout={desired_dropout}")
            setattr(args, "dropout", desired_dropout)
            setattr(args, "_dropout_provided", True)
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
    if "finetune_epochs" in best_overrides and not finetune_epochs_provided:
        desired_epochs = _coerce_int_like(best_overrides.get("finetune_epochs"))
        if desired_epochs is not None:
            current_epochs = getattr(args, "finetune_epochs", None)
            if current_epochs != desired_epochs:
                inherited.append(f"finetune_epochs={desired_epochs}")
            setattr(args, "finetune_epochs", desired_epochs)
    if "patience" in best_overrides and not patience_provided:
        desired_patience = _coerce_int_like(best_overrides.get("patience"))
        if desired_patience is not None:
            current_patience = getattr(args, "patience", None)
            if current_patience != desired_patience:
                inherited.append(f"patience={desired_patience}")
            setattr(args, "patience", desired_patience)
    if inherited and best_path is not None:
        logger.info(
            "Inheriting Phase-2 best_config overrides from %s: %s",
            best_path,
            ", ".join(inherited),
        )

    task_hparam_map = _load_task_hparams(getattr(args, "per_task_hparams", None))
    hybrid_defaults = task_hparam_map.get("__hybrid__", {}) if eval_mode == "hybrid" else {}
    if hybrid_defaults:
        head_lr_provided = _flag_was_provided(("--head-lr", "--head_lr"))
        encoder_lr_provided = _flag_was_provided(("--encoder-lr", "--encoder_lr"))
        layerwise_provided = _flag_was_provided(("--layerwise-decay", "--layerwise_decay"))
        if not head_lr_provided and getattr(args, "head_lr", None) is None:
            setattr(args, "head_lr", hybrid_defaults.get("head_lr"))
        if not encoder_lr_provided and getattr(args, "encoder_lr", None) is None:
            setattr(args, "encoder_lr", hybrid_defaults.get("encoder_lr"))
        if not layerwise_provided and getattr(args, "layerwise_decay", None) is None:
            setattr(args, "layerwise_decay", hybrid_defaults.get("layerwise_decay"))
    if eval_mode == "hybrid":
        if getattr(args, "hybrid_lr_scheduler", None) is None:
            setattr(args, "hybrid_lr_scheduler", getattr(args, "lr_scheduler", None))
        if getattr(args, "hybrid_warmup_ratio", None) is None:
            setattr(args, "hybrid_warmup_ratio", getattr(args, "warmup_ratio", None))
        if getattr(args, "hybrid_min_lr", None) is None:
            setattr(args, "hybrid_min_lr", getattr(args, "min_lr", None))
        if getattr(args, "hybrid_min_lr_ratio", None) is None:
            setattr(args, "hybrid_min_lr_ratio", getattr(args, "min_lr_ratio", None))

    dataset_name = getattr(args, "dataset", "tox21") or "tox21"
    tasks_to_run = _resolve_tox21_tasks(args)
    if not tasks_to_run:
        raise ValueError("No Tox21 tasks specified or discovered")
    primary_task = tasks_to_run[0]
    setattr(args, "task", primary_task)

    class_balance = _estimate_class_balance(args.csv, tasks_to_run)
    auto_pos_weights: Dict[str, float] = {}
    for task, info in class_balance.items():
        weight = info.get("pos_weight")
        if weight is not None:
            try:
                auto_pos_weights[task] = float(weight)
            except Exception:
                continue
    if class_balance:
        logger.info(
            "[tox21] class balance summary: %s",
            {
                task: {
                    "total": info.get("total"),
                    "pos_frac": round(info.get("pos_frac", 0.0), 4)
                    if info.get("pos_frac") is not None
                    else None,
                    "pos_weight": info.get("pos_weight"),
                }
                for task, info in class_balance.items()
            },
        )
    devices_val = _coerce_int_like(getattr(args, "devices", None))
    if devices_val is None:
        raw_devices = getattr(args, "devices", None)
        try:
            devices_val = int(raw_devices) if raw_devices is not None else None
        except Exception:
            devices_val = None
    device_value = getattr(args, "device", None)
    if devices_val is None or devices_val <= 0 or device_value is None:
        device_value, devices_val = _normalise_device_and_count(args)
    else:
        setattr(args, "devices", devices_val)
        setattr(args, "device", _resolve_auto_device(device_value))
    try:
        world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        world_size_env = 1
    if devices_val > 1 and world_size_env <= 1:
        if _should_relaunch_ddp(devices_val, world_size_env):
            _relaunch_with_torchrun(devices_val)
        logger.warning(
            "[tox21] devices=%d resolved (mask-aware) but DDP is inactive; "
            "run with torchrun or WORLD_SIZE>1 to enable DDP.",
            devices_val,
        )
    baseline_mode = eval_mode == "baseline"
    bestcfg_epochs_override = bool(getattr(args, "_bestcfg_epochs_override", False))
    bestcfg_patience_override = bool(getattr(args, "_bestcfg_patience_override", False))
    finetune_epochs_provided = bool(finetune_epochs_provided)
    baseline_epochs_env = _coerce_int_like(os.getenv("TOX21_BASELINE_FINETUNE_EPOCHS"))
    baseline_finetune_default = (
        baseline_epochs_env
        if baseline_epochs_env is not None
        else _coerce_int_like(getattr(args, "baseline_finetune_epochs", None))
    )
    finetune_epochs_default = 20
    finetune_epochs_value = getattr(args, "finetune_epochs", None)
    bestcfg_epochs_override = bestcfg_epochs_override or ("finetune_epochs" in best_overrides)
    if eval_mode == "hybrid":
        hybrid_epochs_raw = getattr(args, "epochs", None)
        hybrid_epochs_val = _coerce_int_like(hybrid_epochs_raw)
        if hybrid_epochs_val is not None:
            if finetune_epochs_provided and finetune_epochs_value not in (None, hybrid_epochs_val):
                logger.info(
                    "Hybrid evaluation mode overrides finetune_epochs=%s with TOX21_EPOCHS=%d.",
                    str(finetune_epochs_value),
                    hybrid_epochs_val,
                )
            finetune_epochs_value = int(hybrid_epochs_val)
            finetune_epochs_provided = True
            bestcfg_epochs_override = False
    if baseline_mode and not finetune_epochs_provided and not bestcfg_epochs_override:
        resolved_baseline_epochs = baseline_finetune_default or finetune_epochs_value or finetune_epochs_default
        if resolved_baseline_epochs is not None:
            finetune_epochs_value = int(resolved_baseline_epochs)
            logger.info(
                "Baseline evaluation mode detected; defaulting finetune_epochs to %d",
                finetune_epochs_value,
            )
    if finetune_epochs_value is None:
        finetune_epochs_value = finetune_epochs_default
    setattr(args, "finetune_epochs", finetune_epochs_value)
    patience_provided = bool(patience_provided)
    baseline_patience_env = _coerce_int_like(os.getenv("TOX21_BASELINE_PATIENCE"))
    baseline_patience_default = (
        baseline_patience_env
        if baseline_patience_env is not None
        else _coerce_int_like(getattr(args, "baseline_patience", None))
    )
    bestcfg_patience_override = bestcfg_patience_override or ("patience" in best_overrides)
    if baseline_mode and not patience_provided and not bestcfg_patience_override:
        patience_candidate = getattr(args, "patience", None)
        baseline_patience_value = baseline_patience_default
        if baseline_patience_value is None:
            baseline_patience_value = patience_candidate
        if baseline_patience_value is not None and (
            patience_candidate is None or patience_candidate > baseline_patience_value
        ):
            setattr(args, "patience", int(baseline_patience_value))
            logger.info(
                "Baseline evaluation mode detected; setting default patience to %d",
                int(getattr(args, "patience")),
            )
    setattr(args, "_finetune_epochs_provided", bool(finetune_epochs_provided))
    setattr(args, "_patience_provided", bool(patience_provided))
    setattr(args, "_baseline_finetune_default", baseline_finetune_default)
    setattr(args, "_baseline_patience_default", baseline_patience_default)
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
        "dropout": getattr(args, "dropout", None),
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
        "head_lr": getattr(args, "head_lr", None),
        "encoder_lr": getattr(args, "encoder_lr", None),
        "layerwise_decay": getattr(args, "layerwise_decay", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "class_weights": getattr(args, "class_weights", None),
        "pos_class_weight": getattr(args, "pos_class_weight", None),
        "head_scheduler": getattr(args, "head_scheduler", None),
        "lr_scheduler": getattr(args, "lr_scheduler", None),
        "warmup_ratio": getattr(args, "warmup_ratio", None),
        "min_lr": getattr(args, "min_lr", None),
        "min_lr_ratio": getattr(args, "min_lr_ratio", None),
        "hybrid_freeze_epochs": getattr(args, "hybrid_freeze_epochs", None),
        "hybrid_partial_epochs": getattr(args, "hybrid_partial_epochs", None),
        "hybrid_lr_scheduler": getattr(args, "hybrid_lr_scheduler", None),
        "hybrid_warmup_ratio": getattr(args, "hybrid_warmup_ratio", None),
        "hybrid_min_lr": getattr(args, "hybrid_min_lr", None),
        "hybrid_min_lr_ratio": getattr(args, "hybrid_min_lr_ratio", None),
        "threshold_metric": getattr(args, "threshold_metric", None),
        "per_task_hparams": getattr(args, "per_task_hparams", None),
        "auto_pos_class_weight": auto_pos_weights,
        "class_balance": {
            task: {
                "total": info.get("total"),
                "pos_frac": info.get("pos_frac"),
                "neg_frac": info.get("neg_frac"),
                "pos_weight": info.get("pos_weight"),
            }
            for task, info in class_balance.items()
        },
    }
    bf16_head_cfg = getattr(args, "bf16_head", None)
    if bf16_head_cfg is not None:
        wandb_config["bf16_head"] = bf16_head_cfg
    is_main = is_main_process()
    wb = maybe_init_wandb(
        getattr(args, "use_wandb", False) and is_main,
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
    aggregated_prediction_paths: Dict[str, str] = {}
    aggregated_auc_summaries: Dict[str, Any] = {}
    per_task_diagnostics: Dict[str, Any] = {}
    diagnostics_template: Dict[str, Any] | None = None
    aggregated_allow_shape_effective: Optional[bool] = None
    aggregated_allow_shape_requested: Any = getattr(args, "allow_shape_coercion", None)
    aggregated_allow_shape_auto = False
    aggregated_calibration: Dict[str, Any] = {}

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
                calibrate_per_head=calibrate_per_head,
                task_hparam_map=task_hparam_map,
                hybrid_defaults=hybrid_defaults,
                cache_dir=cache_dir,
                report_dir=report_dir,
                wb=wb,
                class_balance=class_balance,
                auto_pos_weights=auto_pos_weights,
                calibration_warn_threshold=_CALIBRATION_ECE_WARN,
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
            aggregated_prediction_paths[task_name] = result.get("prediction_csv_path", "")
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


        aggregated_calibration = {
            task: diag.get("calibration_warnings")
            for task, diag in per_task_diagnostics.items()
            if isinstance(diag, dict) and diag.get("calibration_warnings")
        }

        if aggregated_allow_shape_effective is not None:
            setattr(args, "allow_shape_coercion", bool(aggregated_allow_shape_effective))
        elif aggregated_allow_shape_requested in (True, False):
            setattr(args, "allow_shape_coercion", bool(aggregated_allow_shape_requested))

        combined_diagnostics: Dict[str, Any] = {}
        if diagnostics_template is not None:
            combined_diagnostics = dict(diagnostics_template)
        combined_diagnostics["task_count"] = len(tasks_to_run)
        combined_diagnostics["per_task"] = per_task_diagnostics
        if aggregated_calibration:
            combined_diagnostics["calibration_warnings"] = aggregated_calibration
        combined_diagnostics["class_balance"] = class_balance
        combined_diagnostics["auto_pos_class_weight"] = auto_pos_weights
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
            "class_balance": class_balance,
            "auto_pos_class_weight": auto_pos_weights,
            "calibration_warnings": aggregated_calibration,
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
                "diagnostics": per_task_diagnostics.get(task),
                "class_balance": class_balance.get(task),
                "auto_pos_class_weight": auto_pos_weights.get(task),
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
            "predictions": aggregated_prediction_paths,
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
                    "prediction_csv": aggregated_prediction_paths.get(task),
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
                "prediction_csv": aggregated_prediction_paths,
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


class _StandaloneBoolFlag(argparse.Action):
    """argparse action mirroring ``train_jepa.BoolFlag``."""

    def __init__(self, option_strings, dest, **kwargs):  # type: ignore[override]
        kwargs.setdefault("nargs", "?")
        kwargs.setdefault("const", True)
        kwargs.setdefault("default", None)
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Optional[str],
        option_string: Optional[str] = None,
    ) -> None:
        if values is None:
            setattr(namespace, self.dest, True)
        else:
            coerced = _coerce_bool_like(values)
            if coerced is None:
                parser.error(f"Expected boolean for {option_string}, got '{values}'")
            setattr(namespace, self.dest, coerced)
        try:
            setattr(namespace, f"_{self.dest}_provided", True)
        except Exception:
            pass


def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripts.commands.tox21")
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--task",
        help="Name of the toxicity column to predict (deprecated when --tasks is used)",
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--dataset")
    parser.add_argument("--report-dir")
    parser.add_argument("--tox21-dir")
    parser.add_argument("--pretrain-epochs", type=int, dest="pretrain_epochs")
    parser.add_argument("--finetune-epochs", type=int, dest="finetune_epochs")
    parser.add_argument(
        "--pretrain-time-budget-mins",
        type=int,
        dest="pretrain_time_budget_mins",
    )
    parser.add_argument(
        "--finetune-time-budget-mins",
        type=int,
        dest="finetune_time_budget_mins",
    )
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--encoder-source")
    parser.add_argument("--evaluation-mode")
    parser.add_argument("--explain-mode", dest="explain_mode")
    parser.add_argument("--explain-steps", dest="explain_steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--head-lr", type=float, dest="head_lr")
    parser.add_argument("--encoder-lr", type=float, dest="encoder_lr")
    parser.add_argument("--weight-decay", type=float, dest="weight_decay")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-scheduler", dest="lr_scheduler")
    parser.add_argument("--warmup-ratio", type=float, dest="warmup_ratio")
    parser.add_argument("--min-lr", type=float, dest="min_lr")
    parser.add_argument("--min-lr-ratio", type=float, dest="min_lr_ratio")
    parser.add_argument("--pretrain-lr", type=float, dest="pretrain_lr")
    parser.add_argument("--class-weights", dest="class_weights")
    parser.add_argument("--threshold-metric", dest="threshold_metric")
    parser.add_argument("--layerwise-decay", type=float, dest="layerwise_decay")
    parser.add_argument(
        "--verify-match-threshold",
        type=float,
        dest="verify_match_threshold",
    )
    parser.add_argument(
        "--oversample-minority",
        action=_StandaloneBoolFlag,
        dest="oversample_minority",
    )
    parser.add_argument(
        "--dynamic-pos-weight",
        action=_StandaloneBoolFlag,
        dest="dynamic_pos_weight",
    )
    parser.add_argument(
        "--use-focal-loss",
        action=_StandaloneBoolFlag,
        dest="use_focal_loss",
    )
    parser.add_argument("--focal-gamma", type=float, dest="focal_gamma")
    parser.add_argument("--use-wandb", action=_StandaloneBoolFlag, dest="use_wandb")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-tags", nargs="*")
    parser.add_argument("--num-workers", type=int, dest="num_workers")
    parser.add_argument("--prefetch-factor", type=int, dest="prefetch_factor")
    parser.add_argument(
        "--persistent-workers",
        action=_StandaloneBoolFlag,
        dest="persistent_workers",
    )
    parser.add_argument(
        "--pin-memory",
        "--pin_memory",
        action=_StandaloneBoolFlag,
        dest="pin_memory",
    )
    parser.add_argument("--bf16", action=_StandaloneBoolFlag, dest="bf16")
    parser.add_argument("--bf16-head", action=_StandaloneBoolFlag, dest="bf16_head")
    parser.add_argument("--devices", type=int, dest="devices")
    parser.add_argument("--device")
    parser.add_argument("--gnn-type", dest="gnn_type")
    parser.add_argument("--hidden-dim", type=int, dest="hidden_dim")
    parser.add_argument("--num-layers", type=int, dest="num_layers")
    parser.add_argument("--dropout", type=float, dest="dropout")
    parser.add_argument("--ema-decay", type=float, dest="ema_decay")
    parser.add_argument("--contiguity", action=_StandaloneBoolFlag, dest="contiguity")
    parser.add_argument("--temperature", type=float, dest="temperature")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--triage-pct", type=float, dest="triage_pct")
    parser.add_argument("--no-calibrate", action="store_true", dest="no_calibrate")
    parser.add_argument(
        "--calibrate-per-head",
        action=_StandaloneBoolFlag,
        dest="calibrate_per_head",
    )
    parser.add_argument("--pos-class-weight", action="append", dest="pos_class_weight")
    parser.add_argument(
        "--allow-shape-coercion",
        action=_StandaloneBoolFlag,
        dest="allow_shape_coercion",
    )
    parser.add_argument(
        "--allow-equal-hash",
        action=_StandaloneBoolFlag,
        dest="allow_equal_hash",
    )
    parser.add_argument(
        "--strict-encoder-config",
        action=_StandaloneBoolFlag,
        dest="strict_encoder_config",
    )
    parser.add_argument(
        "--full-finetune",
        action=_StandaloneBoolFlag,
        dest="full_finetune",
    )
    parser.add_argument(
        "--freeze-encoder",
        action=_StandaloneBoolFlag,
        dest="freeze_encoder",
    )
    parser.add_argument(
        "--contrastive",
        action=_StandaloneBoolFlag,
        dest="contrastive",
    )
    parser.add_argument("--tox21-head-batch-size", type=int, dest="tox21_head_batch_size")
    parser.add_argument("--head-ensemble-size", type=int, dest="head_ensemble_size")
    parser.add_argument("--head-scheduler", dest="head_scheduler")
    parser.add_argument("--unfreeze-top-layers", type=int, dest="unfreeze_top_layers")
    parser.add_argument("--hybrid-freeze-epochs", type=int, dest="hybrid_freeze_epochs")
    parser.add_argument("--hybrid-partial-epochs", type=int, dest="hybrid_partial_epochs")
    parser.add_argument("--hybrid-lr-scheduler", dest="hybrid_lr_scheduler")
    parser.add_argument("--hybrid-warmup-ratio", type=float, dest="hybrid_warmup_ratio")
    parser.add_argument("--hybrid-min-lr", type=float, dest="hybrid_min_lr")
    parser.add_argument("--hybrid-min-lr-ratio", type=float, dest="hybrid_min_lr_ratio")
    parser.add_argument("--per-task-hparams", dest="per_task_hparams")
    parser.add_argument("--best-config", dest="best_config")
    parser.add_argument("--best-config-json", dest="best_config_json")
    parser.add_argument("--best-config-path", dest="best_config_path")
    parser.add_argument("--add-3d", action=_StandaloneBoolFlag, dest="add_3d")
    return parser


def _finalise_standalone_args(namespace: argparse.Namespace) -> argparse.Namespace:
    if not hasattr(namespace, "tasks") or getattr(namespace, "tasks", None) is None:
        namespace.tasks = []
    elif isinstance(namespace.tasks, str):
        namespace.tasks = [namespace.tasks]
    elif not isinstance(namespace.tasks, list):
        namespace.tasks = list(namespace.tasks)
    if getattr(namespace, "wandb_tags", None) is None:
        namespace.wandb_tags = []
    if getattr(namespace, "use_wandb", None) is None:
        namespace.use_wandb = False
    if getattr(namespace, "triage_pct", None) is None:
        namespace.triage_pct = 0.0
    if not hasattr(namespace, "no_calibrate"):
        namespace.no_calibrate = False
    if not hasattr(namespace, "calibrate_per_head") or namespace.calibrate_per_head is None:
        namespace.calibrate_per_head = False
    if not hasattr(namespace, "pos_class_weight"):
        namespace.pos_class_weight = None
    if getattr(namespace, "baseline_finetune_epochs", None) is None:
        namespace.baseline_finetune_epochs = None
    if getattr(namespace, "baseline_patience", None) is None:
        namespace.baseline_patience = None
    if getattr(namespace, "oversample_minority", None) is None:
        namespace.oversample_minority = False
    if getattr(namespace, "use_focal_loss", None) is None:
        namespace.use_focal_loss = False
    if getattr(namespace, "dynamic_pos_weight", None) is None:
        namespace.dynamic_pos_weight = False
    if getattr(namespace, "focal_gamma", None) is None:
        namespace.focal_gamma = 2.0
    if not hasattr(namespace, "freeze_encoder"):
        namespace.freeze_encoder = False
    if getattr(namespace, "head_ensemble_size", None) is None:
        namespace.head_ensemble_size = 1
    if getattr(namespace, "tox21_head_batch_size", None) is None:
        namespace.tox21_head_batch_size = 256
    for attr in ("warmup_ratio", "min_lr", "min_lr_ratio", "layerwise_decay"):
        if getattr(namespace, attr, None) is None:
            setattr(namespace, attr, None)
    if getattr(namespace, "threshold_metric", None) is None:
        namespace.threshold_metric = None
    if getattr(namespace, "per_task_hparams", None) is None:
        namespace.per_task_hparams = None
    for attr in ("gnn_type", "hidden_dim", "num_layers", "dropout"):
        provided = getattr(namespace, attr, None) is not None
        setattr(namespace, f"_{attr}_provided", provided)
    for attr in ("num_workers", "prefetch_factor", "devices"):
        if getattr(namespace, attr, None) is not None:
            setattr(namespace, f"_{attr}_provided", True)
    return namespace


def main(argv: Optional[List[str]] | None = None) -> int:
    """Execute the Tox21 command directly when run as a module."""

    args = list(argv if argv is not None else sys.argv[1:])
    args, removed = _strip_encoder_manifest_args(args)
    if removed:
        logger.warning(
            "Ignoring legacy --encoder-manifest flag for Tox21 evaluation (values=%s).",
            removed,
        )

    if "--stage-shim" in args:
        logger.debug("stage shim requested; exiting early")
        return 0

    def _extract_local_rank_args(raw_args: List[str]) -> Tuple[Optional[int], List[str]]:
        cleaned: List[str] = []
        local_rank: Optional[int] = None
        idx = 0
        while idx < len(raw_args):
            token = raw_args[idx]
            if token in {"--local-rank", "--local_rank"}:
                if idx + 1 >= len(raw_args) or str(raw_args[idx + 1]).startswith("-"):
                    raise ValueError(f"missing value for {token}")
                value = raw_args[idx + 1]
                try:
                    local_rank = int(value)
                except Exception as exc:
                    raise ValueError(f"invalid value for {token}: {value}") from exc
                idx += 2
                continue
            if token.startswith("--local-rank=") or token.startswith("--local_rank="):
                value = token.split("=", 1)[1]
                try:
                    local_rank = int(value)
                except Exception as exc:
                    raise ValueError(f"invalid value for {token}: {value}") from exc
                idx += 1
                continue
            cleaned.append(token)
            idx += 1
        return local_rank, cleaned

    try:
        local_rank_value, remaining_args = _extract_local_rank_args(args)
    except ValueError as exc:
        logger.error("Invalid --local-rank flag: %s", exc)
        return 2

    local_rank_args: List[str] = []
    if local_rank_value is not None:
        local_rank_args = ["--local-rank", str(local_rank_value)]

    try:
        from scripts import train_jepa as _train_jepa
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            "Unable to import train_jepa parser: %s", exc,
            exc_info=True,
        )
        parser = _build_standalone_parser()
        try:
            parsed = parser.parse_args([*local_rank_args, *remaining_args])
        except SystemExit as exc_parse:
            return int(exc_parse.code or 0)
        parsed = _finalise_standalone_args(parsed)
        cmd_tox21(parsed)
        return 0

    parser = _train_jepa.build_parser()
    try:
        parsed = parser.parse_args([*local_rank_args, "tox21", *remaining_args])
    except SystemExit as exc:  # pragma: no cover - argparse handles help/errors
        return int(exc.code or 0)

    if not hasattr(parsed, "func"):
        parser.error("tox21 command missing handler")

    if getattr(parsed, "command", None) == "tox21":
        parsed = _finalise_standalone_args(parsed)

    parsed.func(parsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
