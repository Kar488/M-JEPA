#!/usr/bin/env python3
"""Recheck the top-k runs from a Phase-2 sweep and materialise the winner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import numbers
import os
import pathlib
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from utils.wandb_filters import silence_pydantic_field_warnings

silence_pydantic_field_warnings()

import numpy as np
import wandb
from urllib.parse import urlparse

try:
    from reports.wandb_utils import resolve_wandb_http_timeout
except ImportError:  # pragma: no cover - optional dependency
    def resolve_wandb_http_timeout(default: Union[int, float]) -> Union[int, float]:
        return default

# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def _env_positive_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"[recheck][warn] invalid int for {name}={raw!r}; ignoring", flush=True)
        return None
    if value <= 0:
        print(f"[recheck][warn] non-positive {name}={raw!r}; ignoring", flush=True)
        return None
    return value


def _coerce_positive_int(raw: Optional[str], label: str) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        print(f"[recheck][warn] ignoring non-integer {label}={raw!r}", flush=True)
        return None
    if value <= 0:
        print(f"[recheck][warn] ignoring non-positive {label}={raw!r}", flush=True)
        return None
    return value


WINNER_FALLBACK: Optional[Dict[str, Any]] = None
WINNER_SOURCE: Optional[str] = None


def _maybe_deserialise(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def _load_winner_config() -> Dict[str, Any]:
    global WINNER_FALLBACK, WINNER_SOURCE
    if WINNER_FALLBACK is not None:
        return dict(WINNER_FALLBACK)

    grid_dir = os.environ.get("GRID_DIR") or os.environ.get("GRID_SOURCE_DIR") or os.environ.get("GRID_CACHE_DIR")
    candidates: List[pathlib.Path] = []
    if grid_dir:
        grid_root = pathlib.Path(grid_dir)
        candidates.extend(
            [
                grid_root / "phase2_winner_config.csv",
                grid_root / "phase2_export" / "phase2_winner_config.csv",
                grid_root / "phase2_export" / "best_grid_config.json",
                grid_root / "best_grid_config.json",
            ]
        )
    candidates.append(pathlib.Path("/data/mjepa/cache/grid/phase2_winner_config.csv"))

    def _parse_csv(path: pathlib.Path) -> Dict[str, Any]:
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            return {}
        if not rows:
            return {}
        row = rows[0]
        cfg: Dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                continue
            cleaned = key.split(".", 1)[1] if key.startswith("config.") else key
            text = str(value).strip()
            if not text:
                continue
            cfg[cleaned] = _maybe_deserialise(text)
        return cfg

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".csv":
            cfg = _parse_csv(candidate)
        else:
            try:
                cfg = json.load(candidate.open())
            except Exception:
                cfg = {}
        if cfg:
            WINNER_FALLBACK = cfg
            WINNER_SOURCE = str(candidate)
            break

    if WINNER_FALLBACK is None:
        WINNER_FALLBACK = {}
    return dict(WINNER_FALLBACK)


def _discover_visible_gpu_ids() -> List[str]:
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if visible and visible != "-1":
        return [token for token in visible.split(",") if token]

    try:
        import torch  # type: ignore

        count = int(torch.cuda.device_count())  # type: ignore[attr-defined]
    except Exception:
        count = 0

    if count > 0:
        return [str(i) for i in range(count)]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        ids = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        if ids:
            return ids
    except Exception:
        pass

    return []


def _split_gpu_ids(ids: Sequence[str], agent_count: int) -> List[str]:
    if agent_count <= 0:
        return []
    total = len(ids)
    if total == 0:
        return [""] * agent_count
    base = total // agent_count
    remainder = total % agent_count
    result: List[str] = []
    index = 0
    for _ in range(agent_count):
        take = base
        if remainder > 0:
            take += 1
            remainder -= 1
        if take <= 0:
            result.append("")
            continue
        chunk = ids[index : index + take]
        result.append(",".join(chunk))
        index += take
    return result


def _extract_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, bool):
        value = int(value)

    if isinstance(value, numbers.Integral):
        result = int(value)
    elif isinstance(value, numbers.Real) and math.isfinite(float(value)):
        result = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            result = int(text, 10)
        except ValueError:
            try:
                parsed = float(text)
            except ValueError:
                return None
            if not math.isfinite(parsed) or not parsed.is_integer():
                return None
            result = int(parsed)
    else:
        return None

    if result <= 0:
        return None
    return result


def _resolve_agent_count(visible_gpu_ids: Sequence[str], override_devices: Optional[int] = None) -> int:
    gpu_ids = list(visible_gpu_ids)

    explicit = _env_positive_int("PHASE2_RECHECK_AGENT_COUNT")
    if explicit is None:
        explicit = _env_positive_int("PHASE2_AGENT_COUNT")

    if explicit is not None:
        count = explicit
    elif gpu_ids:
        count = len(gpu_ids)
    else:
        count = 1

    if gpu_ids:
        max_parallel = len(gpu_ids)
        if override_devices and override_devices > 1:
            max_parallel = len(gpu_ids) // override_devices
            if max_parallel <= 0:
                max_parallel = 1
        count = min(count, max_parallel)

    return max(1, count)


def _compute_worker_gpu_masks(
    visible_gpu_ids: Sequence[str],
    desired_workers: int,
    devices_per_run: Optional[int],
) -> Tuple[int, List[str]]:
    gpu_ids = [gpu for gpu in visible_gpu_ids if str(gpu).strip()]
    total = len(gpu_ids)
    worker_count = max(1, desired_workers)

    per_run = devices_per_run or 1
    if per_run <= 0:
        per_run = 1

    if not gpu_ids:
        return worker_count, []

    if per_run > total:
        return 1, [",".join(gpu_ids)] if gpu_ids else []

    max_parallel = total // per_run if per_run > 0 else total
    if max_parallel <= 0:
        max_parallel = 1

    if worker_count > max_parallel:
        worker_count = max_parallel

    if worker_count <= 1:
        return 1, [",".join(gpu_ids)] if gpu_ids else []

    if per_run <= 1:
        masks = [mask for mask in _split_gpu_ids(gpu_ids, worker_count) if mask]
        return worker_count, masks

    masks: List[str] = []
    index = 0
    for _ in range(worker_count):
        chunk = gpu_ids[index : index + per_run]
        if len(chunk) < per_run:
            break
        masks.append(",".join(chunk))
        index += per_run

    if not masks:
        return 1, [",".join(gpu_ids)] if gpu_ids else []

    return worker_count, masks


def _apply_config_overrides(cfg: Dict[str, Any], *, override_devices: Optional[int]) -> None:
    if override_devices is not None:
        cfg["devices"] = override_devices
        cfg.pop("device", None)

def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """Like ``getattr`` but tolerates wrappers that raise ``KeyError``."""

    try:
        return getattr(obj, name)
    except (AttributeError, KeyError):
        return default
    except Exception:  # pragma: no cover - defensive
        return default


def _coerce_config(config: Any) -> Dict[str, Any]:
    """Convert a W&B config/summary object into a standard dictionary."""

    if isinstance(config, Mapping):
        return dict(config)

    to_dict = _safe_getattr(config, "to_dict")
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:  # pragma: no cover - defensive
            converted = None
        if isinstance(converted, dict):
            return converted

    if isinstance(config, str):
        try:
            parsed = json.loads(config)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    json_dict = _safe_getattr(config, "_json_dict")
    if isinstance(json_dict, dict):
        return dict(json_dict)

    items = _safe_getattr(config, "items")
    if callable(items):
        try:
            return dict(items())
        except Exception:  # pragma: no cover - defensive
            pass

    try:
        return dict(config)
    except Exception:  # pragma: no cover - defensive
        return {}


def _unwrap_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "value" in value and len(value) <= 2:
            return _unwrap_config_value(value["value"])
        for key in ("value", "values", "_value"):
            if key in value:
                return _unwrap_config_value(value[key])
        if value.get("_type") == "quantized_params" and isinstance(value.get("params"), Mapping):
            return {k: _unwrap_config_value(v) for k, v in value["params"].items()}
        if len(value) == 1:
            return _unwrap_config_value(next(iter(value.values())))
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_config_value(value[0])
    return value


def _lookup_nested(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]

    for sep in ("/", "."):
        if sep not in key:
            continue
        node: Any = mapping
        for part in key.split(sep):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                node = None
                break
        if node is not None:
            return node
    return None


def _coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None

    if isinstance(value, Mapping):
        for key in ("value", "latest", "last", "mean", "median", "max", "min", "best"):
            if key in value:
                numeric = _coerce_numeric(value[key])
                if numeric is not None:
                    return numeric
        for candidate in value.values():
            numeric = _coerce_numeric(candidate)
            if numeric is not None:
                return numeric
        return None

    if isinstance(value, (list, tuple)):
        for candidate in value:
            numeric = _coerce_numeric(candidate)
            if numeric is not None:
                return numeric
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def _norm_key(k: Any) -> str:
    """Canonicalise config keys so hyphen/underscore forms match."""
    return str(k).strip().lower().replace("-", "_")

def _num_close(a: Any, b: Any, rtol: float = 1e-6, atol: float = 1e-8) -> bool:
    """Numeric-safe equality with tolerance; falls back to exact match for non-numerics."""
    try:
        fa = float(a); fb = float(b)
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return False
        return abs(fa - fb) <= max(atol, rtol * max(abs(fa), abs(fb)))
    except Exception:
        return a == b

def _norm_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten W&B wrappers, normalise keys, leave values comparable."""
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        out[_norm_key(k)] = _unwrap_config_value(v)
    return out

def _metric_candidates(metric: str) -> List[str]:
    base = str(metric).strip()
    candidates: List[str] = []
    seen: set[str] = set()

    def _add(key: str) -> None:
        k = key.strip()
        if not k or k in seen:
            return
        seen.add(k)
        candidates.append(k)

    def _add_variants(name: str) -> None:
        _add(name)
        for repl in ("/", "."):
            _add(name.replace("_", repl))
        if "/" in name:
            _add(name.replace("/", "."))
        if "." in name:
            _add(name.replace(".", "/"))

    _add_variants(base)

    lower = base.lower()
    # If the metric already uses a dotted or slashed namespace, avoid expanding
    # it into suffix/prefix variants that can over-match unrelated keys.
    if any(sep in base for sep in ("/", ".")):
        if lower != base:
            _add_variants(lower)
        return candidates

    prefixes = ("val_", "val.", "val/", "validation_", "validation.", "validation/")
    core = lower
    for prefix in prefixes:
        if core.startswith(prefix):
            core = core[len(prefix) :]
            break
    if core and core != lower:
        _add_variants(core)

    common_suffixes = ("_mean", ".mean", "/mean", ".value", "/value")
    for name in list(candidates):
        for suffix in common_suffixes:
            _add(name + suffix)

    # Explicitly include historical metric spellings for RMSE-style probes.
    if "rmse" in core or "rmse" in lower:
        for alias in (
            "rmse",
            "rmse_mean",
            "rmse/value",
            "rmse.mean",
            "metrics/rmse",
            "metrics.rmse",
            "probe_rmse_mean",
        ):
            _add_variants(alias)

    return candidates


def _history_latest(run: Any, candidates: Sequence[str], limit: int = 512) -> Tuple[Optional[float], Optional[str]]:
    history = _safe_getattr(run, "history")
    if not callable(history):
        return None, None

    attempts: Iterable[Dict[str, Any]] = (
        {"keys": list(candidates), "pandas": False},
        {"keys": list(candidates)},
        {},
    )
    rows: List[Dict[str, Any]] = []
    for kwargs in attempts:
        try:
            iterator = history(**kwargs)
        except TypeError:
            continue
        except Exception:
            return None, None
        if iterator is None:
            continue
        try:
            for row in islice(iterator, limit):
                rows.append(_coerce_config(row))
        except TypeError:  # pandas.DataFrame slicing path
            try:
                rows = [_coerce_config(r) for r in iterator[:limit]]  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive
                rows = []
        break

    for row in reversed(rows):
        for candidate in candidates:
            numeric = _coerce_numeric(_lookup_nested(row, candidate))
            if numeric is not None:
                return numeric, candidate
    return None, None


def metric_of(run: Any, name: str, default: Optional[float] = None) -> Optional[float]:
    """
    Robustly extract `name` from a W&B run:
      1) run.summary and common alternates
      2) run._attrs.{summaryMetrics, summary_metrics}
      3) history fallback: latest non-NaN/finite among candidate keys
    Returns a float or `default`.
    """
    candidates = _metric_candidates(name)

    def _as_mapping(obj: Any) -> Optional[Mapping[str, Any]]:
        if obj is None:
            return None
        if isinstance(obj, Mapping):
            return obj
        # W&B Summary is a custom object; try to get a dict view
        to_dict = _safe_getattr(obj, "to_dict")
        if callable(to_dict):
            try:
                d = to_dict()
                if isinstance(d, Mapping):
                    return d
            except Exception:
                pass
        # Some versions expose private dicts
        for attr in ("_json_dict", "_root"):
            inner = _safe_getattr(obj, attr)
            if isinstance(inner, Mapping):
                return inner
        # Fall back to repo's coercer if available
        try:
            return _coerce_config(obj)  # type: ignore
        except Exception:
            return None

    # 1) Collect all plausible summary sources
    summary_sources: List[Mapping[str, Any]] = []
    for attr in ("summary", "summary_metrics", "summaryMetrics"):
        source = _safe_getattr(run, attr)
        payload = _as_mapping(source)
        if payload:
            summary_sources.append(payload)

    attrs = _safe_getattr(run, "_attrs")
    if isinstance(attrs, Mapping):
        for key in ("summaryMetrics", "summary_metrics"):
            payload = _as_mapping(attrs.get(key))
            if payload:
                summary_sources.append(payload)

    summary_keys: List[str] = []
    # Try summaries first (fast path)
    for summary in summary_sources:
        try:
            summary_keys.extend(list(summary.keys()))
        except Exception:
            pass
        for candidate in candidates:
            for alias in (candidate, f"{candidate}_mean", f"{candidate}.mean", f"{candidate}/mean", f"{candidate}.value", f"{candidate}/value"):
                numeric = _coerce_numeric(_lookup_nested(summary, alias))
                if numeric is not None:
                    return numeric

    # 2) History fallback: latest non-NaN/finite across candidates
    history_value, _ = _history_latest(run, candidates)
    if history_value is not None:
        return history_value

    run_id = _safe_getattr(run, "id") or _safe_getattr(run, "name") or "(unknown)"
    try:
        summary_preview = dict(list(summary_sources[0].items())[:10]) if summary_sources else {}
    except Exception:
        summary_preview = {}
    print(
        f"[recheck][debug] metric '{name}' not found for run {run_id}; candidates={candidates}; "
        f"summary_keys={sorted(set(summary_keys))}; summary_preview={summary_preview}",
        flush=True,
    )

    return default


# ---------------------------------------------------------------------------
# Top-k selection
# ---------------------------------------------------------------------------

def pick_topk(api: wandb.Api, sweep: Any, metric: str, maximize: bool, k: int,
              attempts: int = 5, delay: float = 15.0) -> Tuple[List[Tuple[Any, float]], Dict[str, Any]]:
    """Fetch and rank sweep runs, retrying until metrics appear or attempts exhaust."""

    attempts = max(1, int(attempts))
    delay = max(0.0, float(delay))

    diagnostics: Dict[str, Any] = {
        "attempts": 0,
        "total_runs": 0,
        "missing": [],
        "method_counts": Counter(),
    }

    metric_candidates = _metric_candidates(metric)
    print(
        f"[recheck] metric candidates for '{metric}': {metric_candidates}",
        flush=True,
    )

    if isinstance(sweep, str):
        sweep_path = sweep
    else:
        sweep_path = _safe_getattr(sweep, "path") or _safe_getattr(sweep, "sweep_path")
        if not sweep_path:
            sweep_path = _safe_getattr(sweep, "id")
        if sweep_path:
            sweep_path = str(sweep_path)
    if not sweep_path:
        raise ValueError("pick_topk requires a sweep path or object with a path")

    for attempt in range(1, attempts + 1):
        diagnostics["attempts"] = attempt
        try:
            sweep_obj = api.sweep(sweep_path)
        except Exception as exc:  # pragma: no cover - network/API defensive guard
            diagnostics["error"] = str(exc)
            print(
                f"[recheck] attempt {attempt}/{attempts}: failed to load sweep {sweep_path}: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(delay)
                continue
            break

        runs_attr = _safe_getattr(sweep_obj, "runs", [])
        runs = list(runs_attr)
        diagnostics["total_runs"] = len(runs)
        diagnostics["missing"] = []
        diagnostics["method_counts"] = Counter()

        ranked: List[Tuple[Any, float]] = []
        for run in runs:
            config_payload = _safe_getattr(run, "config", {})
            config = _coerce_config(config_payload)
            method = str(_unwrap_config_value(config.get("training_method", "unknown"))).lower()
            diagnostics["method_counts"][method] += 1

            value = metric_of(run, metric)
            if value is None:
                run_id = _safe_getattr(run, "id")
                if run_id is None:
                    run_id = _safe_getattr(run, "name")
                diagnostics["missing"].append(str(run_id))
                continue
            ranked.append((run, value))

        if ranked:
            ranked.sort(key=lambda item: item[1], reverse=maximize)
            limit = max(1, int(k))
            return ranked[:limit], diagnostics

        if len(runs) == 0:
            print(
                f"[recheck] attempt {attempt}/{attempts}: sweep {sweep_path} returned zero runs; retrying",
                flush=True,
            )
        else:
            print(
                f"[recheck] attempt {attempt}/{attempts}: waiting for metric '{metric}' to appear (runs={len(runs)})",
                flush=True,
            )

        if attempt < attempts:
            time.sleep(delay)

    return [], diagnostics


# ---------------------------------------------------------------------------
# Training launcher
# ---------------------------------------------------------------------------

def run_once(
    mm: str,
    program: str,
    subcmd: str,
    cfg: Dict[str, Any],
    seed: int,
    unlabeled: str,
    labeled: str,
    log_dir: str,
    project: Optional[str],
    group: Optional[str],
    config_idx: int,
    exp_id: Optional[str],
    device_mask: Optional[str] = None,
) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("WANDB_RUN_ID", None)
    if project and "WANDB_PROJECT" not in env:
        env["WANDB_PROJECT"] = project
    expected_group = f"recheck_cfg{config_idx}"
    env["WANDB_RUN_GROUP"] = expected_group
    run_name = f"recheck_cfg{config_idx}_seed{seed}"
    env["WANDB_NAME"] = run_name
    env["RECHECK_CONFIG_INDEX"] = str(config_idx)
    env["RECHECK_SEED"] = str(seed)
    if exp_id:
        env["RECHECK_EXP_ID"] = exp_id
    if group:
        env.setdefault("RECHECK_PARENT_GROUP", group)
    env.setdefault("WANDB_JOB_TYPE", "recheck")
    if device_mask is not None:
        mask = str(device_mask).strip()
        if mask:
            env["CUDA_VISIBLE_DEVICES"] = mask
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

    winner_cfg = _load_winner_config()
    if winner_cfg:
        merged_cfg = dict(winner_cfg)
        merged_cfg.update({k: v for k, v in cfg.items() if v is not None})
        if merged_cfg != cfg:
            source_note = f" from {WINNER_SOURCE}" if WINNER_SOURCE else ""
            print(
                f"[recheck][defaults] merged Phase-2 winner config{source_note} for cfg {config_idx}",
                flush=True,
            )
        cfg = merged_cfg

    args = [
        mm, "run", "-n", "mjepa", "env", "PYTHONUNBUFFERED=1",
        "python", "-u", program, subcmd,
        "--unlabeled-dir", unlabeled,
        "--labeled-dir", labeled,
    ]

    print(
        f"[recheck][launch] program={program} subcmd={subcmd} use_wandb=1 ",
        f"cwd={os.getcwd()} device_mask={device_mask} APP_DIR={env.get('APP_DIR')}",
        flush=True,
    )

    method = str(cfg.get("training_method", "jepa")).lower()
    args += ["--training_method", method]

    allowed = {
        "task-type", "label-col",
        "gnn-type", "hidden-dim", "num-layers", "contiguity", "add-3d",
        "learning-rate", "pretrain-batch-size", "finetune-batch-size",
        "pretrain-epochs", "finetune-epochs",
        "max-pretrain-batches", "max-finetune-batches",
        "sample-unlabeled", "sample-labeled",
        "time-budget-mins",
        "aug-bond-deletion", "aug-atom-masking", "aug-subgraph-removal",
        "aug-rotate", "aug-mask-angle", "aug-dihedral",
        "prefetch-factor", "pin-memory", "persistent-workers",
        "bf16", "num-workers", "devices", "device",
        "mask-ratio", "ema-decay",
        "temperature",
        "cache-dir", "use-wandb", "wandb-project", "wandb-tags",
        "target-pretrain-samples",
    }
    mapping = {
        "pretrain_bs": "pretrain-batch-size",
        "finetune_bs": "finetune-batch-size",
        "lr": "learning-rate",
        "label_col": "label-col",
        "gnn_type": "gnn-type",
        "add_3d": "add-3d",
        "contiguous": "contiguity",
    }
    drop = {
        "augmentations", "pair_id", "pair_key", "_wandb", "wandb_version",
        "seed", "name", "id", "group", "use_wandb",
    }

    forwarded: List[Tuple[str, Any]] = []
    for key, value in cfg.items():
        if key in drop or value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        if key == "training_method":
            continue

        cli_key = mapping.get(key, key).replace("_", "-")
        if cli_key not in allowed:
            continue

        flag = f"--{cli_key}"
        if isinstance(value, bool):
            args += [flag, "1" if value else "0"]
        else:
            args += [flag, str(value)]
        forwarded.append((flag, value))

    args += ["--use-wandb", "1"]
    # Tag recheck runs without relying on trainer CLI support
    existing_tags = env.get("WANDB_TAGS", "")
    tags = [t for t in (s.strip() for s in existing_tags.split(",")) if t]
    if "phase2-recheck" not in tags:
        tags.append("phase2-recheck")
    env["WANDB_TAGS"] = ",".join(tags)
    forwarded.append(("--use-wandb", 1))

    args += ["--seed", str(seed)]

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"recheck_{method}_seed{seed}.log")
    forwarded_flags = list(forwarded)

    seed_wall_secs: Optional[int] = None
    seed_grace_secs: int = 120
    raw_seed_wall = os.environ.get("PHASE2_SEED_WALL_SECS")
    if raw_seed_wall:
        try:
            parsed = int(float(raw_seed_wall))
            if parsed > 0:
                seed_wall_secs = parsed
        except ValueError:
            print(
                f"[recheck][warn] invalid PHASE2_SEED_WALL_SECS={raw_seed_wall!r}; ignoring",
                flush=True,
            )
    raw_seed_grace = os.environ.get("PHASE2_SEED_GRACE_SECS") or os.environ.get("PHASE2_RECHECK_GRACE_SECS")
    if raw_seed_grace:
        try:
            parsed_grace = int(float(raw_seed_grace))
            if parsed_grace > 0:
                seed_grace_secs = parsed_grace
        except ValueError:
            print(
                f"[recheck][warn] invalid seed grace value={raw_seed_grace!r}; using {seed_grace_secs}",
                flush=True,
            )

    raw_retry_env = os.environ.get("RECHECK_RUN_RETRIES")
    try:
        retries = int(raw_retry_env) if raw_retry_env is not None else 1
    except ValueError:
        print(
            f"[recheck][warn] invalid int for RECHECK_RUN_RETRIES={raw_retry_env!r}; using 1",
            flush=True,
        )
        retries = 1
    retries = max(1, retries)

    raw_delay_env = os.environ.get("RECHECK_RUN_RETRY_DELAY")
    try:
        retry_delay = float(raw_delay_env) if raw_delay_env is not None else 5.0
    except ValueError:
        print(
            f"[recheck][warn] invalid float for RECHECK_RUN_RETRY_DELAY={raw_delay_env!r}; using 5.0",
            flush=True,
        )
        retry_delay = 5.0
    retry_delay = max(0.0, retry_delay)

    default_transient = {255, 254}
    raw_transient = os.environ.get("RECHECK_TRANSIENT_RCS")
    transient_codes = set(default_transient)
    if raw_transient:
        tokens = [tok.strip() for tok in re.split(r"[;,\s]+", raw_transient) if tok.strip()]
        parsed_codes: set[int] = set()
        invalid_tokens: List[str] = []
        for token in tokens:
            try:
                parsed_codes.add(int(token))
            except ValueError:
                invalid_tokens.append(token)
        if parsed_codes:
            transient_codes = parsed_codes
        if invalid_tokens:
            joined = ", ".join(sorted(invalid_tokens))
            print(
                f"[recheck][warn] ignoring non-integer codes in RECHECK_TRANSIENT_RCS: {joined}",
                flush=True,
            )

    last_rc: Optional[int] = None

    def _append_log(message: str) -> None:
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except Exception:  # pragma: no cover - diagnostics only
            pass

    def _launch(attempt: int) -> int:
        mode = "w" if attempt == 1 else "a"
        with open(log_path, mode, encoding="utf-8") as handle:
            if attempt == 1 and forwarded_flags:
                handle.write("[recheck] forwarded flags: " + repr(forwarded_flags) + "\n")
            elif attempt > 1:
                suffix = f" (previous exit={last_rc})" if last_rc is not None else ""
                handle.write(f"[recheck] retry attempt {attempt}{suffix}\n")
            launch_args = list(args)
            if seed_wall_secs:
                launch_args = [
                    "timeout",
                    "--signal=SIGTERM",
                    "--kill-after",
                    str(seed_grace_secs),
                    str(seed_wall_secs),
                    *launch_args,
                ]
            process = subprocess.Popen(
                launch_args,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )

        msg = f"[recheck][seed {seed}] started PID {process.pid}; streaming logs to {log_path}"
        if retries > 1:
            msg += f" (attempt {attempt}/{retries})"
        print(msg, flush=True)

        start = time.time()
        next_ping = start + 60.0
        while True:
            rc_local = process.poll()
            if rc_local is not None:
                return rc_local

            now = time.time()
            if now >= next_ping:
                mins = max(1, int((now - start) // 60))
                print(
                    f"[recheck][seed {seed}] still running after {mins} min; tail -f {log_path} for live output",
                    flush=True,
                )
                next_ping = now + 60.0

            time.sleep(5.0)

    for attempt in range(1, retries + 1):
        rc = _launch(attempt)
        if rc == 0:
            return 0

        last_rc = rc
        _append_log(f"[recheck] exit code {rc} on attempt {attempt}/{retries}")
        if rc in transient_codes and attempt < retries:
            print(
                f"[recheck][seed {seed}] transient exit code {rc}; retrying ({attempt + 1}/{retries}) in {retry_delay:.1f}s",
                flush=True,
            )
            if retry_delay > 0:
                time.sleep(retry_delay)
            continue

        return rc

    return last_rc if last_rc is not None else 1


def _print_log_tail(method: str, seed: int, log_dir: str) -> None:
    log_file = os.path.join(log_dir, f"recheck_{method}_seed{seed}.log")
    try:
        with open(log_file, "r", encoding="utf-8") as handle:
            tail = handle.readlines()[-20:]
        for line in tail:
            print(f"[recheck][seed {seed}][log] {line.rstrip()}", flush=True)
    except Exception as exc:  # pragma: no cover - diagnostics
        print(f"[recheck][seed {seed}] unable to read log {log_file}: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def ci95(xs: Sequence[float]) -> Tuple[float, float]:
    bs = [np.mean(np.random.choice(xs, size=len(xs), replace=True)) for _ in range(4000)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _std_error(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    try:
        return float(np.std(xs, ddof=1) / math.sqrt(len(xs)))
    except Exception:
        return None


def _flatten_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    def _serialise_value(value: Any) -> Any:
        if isinstance(value, (list, tuple, set, dict)):
            try:
                return json.dumps(value, sort_keys=True)
            except Exception:
                return str(value)
        return value

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, inner in value.items():
                _walk(f"{prefix}.{key}" if prefix else str(key), inner)
            return
        flat[prefix] = _serialise_value(_unwrap_config_value(value))

    for key, value in config.items():
        _walk(f"config.{key}", value)

    return flat


def _metric_prefix(metric_name: Optional[str]) -> str:
    if not metric_name:
        return "metric"
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(metric_name)).strip("_")
    return cleaned or "metric"


def _write_runs_csv(
    results: Sequence[Mapping[str, Any]],
    path: Optional[str],
    index_to_run_id: Mapping[int, Optional[str]],
    primary_metric: str,
    secondary_metric: Optional[str] = None,
) -> int:
    primary_prefix = _metric_prefix(primary_metric)
    secondary_prefix = _metric_prefix(secondary_metric) if secondary_metric else None

    base_fields: List[str] = ["index", "run_id"]
    base_fields.extend(
        [
            f"{primary_prefix}_mean",
            f"{primary_prefix}_se",
            f"{primary_prefix}_ci95_low",
            f"{primary_prefix}_ci95_high",
            f"{primary_prefix}_n",
        ]
    )
    if secondary_prefix:
        base_fields.extend(
            [
                f"{secondary_prefix}_mean",
                f"{secondary_prefix}_se",
                f"{secondary_prefix}_ci95_low",
                f"{secondary_prefix}_ci95_high",
                f"{secondary_prefix}_n",
            ]
        )
    base_fields.extend(
        [
            "is_winner",
            "training_method",
            "phase2_winner_method",
            "is_phase2_winner_method",
        ]
    )

    if not path or not results:
        try:
            out_path = pathlib.Path(path) if path else None
            if out_path:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=base_fields).writeheader()
                print(f"[recheck] wrote runs CSV header → {out_path}", flush=True)
        except Exception:
            return 0
        return 0

    try:
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return 0

    winner_entry: Optional[Mapping[str, Any]] = None
    winners = [entry for entry in results if entry.get("mean") is not None]
    if winners:
        winner_entry = min(winners, key=lambda item: item.get("mean"))
    winner_index = winner_entry.get("index") if isinstance(winner_entry, Mapping) else None

    winner_method: Optional[str] = None
    if isinstance(winner_entry, Mapping):
        winner_cfg = winner_entry.get("config") if isinstance(winner_entry.get("config"), Mapping) else {}
        cfg_method = winner_cfg.get("training_method") if isinstance(winner_cfg, Mapping) else None
        if isinstance(cfg_method, str):
            winner_method = cfg_method
    env_winner = os.environ.get("METHOD_WINNER")
    if env_winner:
        winner_method = env_winner.strip()

    rows: List[Dict[str, Any]] = []
    for entry in results:
        idx = entry.get("index") if isinstance(entry, Mapping) else None
        cfg = entry.get("config") if isinstance(entry, Mapping) else None
        config_flat = _flatten_config(cfg) if isinstance(cfg, Mapping) else {}
        ci = entry.get("ci95") if isinstance(entry, Mapping) else None
        ci_low = ci[0] if isinstance(ci, (list, tuple)) and ci else None
        ci_high = ci[1] if isinstance(ci, (list, tuple)) and len(ci) > 1 else None
        ci_r2 = entry.get("ci95_r2") if isinstance(entry, Mapping) else None
        ci_r2_low = ci_r2[0] if isinstance(ci_r2, (list, tuple)) and ci_r2 else None
        ci_r2_high = ci_r2[1] if isinstance(ci_r2, (list, tuple)) and len(ci_r2) > 1 else None

        row: Dict[str, Any] = {
            "index": idx,
            "run_id": index_to_run_id.get(idx) if isinstance(idx, int) else None,
            "is_winner": bool(idx is not None and winner_index is not None and idx == winner_index),
        }
        row.update(
            {
                f"{primary_prefix}_mean": entry.get("mean") if isinstance(entry, Mapping) else None,
                f"{primary_prefix}_se": entry.get("metric_se") if isinstance(entry, Mapping) else None,
                f"{primary_prefix}_ci95_low": ci_low,
                f"{primary_prefix}_ci95_high": ci_high,
                f"{primary_prefix}_n": entry.get("n") if isinstance(entry, Mapping) else None,
            }
        )
        if secondary_prefix:
            row.update(
                {
                    f"{secondary_prefix}_mean": entry.get("r2_mean") if isinstance(entry, Mapping) else None,
                    f"{secondary_prefix}_se": entry.get("r2_se") if isinstance(entry, Mapping) else None,
                    f"{secondary_prefix}_ci95_low": ci_r2_low,
                    f"{secondary_prefix}_ci95_high": ci_r2_high,
                    f"{secondary_prefix}_n": entry.get("r2_n") if isinstance(entry, Mapping) else None,
                }
            )
        if isinstance(cfg, Mapping):
            row["training_method"] = cfg.get("training_method")
        if winner_method:
            row["phase2_winner_method"] = winner_method
            row["is_phase2_winner_method"] = (
                isinstance(cfg, Mapping) and cfg.get("training_method") == winner_method
            )
        row.update(config_flat)
        rows.append(row)

    if not rows:
        return 0

    config_fields = sorted({key for row in rows for key in row.keys() if key.startswith("config.")})
    other_fields = sorted(
        {key for row in rows for key in row.keys()} - set(base_fields) - set(config_fields)
    )

    fieldnames: List[str] = []
    fieldnames.extend(base_fields)
    fieldnames.extend(config_fields)
    fieldnames.extend(other_fields)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    print(
        f"[recheck] wrote runs CSV → {out_path} (rows={len(rows)}, hyperparams={len(config_fields)})",
        flush=True,
    )

    return len(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--project", default=os.getenv("WANDB_PROJECT"))
    ap.add_argument("--group", default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--metric", default=os.getenv("PHASE2_METRIC", "val_rmse"))
    ap.add_argument("--direction", choices=["min", "max"], default="min")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--extra_seeds", type=int, default=3)
    ap.add_argument("--program", required=True)
    ap.add_argument("--subcmd", default="sweep-run")
    ap.add_argument("--unlabeled-dir", "--unlabeled_dir", "--unlabeled", dest="unlabeled_dir", required=True)
    ap.add_argument("--labeled-dir", "--labeled_dir", "--labeled", dest="labeled_dir", required=True)
    ap.add_argument("--mm", default=os.environ.get("MMBIN", "micromamba"))
    ap.add_argument("--log_dir", default=os.environ.get("LOG_DIR", "./logs"))
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--runs-csv",
        dest="runs_csv",
        default=None,
        help="Write per-configuration aggregates to this CSV (default: $GRID_DIR/phase2_export/stage-outputs/phase2_runs.csv)",
    )
    ap.add_argument("--strict", action="store_true",
                    help="Fail if fewer than topk runs expose the metric after retries")
    ap.add_argument("--resume", action="store_true",
                    help="Skip seeds with existing local results and resume pending ones")
    default_override_devices = os.environ.get("PHASE2_RECHECK_FORCE_DEVICES")
    if default_override_devices is None:
        default_override_devices = os.environ.get("PHASE2_FORCE_DEVICES")
    ap.add_argument(
        "--override-devices",
        dest="override_devices",
        default=default_override_devices,
        help="Force a specific --devices value for each recheck launch",
    )
    args = ap.parse_args()

    resume_env = os.environ.get("PHASE2_RECHECK_RESUME")
    if resume_env is not None:
        args.resume = str(resume_env).strip().lower() in {"1", "true", "yes", "on"}

    override_devices = _coerce_positive_int(args.override_devices, "override-devices")
    args.override_devices = override_devices
    if override_devices is not None:
        print(f"[recheck] forcing devices={override_devices} for all seeds", flush=True)

    r2_metric_env = os.environ.get("PHASE2_R2_METRIC", "val_r2")
    r2_metric_name = r2_metric_env.strip() if isinstance(r2_metric_env, str) else None
    if r2_metric_name == "":
        r2_metric_name = None

    print("[recheck] args parsed:", flush=True)
    for field in (
        "sweep", "project", "group", "metric", "direction", "topk", "extra_seeds",
        "program", "subcmd", "unlabeled_dir", "labeled_dir", "mm", "log_dir", "out", "strict", "resume", "override_devices",
    ):
        print(f"  {field:<11}= {getattr(args, field)}", flush=True)

    def _safe_out(user_out: Optional[str]) -> str:
        if user_out:
            out_path = pathlib.Path(user_out)
        else:
            base = os.environ.get("GRID_DIR") or os.getcwd()
            out_path = pathlib.Path(base) / "recheck_summary.json"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            testfile = out_path.parent / ".writetest"
            with open(testfile, "w", encoding="utf-8") as handle:
                handle.write("ok")
            testfile.unlink(missing_ok=True)
        except Exception:
            out_path = pathlib.Path(tempfile.gettempdir()) / (out_path.name or "recheck_summary.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        return str(out_path)

    def _safe_dir(directory: Optional[str], fallback_name: str = "recheck_logs") -> str:
        base = pathlib.Path(directory or "./logs")
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".writetest"
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            probe.unlink(missing_ok=True)
            return str(base)
        except Exception:
            tmp = pathlib.Path(tempfile.gettempdir()) / fallback_name
            tmp.mkdir(parents=True, exist_ok=True)
            return str(tmp)

    args.out = _safe_out(args.out)
    args.log_dir = _safe_dir(args.log_dir)

    def _resolve_runs_csv(path_hint: Optional[str]) -> str:
        base = pathlib.Path(os.environ.get("GRID_DIR") or os.getcwd()) / "phase2_export" / "stage-outputs"
        if path_hint:
            candidate = pathlib.Path(path_hint)
            if not candidate.is_absolute():
                candidate = base / candidate
        else:
            candidate = base / "phase2_runs.csv"

        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / ".runs_csv.writetest"
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            probe.unlink(missing_ok=True)
        except Exception as exc:
            print(
                f"[recheck][fatal] unable to create runs CSV at {candidate} (parent={candidate.parent}): {exc}",
                flush=True,
            )
            sys.exit(4)
        return str(candidate)

    args.runs_csv = _resolve_runs_csv(args.runs_csv)

    out_dir = os.path.dirname(args.out)
    best_path = os.path.join(out_dir, "best_grid_config.json")
    backup_path = os.path.join(out_dir, "best_grid_config.phase1.json")
    tmp_path = os.path.join(out_dir, ".best_grid_config.tmp")
    incomplete_flag = os.environ.get("PHASE2_RECHECK_INCOMPLETE")
    incomplete_path = pathlib.Path(incomplete_flag) if incomplete_flag else None
    backup_taken = False
    index_to_run_id: Dict[int, Optional[str]] = {}
    last_best_index: Optional[int] = None
    results: List[Dict[str, Any]] = []
    produced_partial = False

    def _mark_incomplete() -> None:
        if incomplete_path is None:
            return
        try:
            incomplete_path.parent.mkdir(parents=True, exist_ok=True)
            with open(incomplete_path, "a", encoding="utf-8"):
                os.utime(incomplete_path, None)
            print(f"[recheck] wrote incomplete marker → {incomplete_path}", flush=True)
        except Exception as exc:
            print(f"[recheck][warn] unable to write incomplete marker {incomplete_path}: {exc}", flush=True)

    def _clear_incomplete() -> None:
        if incomplete_path is None:
            return
        try:
            incomplete_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[recheck][warn] unable to clear incomplete marker {incomplete_path}: {exc}", flush=True)

    def _ensure_backup() -> None:
        nonlocal backup_taken
        if backup_taken:
            return
        try:
            if os.path.isfile(best_path) and not os.path.isfile(backup_path):
                shutil.copy2(best_path, backup_path)
                print(f"[recheck] backed up Phase-1 best → {backup_path}", flush=True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[recheck][warn] backup skipped: {exc}", flush=True)
        backup_taken = True

    def _write_best_from_entry(entry: Mapping[str, Any]) -> None:
        nonlocal last_best_index, produced_partial
        config = entry.get("config") if isinstance(entry, Mapping) else None
        if not isinstance(config, dict) or not config:
            return
        _ensure_backup()
        payload = {
            "config": config,
            "metric": args.metric,
            "direction": args.direction,
            "source": "phase2-recheck",
        }
        run_id: Optional[str] = None
        winner_index = entry.get("index") if isinstance(entry, Mapping) else None
        if isinstance(winner_index, int):
            run_id = index_to_run_id.get(winner_index)
        if run_id:
            payload["run_id"] = run_id
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, best_path)
        if isinstance(winner_index, int) and winner_index != last_best_index:
            if run_id:
                print(
                    f"[recheck] provisional Phase-2 winner {run_id} (cfg{winner_index}) → {best_path}",
                    flush=True,
                )
            else:
                print(
                    f"[recheck] provisional Phase-2 winner cfg{winner_index} → {best_path}",
                    flush=True,
                )
            last_best_index = winner_index
        produced_partial = True

    def _current_winner(entries: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        candidates = [entry for entry in entries if isinstance(entry, Mapping) and entry.get("mean") is not None]
        if not candidates:
            return None
        if args.direction == "min":
            return min(candidates, key=lambda item: item["mean"])
        return max(candidates, key=lambda item: item["mean"])

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"[recheck][warn] invalid int for {name}={raw!r}; using {default}", flush=True)
            return default

    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"[recheck][warn] invalid float for {name}={raw!r}; using {default}", flush=True)
            return default

    attempts = max(1, _env_int("RECHECK_ATTEMPTS", 5))
    delay = max(0.0, _env_float("RECHECK_DELAY", 15.0))
    print(f"  attempts   = {attempts}", flush=True)
    print(f"  delay      = {delay}", flush=True)

    class _Heartbeat:
        def __init__(self, path: Optional[str], interval: float) -> None:
            self._path = pathlib.Path(path) if path else None
            self._interval = max(1.0, float(interval))
            self._stop = threading.Event()
            self._thread: Optional[threading.Thread] = None

        def _touch(self) -> None:
            if self._path is None:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8"):
                    os.utime(self._path, None)
            except Exception as exc:
                print(f"[recheck][warn] heartbeat update failed: {exc}", flush=True)

        def _run(self) -> None:
            while not self._stop.wait(self._interval):
                self._touch()

        def start(self) -> None:
            if self._path is None:
                return
            self._touch()
            thread = threading.Thread(target=self._run, daemon=True)
            thread.start()
            self._thread = thread

        def stop(self) -> None:
            if self._path is None:
                return
            self._stop.set()
            self._touch()
            thread = self._thread
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

    def _parse_entity_project(sweep: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Try to infer (entity, project) from a sweep path or URL.
        Accepts: 'entity/project/sweeps/<id>' or 'entity/project/<id>' or full https URL.
        """
        s = (sweep or "").strip()
        parts: List[str] = []
        if s.startswith("http://") or s.startswith("https://"):
            try:
                path = urlparse(s).path.strip("/")
                parts = [p for p in path.split("/") if p]
            except Exception:
                parts = []
        else:
            parts = [p for p in s.strip("/").split("/") if p]
        if len(parts) >= 2:
            # Common patterns put entity at 0 and project at 1
            return parts[0], parts[1]
        return None, None

    def _serialize_diagnostics(diag: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in (diag or {}).items():
            if isinstance(value, Counter):
                payload[key] = dict(value)
            else:
                payload[key] = value
        return payload

    def _write_summary(results: List[Dict[str, Any]], diag: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "metric": args.metric,
            "direction": args.direction,
            "topk": args.topk,
            "extra_seeds": args.extra_seeds,
            "results": results,
        }
        serialized_diag = _serialize_diagnostics(diag)
        if serialized_diag:
            payload["diagnostics"] = serialized_diag
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    heartbeat_path = os.environ.get("PHASE2_RECHECK_HEARTBEAT")
    heartbeat_interval = _env_float("PHASE2_RECHECK_HEARTBEAT_SECS", 120.0)
    heartbeat = _Heartbeat(heartbeat_path, heartbeat_interval)
    heartbeat.start()
    sentinel_path = os.environ.get("PHASE2_RECHECK_SENTINEL")

    interrupted = {"flag": False}

    def _handle_signal(signum: int, _: Optional[object]) -> None:
        if not interrupted["flag"]:
            interrupted["flag"] = True
            print(f"[recheck] received signal {signum}; flushing partial state", flush=True)
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:  # pragma: no cover - signal registration best effort
            pass

    success = False
    timeout = resolve_wandb_http_timeout(60)

    try:
        api = wandb.Api(timeout=timeout)
    except TypeError:
        print(
            "[recheck][warn] wandb.Api does not accept a timeout argument; retrying without it",
            flush=True,
        )
        try:
            api = wandb.Api()
        except Exception:
            print("[recheck][fatal] unable to initialise wandb.Api", flush=True)
            raise
    except Exception:
        print("[recheck][fatal] unable to initialise wandb.Api", flush=True)
        raise

    try:
        # Derive entity/project if env vars are missing (avoids empty fetches).
        derived_entity, derived_project = _parse_entity_project(args.sweep)
        entity = os.getenv("WANDB_ENTITY") or derived_entity
        if not args.project:
            args.project = derived_project

        maximize = args.direction == "max"
        top_entries, diagnostics = pick_topk(api, args.sweep, args.metric, maximize, args.topk, attempts, delay)

        if not top_entries:
            total_runs = diagnostics.get("total_runs", 0)
            missing = diagnostics.get("missing", [])
            method_counts = diagnostics.get("method_counts", Counter())

            if total_runs:
                print(
                    f"[recheck][fatal] unable to locate metric '{args.metric}' in sweep {args.sweep} "
                    f"after {diagnostics.get('attempts', attempts)} attempt(s); missing metrics from {len(missing)} run(s)",
                    flush=True,
                )
                if method_counts:
                    print(
                        "[recheck] sweep method counts: "
                        + ", ".join(f"{m or 'unknown'}={c}" for m, c in sorted(method_counts.items())),
                        flush=True,
                    )
                if missing:
                    preview = ", ".join(missing[:5])
                    suffix = "" if len(missing) <= 5 else f" … (+{len(missing) - 5} more)"
                    print(f"[recheck] runs missing metric: {preview}{suffix}", flush=True)
                if diagnostics.get("error"):
                    print(f"[recheck] last API error: {diagnostics['error']}", flush=True)
                sys.exit(2)

            # Empty sweeps should not be treated as fatal: emit an empty summary so callers
            # can decide how to proceed (and the CLI remains testable).
            print(
                f"[recheck][warn] sweep {args.sweep} returned zero runs after {diagnostics.get('attempts', attempts)} attempt(s); "
                "writing empty summary",
                flush=True,
            )
            if diagnostics.get("error"):
                print(f"[recheck] last API error: {diagnostics['error']}", flush=True)
            _write_summary([], diagnostics)
            success = True
            return

        if args.strict and len(top_entries) < max(1, args.topk):
            print(
                f"[recheck][fatal] strict mode requires {args.topk} runs but only {len(top_entries)} expose '{args.metric}'",
                flush=True,
            )
            sys.exit(2)

        top_runs = [run for run, _ in top_entries]
        index_to_run_id.clear()
        for idx, run in enumerate(top_runs):
            run_identifier = _safe_getattr(run, "id") or _safe_getattr(run, "name")
            if run_identifier:
                index_to_run_id[idx] = str(run_identifier)
        print(
            "[recheck] picked top runs: " + ", ".join(getattr(run, "id", getattr(run, "name", "?")) for run in top_runs),
            flush=True,
        )
        for run, value in top_entries:
            print(
                f"  metric[{args.metric}] → {getattr(run, 'id', getattr(run, 'name', '?'))}: {value}",
                flush=True,
            )

        missing = diagnostics.get("missing", [])
        if missing:
            print(f"[recheck] ignored {len(missing)} runs without metric '{args.metric}'", flush=True)

        method_counts = diagnostics.get("method_counts", Counter())
        if method_counts:
            print(
                "[recheck] sweep method counts: "
                + ", ".join(f"{method or 'unknown'}={count}" for method, count in sorted(method_counts.items())),
                flush=True,
            )

        def _kv(cfg: Mapping[str, Any], keys: Sequence[str] = (
            "training_method", "gnn_type", "hidden_dim", "num_layers", "contiguity",
            "mask_ratio", "ema_decay", "temperature", "learning_rate",
        )) -> Dict[str, Any]:
            return {key: cfg.get(key) for key in keys if cfg.get(key) is not None}

        for run in top_runs:
            print(f"  - {run.id}: {_kv(_coerce_config(getattr(run, 'config', {})))}", flush=True)

        top_configs: List[Dict[str, Any]] = []
        for run in top_runs:
            cfg: Dict[str, Any] = {}
            raw_cfg = _coerce_config(getattr(run, "config", {}))
            for key, value in raw_cfg.items():
                if key.startswith("_"):
                    continue
                cfg[key] = _unwrap_config_value(value)
            cfg.pop("seed", None)
            _apply_config_overrides(cfg, override_devices=args.override_devices)
            top_configs.append(cfg)

        results.clear()
        offline_mode = (os.getenv("WANDB_MODE", "").lower() == "offline")
        exp_id = _resolve_exp_id(args.group)
        total_completed = 0
        total_pending = 0
        plan_entries: List[Tuple[int, Dict[str, Any], List[int], List[int], List[int]]] = []
        for index, cfg in enumerate(top_configs):
            seeds = [1000 + index * 10 + s for s in range(args.extra_seeds)]
            completed_seeds: List[int] = []
            pending_seeds = list(seeds)
            if args.resume:
                pending_seeds = []
                for seed in seeds:
                    val, _ = _read_local_result(_local_result_path(exp_id, index, seed))
                    if val is not None:
                        completed_seeds.append(seed)
                    else:
                        pending_seeds.append(seed)
            total_completed += len(completed_seeds)
            total_pending += len(pending_seeds)
            plan_entries.append((index, cfg, seeds, completed_seeds, pending_seeds))

        print(
            f"[ci] recheck resume plan: {{completed:{total_completed}, pending:{total_pending}}}",
            flush=True,
        )

        visible_gpu_ids = _discover_visible_gpu_ids()
        agent_workers = _resolve_agent_count(visible_gpu_ids, args.override_devices)
        default_gpu_mask = ",".join(visible_gpu_ids) if visible_gpu_ids else ""
        if not visible_gpu_ids and agent_workers > 1:
            print(
                f"[recheck][warn] requested {agent_workers} workers but no GPUs detected; running sequentially",
                flush=True,
            )
            agent_workers = 1
        gpu_warned: set[Tuple[int, int]] = set()

        for index, cfg, seeds, completed_seeds, pending_seeds in plan_entries:
            _print_config_banner(index, seeds, args.project, entity, offline_mode)
            if pending_seeds:
                skip_note = f" (skipping {completed_seeds})" if completed_seeds else ""
                print(
                    f"[recheck] launching seeds {pending_seeds} for config index {index}{skip_note}",
                    flush=True,
                )
                method = str(cfg.get("training_method", "jepa")).lower()
                per_run_devices = _extract_positive_int(cfg.get("devices")) or 1
                worker_masks: List[Optional[str]] = []
                effective_workers = agent_workers
                gpu_masks: List[str] = []
                if visible_gpu_ids:
                    effective_workers, gpu_masks = _compute_worker_gpu_masks(
                        visible_gpu_ids, agent_workers, per_run_devices
                    )
                    if per_run_devices > 1 and agent_workers > effective_workers:
                        key = (agent_workers, per_run_devices)
                        if key not in gpu_warned:
                            print(
                                "[recheck][warn] reducing worker count to "
                                f"{effective_workers} because each run requests "
                                f"{per_run_devices} GPU(s) but only {len(visible_gpu_ids)} visible",
                                flush=True,
                            )
                            gpu_warned.add(key)
                if gpu_masks:
                    worker_masks = [mask or None for mask in gpu_masks]
                worker_count = len(worker_masks) if worker_masks else effective_workers
                if worker_count <= 0:
                    worker_count = 1

                if worker_count > 1:
                    message = (
                        f"[recheck] enabling parallel recheck across {worker_count} worker(s)"
                    )
                    if per_run_devices > 1:
                        message += f" with {per_run_devices} GPU(s) per worker"
                    if gpu_masks:
                        message += f"; GPU splits={gpu_masks}"
                    print(message, flush=True)

                if worker_count == 1 or len(pending_seeds) <= worker_count:
                    masks: List[Optional[str]]
                    if worker_masks:
                        masks = worker_masks
                    elif default_gpu_mask:
                        masks = [default_gpu_mask]
                    else:
                        masks = [None]

                    for idx_seed, seed in enumerate(pending_seeds):
                        mask = masks[idx_seed % len(masks)]
                        effective_mask = mask
                        if not effective_mask and default_gpu_mask:
                            effective_mask = default_gpu_mask
                        rc = run_once(
                            args.mm,
                            args.program,
                            args.subcmd,
                            cfg,
                            seed,
                            args.unlabeled_dir,
                            args.labeled_dir,
                            args.log_dir,
                            args.project,
                            args.group,
                            index,
                            exp_id,
                            device_mask=effective_mask,
                        )
                        print(f"[recheck][seed {seed}] finished with exit code {rc}", flush=True)
                        if rc != 0:
                            _print_log_tail(method, seed, args.log_dir)
                            raise RuntimeError(f"recheck run failed with exit code {rc} (seed {seed})")
                        time.sleep(1.0)
                else:
                    worker_count = min(worker_count, len(pending_seeds))
                    worker_masks = worker_masks[:worker_count]

                    job_queue: "queue.Queue[Optional[int]]" = queue.Queue()
                    for seed in pending_seeds:
                        job_queue.put(seed)
                    for _ in range(worker_count):
                        job_queue.put(None)

                    error_event = threading.Event()
                    error_lock = threading.Lock()
                    errors: List[BaseException] = []

                    def _worker(worker_idx: int, mask: Optional[str]) -> None:
                        while True:
                            seed_item = job_queue.get()
                            if seed_item is None:
                                job_queue.task_done()
                                break
                            try:
                                if error_event.is_set():
                                    continue
                                effective_mask = mask if mask is not None else (default_gpu_mask or None)
                                rc = run_once(
                                    args.mm,
                                    args.program,
                                    args.subcmd,
                                    cfg,
                                    seed_item,
                                    args.unlabeled_dir,
                                    args.labeled_dir,
                                    args.log_dir,
                                    args.project,
                                    args.group,
                                    index,
                                    exp_id,
                                    device_mask=effective_mask,
                                )
                                print(f"[recheck][seed {seed_item}] finished with exit code {rc}", flush=True)
                                if rc != 0:
                                    _print_log_tail(method, seed_item, args.log_dir)
                                    raise RuntimeError(
                                        f"recheck run failed with exit code {rc} (seed {seed_item})"
                                    )
                                time.sleep(1.0)
                            except Exception as exc:  # pragma: no cover - worker diagnostics
                                with error_lock:
                                    if not errors:
                                        errors.append(exc)
                                error_event.set()
                            finally:
                                job_queue.task_done()

                    threads: List[threading.Thread] = []
                    for idx_worker, mask in enumerate(worker_masks):
                        thread = threading.Thread(target=_worker, args=(idx_worker, mask), daemon=True)
                        thread.start()
                        threads.append(thread)

                    job_queue.join()

                    for thread in threads:
                        thread.join()

                    if errors:
                        raise errors[0]
            else:
                print(
                    f"[recheck] all seeds completed for config index {index}; skipping launches",
                    flush=True,
                )

            attempts_env = os.getenv("RECHECK_COLLECT_ATTEMPTS")
            try:
                collect_attempts = int(attempts_env) if attempts_env else 5
            except Exception:
                collect_attempts = 5
            collect_attempts = max(1, collect_attempts)
            delay_env = os.getenv("RECHECK_COLLECT_DELAY")
            try:
                collect_delay = float(delay_env) if delay_env else 15.0
            except Exception:
                collect_delay = 15.0
            collect_delay = max(0.0, collect_delay)
            project_path = f"{entity}/{args.project}" if entity and args.project else None
            seed_to_val = _collect_seed_metrics(
                api,
                project_path,
                cfg,
                seeds,
                args.metric,
                index,
                exp_id,
                collect_attempts,
                collect_delay,
            )
            if r2_metric_name:
                seed_to_r2 = _collect_seed_metrics(
                    api,
                    project_path,
                    cfg,
                    seeds,
                    r2_metric_name,
                    index,
                    exp_id,
                    collect_attempts,
                    collect_delay,
                    warn_missing=False,
                )
            else:
                seed_to_r2 = {}

            missing_seeds = [s for s in seeds if s not in seed_to_val]
            for missing_seed in missing_seeds:
                _warn_missing_metric(args.metric, missing_seed)

            values = [seed_to_val[s] for s in seeds if s in seed_to_val]
            r2_values = [seed_to_r2[s] for s in seeds if s in seed_to_r2]
            r2_mean = float(np.mean(r2_values)) if r2_values else None
            r2_low, r2_high = ci95(r2_values) if r2_values else (None, None)
            r2_se = _std_error(r2_values)

            if values:
                mean_value = float(np.mean(values))
                low, high = ci95(values)
                entry = {
                    "index": index,
                    "mean": mean_value,
                    "ci95": [low, high],
                    "n": len(values),
                    "metric_se": _std_error(values),
                    "r2_mean": r2_mean,
                    "ci95_r2": [r2_low, r2_high],
                    "r2_n": len(r2_values),
                    "r2_se": r2_se,
                    "config": cfg,
                }
            else:
                entry = {
                    "index": index,
                    "mean": None,
                    "ci95": [None, None],
                    "n": 0,
                    "metric_se": None,
                    "r2_mean": r2_mean,
                    "ci95_r2": [r2_low, r2_high],
                    "r2_n": len(r2_values),
                    "r2_se": r2_se,
                    "config": cfg,
                }

            results.append(entry)
            produced_partial = True
            diagnostics["completed_configs"] = len(results)
            _write_summary(results, diagnostics)
            winner_entry = _current_winner(results)
            if winner_entry is not None:
                _write_best_from_entry(winner_entry)

        successful = [entry for entry in results if entry.get("mean") is not None]

        if not successful:
            print(
                f"[recheck][fatal] no metrics collected for metric '{args.metric}' in sweep {args.sweep}; see {args.out}",
                flush=True,
            )
            sys.exit(3)

        if args.direction == "min":
            winner = min(successful, key=lambda item: item["mean"])
        else:
            winner = max(successful, key=lambda item: item["mean"])

        winner_cfg = winner.get("config") if isinstance(winner, Mapping) else None
        if not isinstance(winner_cfg, dict) or not winner_cfg:
            print(
                f"[recheck][fatal] best configuration for sweep {args.sweep} produced an empty config; inspect {args.out}",
                flush=True,
            )
            sys.exit(3)

        _write_best_from_entry(winner)
        winner_index = winner.get("index") if isinstance(winner, Mapping) else None
        winner_mean = winner.get("mean") if isinstance(winner, Mapping) else None
        run_id_final = index_to_run_id.get(winner_index) if isinstance(winner_index, int) else None
        if isinstance(winner_index, int):
            if run_id_final:
                print(
                    f"[recheck] final Phase-2 winner {run_id_final} (cfg{winner_index}) metric={winner_mean} → {best_path}",
                    flush=True,
                )
            else:
                print(
                    f"[recheck] final Phase-2 winner cfg{winner_index} metric={winner_mean} → {best_path}",
                    flush=True,
                )
        else:
            print(f"[recheck] final Phase-2 winner saved → {best_path}", flush=True)
        success = True
    except KeyboardInterrupt:
        produced_partial = True
        print("[recheck] interrupted by signal; aborting", flush=True)
        raise
    finally:
        heartbeat.stop()
        try:
            rows_written = _write_runs_csv(
                results, args.runs_csv, index_to_run_id, args.metric, r2_metric_name
            )
            if results and rows_written <= 0:
                raise RuntimeError(
                    f"expected runs CSV rows for {len(results)} configs but wrote {rows_written}"
                )
        except Exception as exc:
            print(f"[recheck][fatal] unable to write runs CSV {args.runs_csv}: {exc}", flush=True)
            sys.exit(5)
        if success and sentinel_path:
            try:
                sentinel = pathlib.Path(sentinel_path)
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                with open(sentinel, "a", encoding="utf-8"):
                    os.utime(sentinel, None)
                print(f"[recheck] wrote sentinel → {sentinel}", flush=True)
            except Exception as exc:
                print(f"[recheck][warn] unable to write sentinel {sentinel_path}: {exc}", flush=True)
            _clear_incomplete()
        elif not success and sentinel_path:
            if produced_partial or results:
                _mark_incomplete()
            print("[recheck][warn] recheck did not complete successfully; sentinel not written", flush=True)
        elif not success and (produced_partial or results):
            _mark_incomplete()




def _expected_group(config_idx: int) -> str:
    return f"recheck_cfg{config_idx}"


def _expected_run_name(config_idx: int, seed: int) -> str:
    return f"{_expected_group(config_idx)}_seed{seed}"


def _local_result_path(exp_id: Optional[str], config_idx: int, seed: int) -> Optional[pathlib.Path]:
    if not exp_id:
        return None
    base = pathlib.Path("/data/mjepa/experiments") / exp_id / "recheck_results"
    return base / f"cfg{config_idx}_seed{seed}.json"


def _read_local_result(path: Optional[pathlib.Path]) -> Tuple[Optional[float], Optional[int]]:
    if path is None:
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None
    val = payload.get("val_rmse")
    best = payload.get("best_step")
    try:
        val_f = float(val) if val is not None else None
    except Exception:
        val_f = None
    try:
        best_i = int(best) if best is not None else None
    except Exception:
        best_i = None
    return val_f, best_i


def _print_config_banner(
    config_idx: int,
    seeds: Sequence[int],
    project: Optional[str],
    entity: Optional[str],
    offline_mode: bool,
) -> None:
    sample_seed = seeds[0] if seeds else 0
    run_name = _expected_run_name(config_idx, sample_seed)
    group = _expected_group(config_idx)
    offline = "yes" if offline_mode else "no"
    print(
        f"[recheck] config {config_idx}: project={project or '-'} entity={entity or '-'} offline={offline}; "
        f"expected group={group} run≈{run_name}",
        flush=True,
    )


def _warn_missing_metric(metric: str, seed: int) -> None:
    print(
        f"[recheck][warn] missing metric '{metric}' for seed {seed} after retries; skipping",
        flush=True,
    )


def _best_effort_wandb_sync() -> None:
    cache_dir = os.getenv("WANDB_CACHE_DIR") or os.getenv("WANDB_DIR")
    if not cache_dir:
        return
    path = pathlib.Path(cache_dir)
    if not path.exists():
        return
    cmd = ["wandb", "sync", "--include-offline", str(path)]
    try:
        print(f"[recheck] triggering wandb sync via: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("[recheck][warn] wandb CLI not available for offline sync", flush=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[recheck][warn] wandb sync command failed: {exc}", flush=True)


def _resolve_exp_id(group_hint: Optional[str]) -> Optional[str]:
    candidate = (
        os.getenv("RECHECK_EXP_ID")
        or group_hint
        or os.getenv("WANDB_RUN_GROUP")
        or os.getenv("WANDB_NAME")
    )
    if not candidate:
        return None
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", str(candidate))
    sanitized = sanitized.strip("_")
    return sanitized or None


def _collect_seed_metrics(
    api: wandb.Api,
    project_path: Optional[str],
    cfg: Mapping[str, Any],
    seeds: Sequence[int],
    metric: str,
    config_idx: int,
    exp_id: Optional[str],
    attempts: int,
    delay: float,
    warn_missing: bool = True,
) -> Dict[int, float]:
    seeds = [int(s) for s in seeds]
    seed_to_val: Dict[int, float] = {}
    used_fallback: set[int] = set()
    expected_group = _expected_group(config_idx)
    exp_norm = _norm_cfg(cfg)
    ident_keys = ("training_method", "gnn_type", "hidden_dim", "num_layers", "contiguity")
    total_attempts = max(1, attempts)

    for attempt in range(1, total_attempts + 1):
        fetched_runs: Sequence[Any] = []
        if project_path:
            try:
                fetched_runs = list(api.runs(project_path, filters={"group": expected_group, "jobType": "recheck"}))
            except Exception as exc:
                print(f"[recheck] fetch attempt #{attempt} failed: {exc}", flush=True)
                fetched_runs = []

        for run in fetched_runs:
            run_cfg = _coerce_config(getattr(run, "config", {}))
            run_seed = _unwrap_config_value(run_cfg.get("seed"))
            try:
                run_seed_int = int(run_seed)
            except Exception:
                continue
            if run_seed_int not in seeds or run_seed_int in seed_to_val:
                continue

            run_name = _safe_getattr(run, "name")
            if run_name:
                expected_name = _expected_run_name(config_idx, run_seed_int)
                # Only enforce the strict naming convention when the run name
                # follows the recheck pattern. Historical runs (or locally
                # triggered evaluations) might use arbitrary names; we still
                # want to collect their metrics as long as the config matches.
                if str(run_name).startswith("recheck_cfg") and run_name != expected_name:
                    continue
            run_group = _safe_getattr(run, "group")
            if run_group and run_group != expected_group:
                continue

            rc = _norm_cfg(run_cfg)
            ok = True
            for k in ident_keys:
                ev = exp_norm.get(k)
                rv = rc.get(k)
                if rv is None and "_" in k:
                    rv = rc.get(k.replace("_", "-"))
                if ev is None or rv is None:
                    continue
                if not _num_close(rv, ev):
                    ok = False
                    break
            if not ok:
                continue

            mv = metric_of(run, metric)
            if mv is None:
                continue
            try:
                seed_to_val[run_seed_int] = float(mv)
            except Exception:
                continue

        missing = [s for s in seeds if s not in seed_to_val]
        for seed_id in missing:
            path = _local_result_path(exp_id, config_idx, seed_id)
            val, _ = _read_local_result(path)
            if val is None:
                continue
            seed_to_val[seed_id] = val
            if seed_id not in used_fallback and path is not None:
                print(f"[recheck] using local fallback for seed {seed_id}: {path}", flush=True)
                used_fallback.add(seed_id)

        if len(seed_to_val) == len(seeds):
            break

        if attempt == 3:
            _best_effort_wandb_sync()

        if attempt < total_attempts and warn_missing:
            remaining = [s for s in seeds if s not in seed_to_val]
            print(
                f"[recheck] waiting for metrics or fallback ({attempt}/{total_attempts}); missing seeds={remaining}",
                flush=True,
            )
            time.sleep(max(0.0, delay))

    return seed_to_val


if __name__ == "__main__":
    main()

