#!/usr/bin/env python3
"""Recheck the top-k runs from a Phase-2 sweep and materialise the winner."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import wandb


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

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


def _metric_candidates(metric: str) -> List[str]:
    candidates = [metric]
    if "/" in metric:
        candidates.append(metric.replace("/", "."))
    if "." in metric:
        candidates.append(metric.replace(".", "/"))
    return list(dict.fromkeys(candidates))


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

    # Try summaries first (fast path)
    for summary in summary_sources:
        for candidate in candidates:
            numeric = _coerce_numeric(_lookup_nested(summary, candidate))
            if numeric is not None:
                return numeric

    # 2) History fallback: latest non-NaN/finite across candidates
    history_value, _ = _history_latest(run, candidates)
    if history_value is not None:
        return history_value

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
            method = str(config.get("training_method", "unknown")).lower()
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

def run_once(mm: str, program: str, subcmd: str, cfg: Dict[str, Any], seed: int,
             unlabeled: str, labeled: str, log_dir: str,
             project: Optional[str], group: Optional[str]) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if project and "WANDB_PROJECT" not in env:
        env["WANDB_PROJECT"] = project
    if group and "WANDB_RUN_GROUP" not in env:
        env["WANDB_RUN_GROUP"] = group
    env.setdefault("WANDB_JOB_TYPE", "recheck")

    args = [
        mm, "run", "-n", "mjepa", "env", "PYTHONUNBUFFERED=1",
        "python", "-u", program, subcmd,
        "--unlabeled-dir", unlabeled,
        "--labeled-dir", labeled,
    ]

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
    # help triage recheck runs in the UI
    args += ["--wandb-tags", "phase2-recheck"]
    forwarded.append(("--use-wandb", 1))

    args += ["--seed", str(seed)]

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"recheck_{method}_seed{seed}.log")

    with open(log_path, "w", encoding="utf-8") as handle:
        if forwarded:
            handle.write("[recheck] forwarded flags: " + repr(forwarded) + "\n")
        process = subprocess.Popen(
            args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print(f"[recheck][seed {seed}] started PID {process.pid}; streaming logs to {log_path}", flush=True)

    start = time.time()
    next_ping = start + 60.0
    while True:
        rc = process.poll()
        if rc is not None:
            return rc

        now = time.time()
        if now >= next_ping:
            mins = max(1, int((now - start) // 60))
            print(
                f"[recheck][seed {seed}] still running after {mins} min; tail -f {log_path} for live output",
                flush=True,
            )
            next_ping = now + 60.0

        time.sleep(5.0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def ci95(xs: Sequence[float]) -> Tuple[float, float]:
    bs = [np.mean(np.random.choice(xs, size=len(xs), replace=True)) for _ in range(4000)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


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
    ap.add_argument("--strict", action="store_true",
                    help="Fail if fewer than topk runs expose the metric after retries")
    args = ap.parse_args()

    print("[recheck] args parsed:", flush=True)
    for field in (
        "sweep", "project", "group", "metric", "direction", "topk", "extra_seeds",
        "program", "subcmd", "unlabeled_dir", "labeled_dir", "mm", "log_dir", "out", "strict",
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

    def _serialize_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in (diag or {}).items():
            if isinstance(value, Counter):
                payload[key] = dict(value)
            else:
                payload[key] = value
        return payload

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

    api = wandb.Api()
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
        return

    if args.strict and len(top_entries) < max(1, args.topk):
        print(
            f"[recheck][fatal] strict mode requires {args.topk} runs but only {len(top_entries)} expose '{args.metric}'",
            flush=True,
        )
        sys.exit(2)

    top_runs = [run for run, _ in top_entries]
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
        top_configs.append(cfg)

    results: List[Dict[str, Any]] = []
    for index, cfg in enumerate(top_configs):
        seeds = [1000 + index * 10 + s for s in range(args.extra_seeds)]
        values: List[float] = []
        print(f"[recheck] launching seeds {seeds} for config index {index}", flush=True)
        for seed in seeds:
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
            )
            print(f"[recheck][seed {seed}] finished with exit code {rc}", flush=True)
            if rc != 0:
                log_file = os.path.join(args.log_dir, f"recheck_{cfg.get('training_method', 'jepa')}_seed{seed}.log")
                try:
                    with open(log_file, "r", encoding="utf-8") as handle:
                        tail = handle.readlines()[-20:]
                    for line in tail:
                        print(f"[recheck][seed {seed}][log] {line.rstrip()}", flush=True)
                except Exception as exc:  # pragma: no cover - diagnostics
                    print(f"[recheck][seed {seed}] unable to read log {log_file}: {exc}", flush=True)
                raise RuntimeError(f"recheck run failed with exit code {rc} (seed {seed})")
            time.sleep(1.0)

       # Post-seed fetch with small retry to absorb W&B eventual consistency.
        collect_attempts = max(1, int(os.getenv("RECHECK_COLLECT_ATTEMPTS", "5")))
        collect_delay    = max(0.0, float(os.getenv("RECHECK_COLLECT_DELAY", "5")))
        for _try in range(1, collect_attempts + 1):
            project_path = f"{entity}/{args.project}" if entity and args.project else None
            fetched_runs: Sequence[Any] = []
            if project_path:
                try:
                    fetched_runs = list(api.runs(project_path, filters={"group": args.group}))
                except Exception as exc:
                    print(f"[recheck] fetch attempt #{_try} failed: {exc}", flush=True)
                    fetched_runs = []
            # Match by config & seed
            values.clear()
            for run in fetched_runs:
                run_cfg = _coerce_config(getattr(run, "config", {}))
                matches = True
                for key, value in cfg.items():
                    if _unwrap_config_value(run_cfg.get(key)) != value:
                        matches = False
                        break
                run_seed = _unwrap_config_value(run_cfg.get("seed"))
                if matches and run_seed in seeds:
                    mv = metric_of(run, args.metric)
                    if mv is not None:
                        values.append(float(mv))
            if values:
                break
            if _try < collect_attempts:
                print(f"[recheck] waiting for metrics to sync ({_try}/{collect_attempts}); found 0 values", flush=True)
                time.sleep(collect_delay)

        if values:
            mean_value = float(np.mean(values))
            low, high = ci95(values)
            results.append({
                "index": index,
                "mean": mean_value,
                "ci95": [low, high],
                "n": len(values),
                "config": cfg,
            })
        else:

            results.append({
                "index": index,
                "mean": None,
                "ci95": [None, None],
                "n": 0,
                "config": cfg,
            })

    successful = [entry for entry in results if entry.get("mean") is not None]
    _write_summary(results, diagnostics)

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

    winner_index = winner.get("index")
    winner_run_id: Optional[str] = None
    if isinstance(winner_index, int) and 0 <= winner_index < len(top_runs):
        winner_run_id = getattr(top_runs[winner_index], "id", None) or getattr(top_runs[winner_index], "name", None)

    out_dir = os.path.dirname(args.out)
    best_path = os.path.join(out_dir, "best_grid_config.json")
    backup_path = os.path.join(out_dir, "best_grid_config.phase1.json")
    tmp_path = os.path.join(out_dir, ".best_grid_config.tmp")

    try:
        if os.path.isfile(best_path) and not os.path.isfile(backup_path):
            import shutil
            shutil.copy2(best_path, backup_path)
            print(f"[recheck] backed up Phase-1 best → {backup_path}", flush=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[recheck][warn] backup skipped: {exc}", flush=True)

    payload = {
        "config": winner_cfg,
        "metric": args.metric,
        "direction": args.direction,
        "source": "phase2-recheck",
        "run_id": winner_run_id,
    }

    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, best_path)
    print(f"[recheck] wrote Phase-2 winner → {best_path}", flush=True)


if __name__ == "__main__":
    main()
