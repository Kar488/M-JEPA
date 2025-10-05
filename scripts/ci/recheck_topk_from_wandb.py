#!/usr/bin/env python3
"""
Pull top-k configs from a Phase-2 sweep, re-run each on extra seeds,
and save a summary JSON with averaged metrics and 95% CI per config.

Usage:
  python recheck_topk_from_wandb.py \
    --sweep "entity/project/abcdef" \
    --project mjepa --group "${WANDB_RUN_GROUP}" \
    --metric val_rmse --direction min \
    --topk 5 --extra_seeds 3 \
    --program "${env:APP_DIR}/scripts/train_jepa.py" \
    --subcmd "sweep-run" \
    --unlabeled-dir "${env:APP_DIR}/data/ZINC-canonicalized" \
    --labeled-dir   "${env:APP_DIR}/data/katielinkmoleculenet_benchmark/train" \
    --out "${GRID_DIR}/recheck_summary.json"
"""
import argparse, json, os, math, time, tempfile, pathlib, sys
from collections import Counter
from collections.abc import Mapping
from itertools import islice
from typing import List, Dict, Any, Tuple, Iterable, Optional
import numpy as np
import wandb
import subprocess
import shlex

def _coerce_config(config: Any) -> Dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)

    to_dict_attr = getattr(config, "to_dict", None)
    if callable(to_dict_attr):
        try:
            converted = to_dict_attr()
        except Exception:
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

    json_dict = getattr(config, "_json_dict", None)
    if isinstance(json_dict, dict):
        return dict(json_dict)

    items_attr = getattr(config, "items", None)
    if callable(items_attr):
        try:
            items_value = items_attr()
        except Exception:
            items_value = None
        else:
            try:
                return dict(items_value)
            except Exception:
                try:
                    return dict(config)
                except Exception:
                    pass

    try:
        return dict(config)
    except Exception:
        pass

    return {}


def _unwrap_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "value" in value and len(value) <= 2:
            # Common wandb wrapper for nested config values
            return _unwrap_config_value(value.get("value"))
        for key in ("value", "values", "_value"):
            if key in value:
                inner = value[key]
                return _unwrap_config_value(inner)
        if "_type" in value and value.get("_type") == "quantized_params":
            params = value.get("params")
            if isinstance(params, Mapping):
                return {k: _unwrap_config_value(v) for k, v in params.items()}
        if len(value) == 1:
            inner = next(iter(value.values()))
            return _unwrap_config_value(inner)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_config_value(value[0])
    return value


def _extract_summary_payload(run: Any) -> Dict[str, Any]:
    summary_sources = []

    try:
        summary_sources.append(getattr(run, "summary"))
    except Exception:
        summary_sources.append(None)

    try:
        summary_sources.append(getattr(run, "summary_metrics"))
    except Exception:
        summary_sources.append(None)

    try:
        summary_sources.append(getattr(run, "summaryMetrics"))
    except Exception:
        summary_sources.append(None)

    attrs = getattr(run, "_attrs", None)
    if isinstance(attrs, Mapping):
        for key in ("summaryMetrics", "summary_metrics"):
            if key in attrs:
                summary_sources.append(attrs.get(key))

    for source in summary_sources:
        summary = _coerce_config(source or {})
        if summary:
            return summary

    return {}


def _lookup_nested(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]

    for sep in ("/", "."):
        if sep not in key:
            continue
        parts = key.split(sep)
        node: Any = mapping
        for part in parts:
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                node = None
                break
        if node is not None:
            return node
    return None


def _coerce_to_float(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, Mapping):
        for key in ("value", "latest", "last", "mean", "median", "max", "min", "best"):
            if key in value:
                coerced = _coerce_to_float(value[key])
                if coerced is not None:
                    return coerced
        for candidate in value.values():
            coerced = _coerce_to_float(candidate)
            if coerced is not None:
                return coerced
        return None

    if isinstance(value, (list, tuple)):
        for candidate in value:
            coerced = _coerce_to_float(candidate)
            if coerced is not None:
                return coerced
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_from_history(run: Any, candidates: Iterable[str], limit: int = 512) -> Tuple[Optional[float], Optional[str]]:
    history = getattr(run, "history", None)
    if not callable(history):
        return None, None

    rows: List[Dict[str, Any]] = []
    history_iters = (
        {"keys": list(candidates), "pandas": False},
        {"keys": list(candidates)},
        {},
    )
    for kwargs in history_iters:
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
        except TypeError:
            try:
                rows = [_coerce_config(r) for r in iterator[:limit]]  # type: ignore[index]
            except Exception:
                rows = []
        break

    for row in reversed(rows):
        for candidate in candidates:
            value = _coerce_to_float(_lookup_nested(row, candidate))
            if value is not None:
                return value, candidate
    return None, None


def _metric_candidates(metric: str) -> List[str]:
    candidates = [metric]
    if "/" in metric:
        candidates.append(metric.replace("/", "."))
    if "." in metric:
        candidates.append(metric.replace(".", "/"))
    return list(dict.fromkeys(candidates))


def _extract_metric_from_run(run: Any, metric: str) -> Tuple[Optional[float], Optional[str]]:
    candidates = _metric_candidates(metric)

    summary = _extract_summary_payload(run)
    for candidate in candidates:
        value = _coerce_to_float(_lookup_nested(summary, candidate))
        if value is not None and math.isfinite(value):
            return float(value), f"summary[{candidate}]"

    config = _coerce_config(getattr(run, "config", {}))
    for candidate in candidates:
        value = _coerce_to_float(_lookup_nested(config, candidate))
        if value is not None and math.isfinite(value):
            return float(value), f"config[{candidate}]"

    value, candidate = _extract_from_history(run, candidates)
    if value is not None and math.isfinite(value):
        return float(value), f"history[{candidate}]"

    return None, None


def _collect_top_runs(api: wandb.Api, sweep_path: str, metric: str, maximize: bool, k: int,
                      max_attempts: int, retry_delay: float) -> Tuple[List[Tuple[Any, float, str]], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"attempts": 0, "total_runs": 0, "missing": [], "method_counts": Counter()}

    for attempt in range(1, max_attempts + 1):
        diagnostics["attempts"] = attempt
        try:
            sweep = api.sweep(sweep_path)
        except Exception as exc:
            diagnostics["error"] = str(exc)
            print(f"[recheck] attempt {attempt}/{max_attempts}: failed to load sweep {sweep_path}: {exc}", flush=True)
            if attempt < max_attempts:
                time.sleep(retry_delay)
                continue
            break

        runs = list(sweep.runs)
        diagnostics["total_runs"] = len(runs)
        diagnostics["missing"] = []
        diagnostics["method_counts"] = Counter()

        ranked: List[Tuple[Any, float, str]] = []
        for run in runs:
            config = _coerce_config(getattr(run, "config", {}))
            method = str(config.get("training_method", "unknown")).lower()
            diagnostics["method_counts"][method] += 1

            value, source = _extract_metric_from_run(run, metric)
            if value is None:
                diagnostics["missing"].append(run.id)
                continue
            ranked.append((run, value, source or ""))

        if ranked:
            ranked.sort(key=lambda item: item[1], reverse=maximize)
            limit = max(1, k)
            return ranked[:limit], diagnostics

        if len(runs) == 0:
            print(f"[recheck] attempt {attempt}/{max_attempts}: sweep returned zero runs; retrying", flush=True)
        else:
            print(
                f"[recheck] attempt {attempt}/{max_attempts}: no runs with metric '{metric}' yet; missing from {len(diagnostics['missing'])} runs",
                flush=True,
            )

        if attempt < max_attempts:
            time.sleep(retry_delay)

    return [], diagnostics

def run_once(mm, program, subcmd, cfg: Dict[str, Any], seed: int,
             unlabeled: str, labeled: str, log_dir: str,
             project: str, group: str) -> int:
    import os, shlex, subprocess

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if "WANDB_PROJECT" not in env and project:
        env["WANDB_PROJECT"] = project
    if "WANDB_RUN_GROUP" not in env and group:
        env["WANDB_RUN_GROUP"] = group
    env.setdefault("WANDB_JOB_TYPE", "recheck")

    # Base command
    args = [
        mm, "run", "-n", "mjepa", "env", "PYTHONUNBUFFERED=1",
        "python", "-u", program, subcmd,
        "--unlabeled-dir", unlabeled,
        "--labeled-dir",   labeled,
    ]

    # Force training method explicitly (default to jepa)
    method = str(cfg.get("training_method", "jepa")).lower()
    args += ["--training-method", method]

    # sweep-run supported flags (hyphen form)
    ALLOWED = {
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
        "mask-ratio", "ema-decay",      # JEPA-only
        "temperature",                  # Contrastive-only
        "cache-dir", "use-wandb", "wandb-project", "wandb-tags",
        "target-pretrain-samples",
    }
    MAP = {
        "pretrain_bs":  "pretrain-batch-size",
        "finetune_bs":  "finetune-batch-size",
        "lr":           "learning-rate",
        "label_col":    "label-col",
        "gnn_type":     "gnn-type",
        "add_3d":       "add-3d",
        "contiguous":   "contiguity",   # CLI expects --contiguity
        # long names fall through
    }
    DROP = {
        "augmentations", "pair_id", "pair_key", "_wandb", "wandb_version",
        "seed", "name", "id", "group", "use_wandb",
    }

    forwarded = []  # for debug print below

    for k, v in cfg.items():
        if k in DROP or v is None:
            continue
        if isinstance(v, (dict, list, tuple)):
            continue
        if k == "training_method":
            continue

        cli_key = MAP.get(k, k).replace("_", "-")
        if cli_key not in ALLOWED:
            continue

        flag = f"--{cli_key}"
        if isinstance(v, bool):
            args += [flag, "1" if v else "0"]
        else:
            args += [flag, str(v)]
        forwarded.append((flag, v))

    # Always enable wandb logging for rechecks so metrics propagate reliably.
    args += ["--use-wandb", "1"]
    forwarded.append(("--use-wandb", 1))

    # Seed per recheck
    args += ["--seed", str(seed)]

    # Make sure log dir exists; write the command at the top of the log
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"recheck_{method}_seed{seed}.log")

    with open(log, "w", encoding="utf-8") as f:
        if forwarded:
            f.write("[recheck] forwarded flags: " + repr(forwarded) + "\n")
        p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env)

    print(
        f"[recheck][seed {seed}] started PID {p.pid}; streaming logs to {log}",
        flush=True,
    )

    start = time.time()
    next_ping = start + 60.0
    while True:
        rc = p.poll()
        if rc is not None:
            return rc

        now = time.time()
        if now >= next_ping:
            mins = int((now - start) // 60)
            if mins == 0:
                mins = 1  # ensure we report at least 1 minute
            print(
                f"[recheck][seed {seed}] still running after {mins} min; tail -f {log} for live output",
                flush=True,
            )
            next_ping = now + 60.0

        time.sleep(5.0)

def ci95(xs: List[float]) -> Tuple[float, float]:
    bs = [np.mean(np.random.choice(xs, size=len(xs), replace=True)) for _ in range(4000)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--project", default=os.getenv("WANDB_PROJECT"))
    ap.add_argument("--group",   default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--metric", default=os.getenv("PHASE2_METRIC","val_rmse"))
    ap.add_argument("--direction", choices=["min","max"], default="min")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--extra_seeds", type=int, default=3)
    ap.add_argument("--program", required=True)
    ap.add_argument("--subcmd", default="sweep-run")
    ap.add_argument("--unlabeled-dir", "--unlabeled_dir", "--unlabeled",
                dest="unlabeled_dir", required=True)
    ap.add_argument("--labeled-dir", "--labeled_dir", "--labeled",
                    dest="labeled_dir", required=True)
    ap.add_argument("--mm", default=os.environ.get("MMBIN","micromamba"))
    ap.add_argument("--log_dir", default=os.environ.get("LOG_DIR","./logs"))
    # Pick a writable default at runtime (cwd or temp) if GRID_DIR is missing or not writable
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="Fail if fewer than topk runs expose the metric after retries")
    args = ap.parse_args()

    print("[recheck] args parsed:", flush=True)
    print("  sweep      =", args.sweep, flush=True)
    print("  project    =", args.project, flush=True)
    print("  group      =", args.group, flush=True)
    print("  metric     =", args.metric, flush=True)
    print("  direction  =", args.direction, flush=True)
    print("  topk       =", args.topk, flush=True)
    print("  extra_seeds=", args.extra_seeds, flush=True)
    print("  program    =", args.program, flush=True)
    print("  subcmd     =", args.subcmd, flush=True)
    print("  unlabeled  =", args.unlabeled_dir, flush=True)
    print("  labeled    =", args.labeled_dir, flush=True)
    print("  mm         =", args.mm, flush=True)
    print("  log_dir    =", args.log_dir, flush=True)
    print("  out        =", args.out, flush=True)
    print("  strict     =", args.strict, flush=True)

    # --- resolve a safe, writable output path and ensure parent exists ---
    def _safe_out(user_out: str | None) -> str:
        if user_out:
            out_path = pathlib.Path(user_out)
        else:
            base = os.environ.get("GRID_DIR") or os.getcwd()
            out_path = pathlib.Path(base) / "recheck_summary.json"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # also check writability
            testfile = out_path.parent / ".writetest"
            with open(testfile, "w", encoding="utf-8") as tf:
                tf.write("ok")
            testfile.unlink(missing_ok=True)
        except Exception:
            # fall back to temp dir on permission/creation error
            out_path = pathlib.Path(tempfile.gettempdir()) / (out_path.name or "recheck_summary.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        return str(out_path)

    args.out = _safe_out(args.out)

    # --- resolve a safe, writable log_dir too ---
    def _safe_dir(d: str | None, fallback_name: str = "recheck_logs") -> str:
        base = pathlib.Path(d or "./logs")
        try:
            base.mkdir(parents=True, exist_ok=True)
            test = base / ".writetest"
            with open(test, "w", encoding="utf-8") as tf:
                tf.write("ok")
            test.unlink(missing_ok=True)
            return str(base)
        except Exception:
            tmp = pathlib.Path(tempfile.gettempdir()) / fallback_name
            tmp.mkdir(parents=True, exist_ok=True)
            return str(tmp)

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

    max_attempts = max(1, _env_int("RECHECK_MAX_ATTEMPTS", 5))
    retry_delay = max(0.0, _env_float("RECHECK_RETRY_DELAY", 15.0))

    print("  max_attempts=", max_attempts, flush=True)
    print("  retry_delay =", retry_delay, flush=True)

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
    maximize = (args.direction == "max")
    top_info, diagnostics = _collect_top_runs(api, args.sweep, args.metric, maximize, args.topk, max_attempts, retry_delay)

    if not top_info:
        method_counts = diagnostics.get("method_counts", Counter())
        total_runs = diagnostics.get("total_runs", 0)
        missing = diagnostics.get("missing", [])
        if total_runs == 0:
            print(
                f"[recheck] sweep {args.sweep} returned zero runs; writing empty summary to {args.out}",
                flush=True,
            )
            _write_summary([], diagnostics)
            return
        print(
            f"[recheck][fatal] unable to locate metric '{args.metric}' in sweep {args.sweep} "
            f"after {diagnostics.get('attempts', max_attempts)} attempt(s); runs={total_runs} missing_metrics={len(missing)}",
            flush=True,
        )
        if method_counts:
            print(
                "[recheck] sweep method counts: " + ", ".join(f"{m or 'unknown'}={c}" for m, c in sorted(method_counts.items())),
                flush=True,
            )
        if missing:
            sample = ", ".join(missing[:5])
            more = "" if len(missing) <= 5 else f" … (+{len(missing)-5} more)"
            print(f"[recheck] runs missing metric: {sample}{more}", flush=True)
        if diagnostics.get("error"):
            print(f"[recheck] last API error: {diagnostics['error']}", flush=True)
        sys.exit(2)

    if args.strict and len(top_info) < max(1, args.topk):
        print(
            f"[recheck][fatal] strict mode requires {args.topk} runs with metric '{args.metric}' "
            f"but only found {len(top_info)}",
            flush=True,
        )
        sys.exit(2)

    top = [item[0] for item in top_info]

    print(
        f"[recheck] picked top {len(top)} from sweep: {[r.id for r in top]} (metric source(s): "
        + ", ".join(f"{r.id}:{src}" for r, _, src in top_info)
        + ")",
        flush=True,
    )

    missing = diagnostics.get("missing", [])
    if missing:
        print(
            f"[recheck] ignored {len(missing)} runs without metric '{args.metric}'",
            flush=True,
        )

    method_counts = diagnostics.get("method_counts", Counter())
    if method_counts:
        print(
            "[recheck] sweep method counts: "
            + ", ".join(f"{method or 'unknown'}={count}" for method, count in sorted(method_counts.items())),
            flush=True,
        )

    def _kv(d, keys=("training_method","gnn_type","hidden_dim","num_layers","contiguity","mask_ratio","ema_decay","temperature","learning_rate")):
        out = {}
        for k in keys:
            v = d.get(k)
            if v is not None:
                out[k] = v
        return out

    print("[recheck] top configs (key fields):", flush=True)
    for r in top:
        print(" ", r.id, _kv(r.config), flush=True)

    # Extract the raw config dict for each top run
    tops: List[Dict[str, Any]] = []
    for r in top:
        cfg = {}
        raw_cfg = _coerce_config(getattr(r, "config", {}))
        for k, v in raw_cfg.items():
            if k.startswith("_"):   # skip internal
                continue
            cfg[k] = _unwrap_config_value(v)
        # Drop seed so we can override
        cfg.pop("seed", None)
        tops.append(cfg)

    #os.makedirs(args.log_dir, exist_ok=True)

    results = []
    for i, cfg in enumerate(tops):
        seeds = [1000 + i*10 + s for s in range(args.extra_seeds)]
        vals = []
        print(f"[recheck] launching seed {seeds} for config index {i}", flush=True)
        for s in seeds:
            rc = run_once(args.mm, args.program, args.subcmd, cfg, s,
              args.unlabeled_dir, args.labeled_dir, args.log_dir,
              args.project, args.group)
            print(
                f"[recheck][seed {s}] finished with exit code {rc}",
                flush=True,
            )
            if rc != 0:
                # show last ~120 lines of the log for quick diagnosis
                log_path = os.path.join(args.log_dir, f"recheck_{cfg.get('training_method','jepa')}_seed{s}.log")
                try:
                    with open(log_path, "r", encoding="utf-8") as lf:
                        lines = lf.readlines()[-120:]
                except Exception as _e:
                    print(f"[recheck][seed {s}] could not read log {log_path}: {_e}", flush=True)
                raise RuntimeError(f"recheck run failed with exit code {rc} (seed {s})")
            # after the run, we could read the metric from W&B; here we assume train_jepa.py writes best val to a file or W&B;
            # to keep this self-contained, we re-fetch the last matching run (same group + tags would be best).
            # Minimal: refetch on metric for runs in the same project/group with matching cfg+seed.
            time.sleep(1)

        # Fetch metrics again
        # fetch from project+group (includes recheck runs; sweeps would miss them)
        entity = os.getenv("WANDB_ENTITY")
        proj_path = f"{entity}/{args.project}" if entity and args.project else None
        runs = []
        if proj_path:
            runs = list(api.runs(proj_path, filters={"group": args.group}))
        for r in runs:
            run_cfg = _coerce_config(getattr(r, "config", {}))
            ok = True
            for k, v in cfg.items():
                if _unwrap_config_value(run_cfg.get(k)) != v:
                    ok = False
                    break
            run_seed = _unwrap_config_value(run_cfg.get("seed"))
            if ok and run_seed in seeds:
                mv, _source = _extract_metric_from_run(r, args.metric)
                if mv is not None:
                    vals.append(float(mv))

        if vals:
            mu = float(np.mean(vals)); lo, hi = ci95(vals)
            results.append({"index": i, "mean": mu, "ci95": [lo,hi], "n": len(vals), "config": cfg})
        else:
            results.append({"index": i, "mean": None, "ci95": [None,None], "n": 0, "config": cfg})

    success_results = [r for r in results if r.get("mean") is not None]

    _write_summary(results, diagnostics)

    if not success_results:
        print(
            f"[recheck][fatal] no metrics collected for metric '{args.metric}' in sweep {args.sweep}; "
            f"see {args.out} for details",
            flush=True,
        )
        sys.exit(3)

    # --- Normalize Phase-2 winner to Phase-1 schema and automically back up best_grid_config.json from phase1 ---
    out_dir = os.path.dirname(args.out)

    if args.direction == "min":
        best = min(success_results, key=lambda r: r["mean"])
    else:
        best = max(success_results, key=lambda r: r["mean"])

    winner_cfg = (best or {}).get("config") or {}
    if not isinstance(winner_cfg, dict) or not winner_cfg:
        print(
            f"[recheck][fatal] best configuration for sweep {args.sweep} produced an empty payload; "
            f"inspect {args.out}",
            flush=True,
        )
        sys.exit(3)

    # Atomically write Phase-2 winner in Phase-1 schema: {"config": {...}}
    p1_best   = os.path.join(out_dir, "best_grid_config.json")
    p1_backup = os.path.join(out_dir, "best_grid_config.phase1.json")
    tmp_write = os.path.join(out_dir, ".best_grid_config.tmp")

    try:
        if os.path.isfile(p1_best) and not os.path.isfile(p1_backup):
            import shutil
            shutil.copy2(p1_best, p1_backup)
            print(f"[recheck] backed up Phase-1 best → {p1_backup}", flush=True)
    except Exception as e:
        print(f"[recheck][warn] backup skipped: {e}", flush=True)

    with open(tmp_write, "w", encoding="utf-8") as f:
        json.dump({"config": winner_cfg}, f, indent=2)
    os.replace(tmp_write, p1_best)
    print(f"[recheck] wrote Phase-2 winner as Phase-1 best → {p1_best}", flush=True)



if __name__ == "__main__":
    main()
