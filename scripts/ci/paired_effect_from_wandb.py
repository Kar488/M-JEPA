from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from itertools import islice
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np, wandb, os, argparse, json, sys


def _coerce_config(config: Any) -> Dict[str, Any]:
    """Normalize a run config into a plain dictionary."""

    if isinstance(config, Mapping):
        return dict(config)

    to_dict_attr = None
    try:
        to_dict_attr = config.to_dict  # type: ignore[attr-defined]
    except (AttributeError, KeyError):
        to_dict_attr = None

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

    return {}


def _lookup_nested(mapping: Mapping[str, Any], key: str) -> Any:
    """Return ``mapping[key]`` while supporting ``foo/bar`` or ``foo.bar`` access."""

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


def _extract_from_history(run: Any, candidates: Iterable[str], limit: int = 512) -> Tuple[Optional[float], Optional[str]]:
    """Best-effort extraction of metric values from a W&B run history.

    Some runs fail to promote validation metrics into the W&B summary – especially
    when they abort prematurely.  Instead of bailing out, attempt to read the
    recorded history events and use the most recent non-NaN scalar.  The helper is
    defensive because ``run.history`` has a fairly loose contract and user stubs
    in the unit tests provide simplified implementations.
    """

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
            # ``history`` may return a list already.
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


MetricStore = Tuple[
    Dict[str, Dict[str, List[float]]],
    Dict[str, Dict[Any, Dict[str, List[float]]]],
    Dict[str, Optional[str]],
]


METRIC_INFO = {
    "rmse": {
        "label": "RMSE",
        "direction": "min",
        "candidates": [
            "val_rmse",
            "val_rmse_mean",
            "rmse",
            "rmse_mean",
            "val_root_mean_squared_error",
        ],
    },
    "roc_auc": {
        "label": "ROC-AUC",
        "direction": "max",
        "candidates": [
            "val_roc_auc",
            "val_roc_auc_mean",
            "roc_auc",
            "roc_auc_mean",
            "val_auc",
            "val_auroc",
            "auroc",
        ],
    },
    "r2": {
        "label": "R²",
        "direction": "max",
        "candidates": [
            "val_r2",
            "val_r2_mean",
            "r2",
            "r2_mean",
        ],
    },
    "brier": {
        "label": "Brier score",
        "direction": "min",
        "candidates": [
            "val_brier",
            "val_brier_mean",
            "brier",
            "brier_mean",
            "val_brier_score",
        ],
    },
}


TASK_PRIMARY = {"regression": "rmse", "classification": "roc_auc"}
TASK_TIEBREAKER = {"regression": "r2", "classification": "brier"}


def _unwrap_config_value(value: Any) -> Any:
    """Best-effort extraction of a primitive value from sweep/config payloads."""

    # W&B sweeps frequently wrap the actual parameter value in small dictionaries
    # such as {'value': 'jepa'} or {'value': 5, '_type': 'int'}.  When the
    # payload exposes an unambiguous candidate ("value", "default", etc.) keep
    # peeling the onion until we reach the underlying primitive.  We fall back to
    # the original input if no recognised wrapper is present so nested
    # dictionaries (e.g. structured configs) remain intact.
    if isinstance(value, Mapping):
        for key in ("value", "default", "actual", "_value"):
            if key in value:
                inner = _unwrap_config_value(value[key])
                if inner is not None:
                    return inner
        if len(value) == 1:
            # Occasionally sweep configs store the payload under an opaque single
            # key (e.g. {'wandb': {'value': ...}}).  Peeking at the only value is
            # safe here because we only reach this branch when the mapping does
            # not expose the standard wrappers checked above.
            inner = next(iter(value.values()))
            return _unwrap_config_value(inner)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_config_value(value[0])
    return value


def _coerce_to_float(value):
    if value is None:
        return None

    # Common numeric scalars from numpy / wandb already satisfy this branch.
    if isinstance(value, (int, float)):
        return float(value)

    # W&B may wrap summary scalars in lightweight containers (e.g. {'value': x})
    # or provide per-statistic aggregates.  Attempt to unwrap a representative
    # value before giving up.
    if isinstance(value, Mapping):
        for key in ("value", "latest", "last", "mean", "median", "max", "min", "best"):
            if key in value:
                coerced = _coerce_to_float(value[key])
                if coerced is not None:
                    return coerced
        # Fall back to inspecting the mapping values in iteration order.  This
        # mirrors the behaviour of wandb's RunSummary where the numeric payload
        # can live under an opaque key (e.g. {'_type': 'summary', 'numeric': x}).
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


def _ensure_metric_store(metric_key: str, store: MetricStore) -> None:
    pair_vals, pair_seed, metric_names = store
    if metric_key not in pair_vals:
        pair_vals[metric_key] = defaultdict(lambda: defaultdict(list))
    if metric_key not in pair_seed:
        pair_seed[metric_key] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    metric_names.setdefault(metric_key, None)


def _normalize_task(value: Any) -> Optional[str]:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered.startswith("regress") or "continuous" in lowered or "float" in lowered:
            return "regression"
        if lowered.startswith("class") or "categor" in lowered or "binary" in lowered:
            return "classification"
    return None


def _infer_task_from_config(config: Dict[str, Any]) -> Optional[str]:
    if not isinstance(config, dict):
        return None

    for key in ("prediction_target_type", "target_type", "label_type"):
        task = _normalize_task(_unwrap_config_value(config.get(key)))
        if task:
            return task

    target_cfg = _unwrap_config_value(config.get("prediction_target"))
    if isinstance(target_cfg, dict):
        task = _normalize_task(
            _unwrap_config_value(target_cfg.get("type") or target_cfg.get("dtype"))
        )
        if task:
            return task
    elif isinstance(target_cfg, str):
        task = _normalize_task(target_cfg)
        if task:
            return task

    task = _normalize_task(_unwrap_config_value(config.get("task_type")))
    if task:
        return task

    label_values = _unwrap_config_value(config.get("label_values")) or _unwrap_config_value(
        config.get("classes")
    )
    if isinstance(label_values, (list, tuple, set)) and label_values:
        return "classification"

    num_classes = _unwrap_config_value(config.get("num_classes"))
    if isinstance(num_classes, int) and num_classes > 1:
        return "classification"

    return None


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    data = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not data:
        return None
    return float(np.mean(data))


def _format_metric(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def _gather_metric_value(
    metric_key: str,
    contrib: Tuple[str, Optional[str], Optional[Any]],
    store: MetricStore,
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    pair_vals, pair_seed, _ = store
    kind, pid, seed = contrib

    if metric_key not in pair_vals:
        return None

    if kind == "seed":
        seeds = pair_seed.get(metric_key, {}).get(pid, {})
        mm = seeds.get(seed)
        if not mm or "jepa" not in mm or "contrastive" not in mm:
            return None
        jv = _safe_mean(mm["jepa"])
        cv = _safe_mean(mm["contrastive"])
        return jv, cv

    if kind == "pair":
        methods = pair_vals.get(metric_key, {}).get(pid, {})
        jv = _safe_mean(methods.get("jepa", ()))
        cv = _safe_mean(methods.get("contrastive", ()))
        if jv is None or cv is None:
            return None
        return jv, cv

    if kind == "global":
        methods = pair_vals.get(metric_key, {})
        j_all, c_all = [], []
        for method_vals in methods.values():
            j_all.extend(method_vals.get("jepa", ()))
            c_all.extend(method_vals.get("contrastive", ()))
        jv = _safe_mean(j_all)
        cv = _safe_mean(c_all)
        if jv is None or cv is None:
            return None
        return jv, cv

    return None


def _aggregate_metric(
    metric_key: str,
    store: MetricStore,
    aggregate: str,
) -> Optional[Tuple[List[float], Dict[str, List[float]], int, List[Tuple[str, Optional[str], Optional[Any]]]]]:
    pair_vals, pair_seed, _ = store
    if metric_key not in pair_vals:
        return None

    deltas: List[float] = []
    per_method: Dict[str, List[float]] = {"jepa": [], "contrastive": []}
    contributions: List[Tuple[str, Optional[str], Optional[Any]]] = []
    used_pairs = 0

    if aggregate == "pair-seed":
        for pid, seeds in pair_seed.get(metric_key, {}).items():
            common = [sd for sd, mm in seeds.items() if "jepa" in mm and "contrastive" in mm]
            pair_used = False
            for sd in common:
                jv = _safe_mean(seeds[sd]["jepa"])
                cv = _safe_mean(seeds[sd]["contrastive"])
                if jv is None or cv is None:
                    continue
                deltas.append(cv - jv)
                per_method["jepa"].append(jv)
                per_method["contrastive"].append(cv)
                contributions.append(("seed", pid, sd))
                pair_used = True
            if pair_used:
                used_pairs += 1
            else:
                methods = pair_vals.get(metric_key, {}).get(pid, {})
                if (
                    "jepa" in methods
                    and "contrastive" in methods
                    and methods["jepa"]
                    and methods["contrastive"]
                ):
                    jv = _safe_mean(methods["jepa"])
                    cv = _safe_mean(methods["contrastive"])
                    if jv is None or cv is None:
                        continue
                    deltas.append(cv - jv)
                    per_method["jepa"].append(jv)
                    per_method["contrastive"].append(cv)
                    contributions.append(("pair", pid, None))
                    used_pairs += 1

        if not deltas and pair_vals.get(metric_key):
            global_methods = {"jepa": [], "contrastive": []}
            for methods in pair_vals[metric_key].values():
                global_methods["jepa"].extend(methods.get("jepa", ()))
                global_methods["contrastive"].extend(methods.get("contrastive", ()))
            jv = _safe_mean(global_methods["jepa"])
            cv = _safe_mean(global_methods["contrastive"])
            if jv is not None and cv is not None:
                deltas = [cv - jv]
                per_method["jepa"] = [jv]
                per_method["contrastive"] = [cv]
                contributions = [("global", None, None)]
                used_pairs = 0

    else:
        direction = METRIC_INFO[metric_key]["direction"]
        for pid, methods in pair_vals[metric_key].items():
            if (
                "jepa" not in methods
                or "contrastive" not in methods
                or not methods["jepa"]
                or not methods["contrastive"]
            ):
                continue

            j_clean = [
                float(v) for v in methods["jepa"] if v is not None and not math.isnan(float(v))
            ]
            c_clean = [
                float(v)
                for v in methods["contrastive"]
                if v is not None and not math.isnan(float(v))
            ]
            if not j_clean or not c_clean:
                continue

            if aggregate == "mean":
                jv = float(np.mean(j_clean))
                cv = float(np.mean(c_clean))
            elif aggregate == "median":
                jv = float(np.median(j_clean))
                cv = float(np.median(c_clean))
            else:  # best
                chooser = max if direction == "max" else min
                jv = chooser(j_clean)
                cv = chooser(c_clean)

            deltas.append(cv - jv)
            per_method["jepa"].append(float(jv))
            per_method["contrastive"].append(float(cv))
            contributions.append(("pair", pid, None))
            used_pairs += 1

    if not deltas:
        return None

    return deltas, per_method, used_pairs, contributions
    
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare JEPA and contrastive runs by automatically selecting the "
            "appropriate validation metric from recorded W&B summaries."
        )
    )
    ap.add_argument("--project", default=os.getenv("WANDB_PROJECT","mjepa"))
    ap.add_argument("--group",   default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--out", default=os.path.join(os.getenv("GRID_DIR", "."), "paired_effect.json"))
    ap.add_argument("--seed", type=int, default=None, help="Seed for bootstrap reproducibility")
    ap.add_argument("--aggregate", choices=["pair-seed","mean","median","best"], default="pair-seed",
        help="How to aggregate multiple runs per (pair_id, method). "
        "'pair-seed' = compute deltas only on shared seeds; "
        "fallback to mean per method if no shared seeds.")
    ap.add_argument("--strict", action="store_true",
        help="Exit non-zero when no runs or no matched pairs are found.")
    ap.add_argument("--min_pretrain_epochs", type=int, default=None,
        help="Minimum pretraining epochs (config: pretrain_epochs, units: epochs) required to include a run.")
    ap.add_argument("--min_finetune_epochs", type=int, default=None,
        help="Minimum fine-tuning epochs (config: finetune_epochs, units: epochs) required to include a run.")
    ap.add_argument("--min_pretrain_batches", type=int, default=None,
        help="Minimum pretraining batches (config: max_pretrain_batches, units: batches) required to include a run.")
    ap.add_argument("--tie_tol", type=float, default=1e-2,
        help="Absolute tolerance for treating the primary metric difference as a tie.")
    args = ap.parse_args()

    api = wandb.Api()
    filters = {"group": args.group} if args.group else None
    runs = api.runs(f"{os.getenv('WANDB_ENTITY')}/{args.project}", filters=filters)
    if not runs:
        # Do not write output when empty; only fail hard if --strict
        if args.strict:
            print("No runs found.", flush=True)
            sys.exit(2)
        return

    # Collect values per metric.
    # by_metric_pair_vals[metric][pair_id][method] -> list[float]
    # by_metric_pair_seed[metric][pair_id][seed][method] -> list[float]
    by_metric_pair_vals: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    by_metric_pair_seed: Dict[str, Dict[str, Dict[Any, Dict[str, List[float]]]]] = {}
    metric_names: Dict[str, Optional[str]] = {}

    metric_store: MetricStore = (by_metric_pair_vals, by_metric_pair_seed, metric_names)

    inferred_task: Optional[str] = None

    total_comparable_runs = 0
    skipped_by_threshold = 0
    missing_threshold_counts: Dict[str, int] = defaultdict(int)
    failed_threshold_counts: Dict[str, int] = defaultdict(int)
    threshold_minimums = {
        "pretrain_epochs": args.min_pretrain_epochs,
        "finetune_epochs": args.min_finetune_epochs,
        "max_pretrain_batches": args.min_pretrain_batches,
    }

    for r in runs:
        run_config = _coerce_config(getattr(r, "config", {}))
        summary = _coerce_config(getattr(r, "summary", {}))
        mid_raw = _unwrap_config_value(run_config.get("training_method"))
        pid_raw = _unwrap_config_value(run_config.get("pair_id"))

        if not pid_raw:
            pid_raw = _unwrap_config_value(summary.get("pair_id"))

        mid = mid_raw.strip().lower() if isinstance(mid_raw, str) else mid_raw
        if isinstance(mid, Mapping):
            mid = _unwrap_config_value(mid)
            mid = mid.strip().lower() if isinstance(mid, str) else mid

        if isinstance(pid_raw, Mapping):
            pid_raw = _unwrap_config_value(pid_raw)

        if not pid_raw or mid not in ("jepa", "contrastive"):
            continue

        total_comparable_runs += 1
        if inferred_task is None:
            inferred_task = _infer_task_from_config(run_config)

        thresholds = (
            ("pretrain_epochs", threshold_minimums["pretrain_epochs"]),
            ("finetune_epochs", threshold_minimums["finetune_epochs"]),
            ("max_pretrain_batches", threshold_minimums["max_pretrain_batches"]),
        )
        skip = False
        for key, minimum in thresholds:
            if minimum is None:
                continue
            conf_val = _coerce_to_float(run_config.get(key))
            # Phase-1 sweeps often terminate early while phase-2 sweeps rely on mature metrics;
            # filtering prevents these under-trained runs from biasing the phase-1/phase-2 workflow.
            if conf_val is None:
                missing_threshold_counts[key] += 1
                continue
            if conf_val < minimum:
                failed_threshold_counts[key] += 1
                skip = True
                break
        if skip:
            skipped_by_threshold += 1
            continue
        
        pid = str(pid_raw)

        # Gather metrics for all known candidates.
        metrics_recorded: List[Tuple[str, float]] = []
        for metric_key, info in METRIC_INFO.items():
            metric_val = None
            metric_name = None
            for candidate in info["candidates"]:
                if candidate in summary:
                    metric_val = _coerce_to_float(summary.get(candidate))
                else:
                    metric_val = _coerce_to_float(_lookup_nested(summary, candidate))
                if metric_val is not None:
                    metric_name = candidate
                    break
            if metric_val is None:
                metric_val, metric_name = _extract_from_history(r, info["candidates"])
            if metric_val is None:
                continue

            _ensure_metric_store(metric_key, metric_store)
            metric_names[metric_key] = metric_names.get(metric_key) or metric_name
            by_metric_pair_vals[metric_key].setdefault(pid, defaultdict(list))[mid].append(metric_val)
            metrics_recorded.append((metric_key, metric_val))

        # capture seed if available
        seed = _unwrap_config_value(run_config.get("seed", None))
        if seed is not None:
            try:
                seed = int(seed)
            except Exception:
                pass
        for metric_key, metric_val in metrics_recorded:
            if seed is None:
                continue
            _ensure_metric_store(metric_key, metric_store)
            by_metric_pair_seed[metric_key][pid][seed][mid].append(metric_val)

    # infer task before reduction
    available_metrics = {key for key, vals in by_metric_pair_vals.items() if vals}
    if inferred_task is None:
        if "roc_auc" in available_metrics and "rmse" not in available_metrics:
            inferred_task = "classification"
        elif "rmse" in available_metrics:
            inferred_task = "regression"
        else:
            inferred_task = None

    # Determine task and select the primary metric.
    task = inferred_task
    task_resolution_reason = "run config"
    if task is None:
        task = "classification" if "roc_auc" in available_metrics else "regression"
        task_resolution_reason = "available metrics"

    primary_key = TASK_PRIMARY.get(task, "rmse")
    primary_resolution_reason = "task defaults"
    if primary_key not in available_metrics:
        fallback_primary = next((k for k in ("rmse", "roc_auc") if k in available_metrics), None)
        if fallback_primary:
            primary_key = fallback_primary
            primary_resolution_reason = "available metrics"
        else:
            primary_key = next(iter(available_metrics), None)
            if primary_key is not None:
                primary_resolution_reason = "first available metric"

    if primary_key is None:
        if args.strict:
            if total_comparable_runs and skipped_by_threshold == total_comparable_runs:
                reasons = []
                if failed_threshold_counts:
                    reasons.append(
                        "; ".join(
                            f"{key}<{threshold_minimums.get(key)} ({count} run(s))"
                            for key, count in failed_threshold_counts.items()
                            if threshold_minimums.get(key) is not None and count
                        )
                    )
                if missing_threshold_counts:
                    reasons.append(
                        ", ".join(
                            f"missing {key} ({count} run(s))"
                            for key, count in missing_threshold_counts.items()
                            if count
                        )
                    )
                detail = "; ".join(filter(None, reasons))
                if detail:
                    print(
                        "No metrics available for comparison after applying threshold filters: "
                        f"{detail}.",
                        flush=True,
                    )
                else:
                    print("No metrics available for comparison after applying threshold filters.", flush=True)
            else:
                print("No metrics available for comparison.", flush=True)
            sys.exit(2)
        return

    print(
        f"[paired-effect] using task={task} (source={task_resolution_reason})",
        flush=True,
    )

    if primary_key == "rmse":
        task = "regression"
    elif primary_key == "roc_auc":
        task = "classification"

    print(
        "[paired-effect] evaluating canonical metric={canonical} (source={source})".format(
            canonical=primary_key,
            source=primary_resolution_reason,
        ),
        flush=True,
    )

    aggregate_result = _aggregate_metric(primary_key, metric_store, args.aggregate)
    if aggregate_result is None:
        pair_vals, _, _ = metric_store
        global_methods = defaultdict(list)
        for methods in pair_vals.get(primary_key, {}).values():
            for method, values in methods.items():
                global_methods[method].extend(values)
        if global_methods.get("jepa") and global_methods.get("contrastive"):
            jv = _safe_mean(global_methods["jepa"])
            cv = _safe_mean(global_methods["contrastive"])
            if jv is not None and cv is not None:
                aggregate_result = (
                    [cv - jv],
                    {"jepa": [jv], "contrastive": [cv]},
                    0,
                    [("global", None, None)],
                )
                print(
                    "[paired-effect] falling back to global mean delta across methods",
                    flush=True,
                )

    if aggregate_result is None:
        if args.strict:
            print("No matched pairs found.", flush=True)
            sys.exit(2)
        return

    deltas, per_method_values, used_pairs, contributions = aggregate_result
    direction = METRIC_INFO[primary_key]["direction"]
    
    mu = float(np.mean(deltas))
    if args.seed is not None:
        np.random.seed(args.seed)
    bs = [np.mean(np.random.choice(deltas, size=len(deltas), replace=True)) for _ in range(5000)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    win = 100.0 * sum((d > 0) if direction == "max" else (d < 0) for d in deltas) / len(deltas)

    print(
        f"Pairs: {len(deltas)}  meanΔ(ctr-JEPA)={mu:.4f}  95%CI[{lo:.4f},{hi:.4f}]  win%={win:.1f}",
        flush=True,
    )

    metric_names_map = metric_store[2]
    primary_label = METRIC_INFO[primary_key]["label"]
    primary_metric_name = metric_names_map.get(primary_key) or METRIC_INFO[primary_key]["candidates"][0]
    primary_values_mean = {
        method: _safe_mean(values) for method, values in per_method_values.items()
    }

    primary_tie = abs(mu) <= args.tie_tol
    base_winner = "contrastive" if ((direction == "min" and mu < 0) or (direction == "max" and mu > 0)) else "jepa"
    winner = base_winner

    print(
        "[paired-effect] primary {label} ({name}): jepa={jepa} contrastive={ctr} "
        "tie_tol={tol:.4g} tie={tie}".format(
            label=primary_label,
            name=primary_metric_name,
            jepa=_format_metric(primary_values_mean.get("jepa")),
            ctr=_format_metric(primary_values_mean.get("contrastive")),
            tol=args.tie_tol,
            tie="yes" if primary_tie else "no",
        ),
        flush=True,
    )

    tie_metric_key = TASK_TIEBREAKER.get(task)
    tie_metric_used = False
    tie_metric_values = {"jepa": None, "contrastive": None}

    if primary_tie and tie_metric_key:
        tie_vals = {"jepa": [], "contrastive": []}
        for contrib in contributions:
            values = _gather_metric_value(tie_metric_key, contrib, metric_store)
            if not values:
                continue
            jv, cv = values
            if jv is not None:
                tie_vals["jepa"].append(jv)
            if cv is not None:
                tie_vals["contrastive"].append(cv)
        tie_metric_values = {
            "jepa": _safe_mean(tie_vals["jepa"]),
            "contrastive": _safe_mean(tie_vals["contrastive"]),
        }
        if tie_metric_values["jepa"] is not None and tie_metric_values["contrastive"] is not None:
            tie_metric_used = True
            if tie_metric_key == "r2":
                if tie_metric_values["contrastive"] > tie_metric_values["jepa"]:
                    winner = "contrastive"
                else:
                    winner = "jepa"
            else:  # brier (lower is better)
                if tie_metric_values["contrastive"] < tie_metric_values["jepa"]:
                    winner = "contrastive"
                else:
                    winner = "jepa"

            tie_label = METRIC_INFO[tie_metric_key]["label"]
            tie_name = metric_names_map.get(tie_metric_key) or METRIC_INFO[tie_metric_key]["candidates"][0]
            print(
                "[paired-effect] tie-breaker {label} ({name}): jepa={jepa} contrastive={ctr}".format(
                    label=tie_label,
                    name=tie_name,
                    jepa=_format_metric(tie_metric_values["jepa"]),
                    ctr=_format_metric(tie_metric_values["contrastive"]),
                ),
                flush=True,
            )
        else:
            print(
                "[paired-effect] tie-breaker {label} unavailable; retaining primary winner".format(
                    label=METRIC_INFO[tie_metric_key]["label"]
                ),
                flush=True,
            )

    decision_source = (
        f"tie-breaker({tie_metric_key})"
        if tie_metric_used
        else ("primary" if not primary_tie else "primary (tie tolerance)")
    )
    print(
        "[paired-effect] selected winner={winner} using {source}".format(
            winner=winner,
            source=decision_source,
        ),
        flush=True,
    )

    # machine-readable artifact
    primary_metric_display_name = primary_metric_name

    payload = {
        "metric": primary_metric_display_name,
        "direction": direction,
        "pairs": len(deltas), "mean_delta_contrastive_minus_jepa": mu,
        "ci95": [lo, hi], "win_pct_contrastive_over_jepa": win,
        "winner": winner,
        "task": task,
        "pairs_used": used_pairs,
        "aggregate": args.aggregate,
        "decision_source": decision_source,
        "tie_breaker_used": tie_metric_used,
    }
    payload["primary_metric"] = {
        "canonical": primary_key,
        "name": primary_metric_name,
        "label": primary_label,
        "jepa": primary_values_mean.get("jepa"),
        "contrastive": primary_values_mean.get("contrastive"),
        "tolerance": args.tie_tol,
        "tied": primary_tie,
    }

    if tie_metric_key:
        tie_info = {
            "canonical": tie_metric_key,
            "name": metric_names_map.get(tie_metric_key),
            "label": METRIC_INFO[tie_metric_key]["label"],
            "jepa": tie_metric_values["jepa"],
            "contrastive": tie_metric_values["contrastive"],
            "used": tie_metric_used,
        }
        payload["tiebreaker_metric"] = tie_info
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # grep-friendly marker line:
    print(
        f"::winner::{winner} ::task::{task} ::metric::{primary_metric_display_name} ::direction::{direction}"
    )
if __name__ == "__main__":
    main()

