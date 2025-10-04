#!/usr/bin/env python3
"""
Step #2 helper:
- Read a W&B sweep (Phase-1 by default), pick best run with task-aware tie-breaks
- Write best config to $GRID_DIR/best_grid_config.json (for pretrain)
- Emit/overwrite $GRID_DIR/grid_sweep_phase2.yaml with the narrowed Bayes spec
  (derived from top-K Phase-1 runs)

Notes:
- Input sweep defaults to $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_SWEEP_ID1
  but you can pass --sweep_id to point at any sweep (e.g., SWEEP_ID2).
- This script does NOT create a sweep; Step #3 will create/use WANDB_SWEEP_ID2.
"""

import argparse, json, math, os
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.logging import maybe_init_wandb

import numpy as np
import wandb
import yaml


# ---------- env helpers ----------

def need_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env: {name}")
    return v


# ---------- metric helpers ----------

def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """Like ``getattr`` but tolerates wrappers that raise ``KeyError``."""

    try:
        return getattr(obj, name)
    except (AttributeError, KeyError):
        return default


def _coerce_mapping(payload: Any) -> Dict[str, Any]:
    """Best-effort conversion of W&B payloads into a plain dictionary."""

    if isinstance(payload, Mapping):
        return dict(payload)

    to_dict = _safe_getattr(payload, "to_dict")
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:
            converted = None
        if isinstance(converted, dict):
            return converted

    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    json_dict = _safe_getattr(payload, "_json_dict")
    if isinstance(json_dict, dict):
        return dict(json_dict)

    items = _safe_getattr(payload, "items")
    if callable(items):
        try:
            pairs = items()
        except Exception:
            pairs = None
        else:
            try:
                return dict(pairs)
            except Exception:
                try:
                    return dict(payload)
                except Exception:
                    pass

    try:
        return dict(payload)
    except Exception:
        return {}


def _merge_missing(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
    """Recursively merge ``src`` into ``dst`` without clobbering existing keys."""

    for key, value in src.items():
        if key in dst:
            if isinstance(value, Mapping) and isinstance(dst[key], Mapping):
                _merge_missing(dst[key], value)
            continue
        dst[key] = value


def _extract_summary(run: Any) -> Dict[str, Any]:
    """Collect the most complete summary mapping exposed by the run."""

    sources: List[Any] = []

    try:
        sources.append(getattr(run, "summary"))
    except Exception:
        sources.append(None)

    for attr in ("summary_metrics", "summaryMetrics"):
        try:
            sources.append(getattr(run, attr))
        except Exception:
            sources.append(None)

    attrs = getattr(run, "_attrs", None)
    if isinstance(attrs, Mapping):
        for key in ("summaryMetrics", "summary_metrics"):
            if key in attrs:
                sources.append(attrs.get(key))

    merged: Dict[str, Any] = {}

    for source in sources:
        summary = _coerce_mapping(source or {})
        if not summary:
            continue
        _merge_missing(merged, summary)

    return merged


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


def metric(run, name: str, default=None):
    summary = _extract_summary(run)
    v = _lookup_nested(summary, name)
    if v is None:
        config = _coerce_mapping(getattr(run, "config", {}) or {})
        v = _lookup_nested(config, name)
    return v if v is not None else default

def detect_task(runs) -> str:
    have_auc = any(metric(r, "val_auc") is not None for r in runs)
    have_rmse = any(metric(r, "val_rmse") is not None for r in runs)
    if have_auc and not have_rmse:
        return "classification"
    if have_rmse and not have_auc:
        return "regression"
    return "classification" if have_auc else "regression"


def choose_best(runs, primary: str, maximize: bool, tie_eps: float,
                tiebreakers: List[Tuple[str, bool]]):
    cand = [r for r in runs if metric(r, primary) is not None]
    if not cand:
        raise RuntimeError(f"No runs have primary metric '{primary}'")

    keyf = (lambda r: -float(metric(r, primary, -math.inf))) if maximize \
           else (lambda r:  float(metric(r, primary,  math.inf)))
    cand.sort(key=keyf)

    best = cand[0]
    best_val = float(metric(best, primary))
    if maximize:
        thresh = best_val * (1.0 - tie_eps)
        tied = [r for r in cand if float(metric(r, primary)) >= thresh]
    else:
        thresh = best_val * (1.0 + tie_eps)
        tied = [r for r in cand if float(metric(r, primary)) <= thresh]

    for tb_name, tb_max in tiebreakers:
        tb = [r for r in tied if metric(r, tb_name) is not None]
        if len(tb) <= 1:
            continue
        tb.sort(key=(lambda r: -float(metric(r, tb_name, -math.inf))) if tb_max
                        else (lambda r:  float(metric(r, tb_name,  math.inf))))
        best = tb[0]
        best_val = float(metric(best, primary))
        if maximize:
            thresh = best_val * (1.0 - tie_eps)
            tied = [r for r in tb if float(metric(r, primary)) >= thresh]
        else:
            thresh = best_val * (1.0 + tie_eps)
            tied = [r for r in tb if float(metric(r, primary)) <= thresh]
    return best


def collect_topk(runs, primary: str, maximize: bool, k: int):
    have = [r for r in runs if metric(r, primary) is not None]
    have.sort(key=(lambda r: -float(metric(r, primary, -math.inf))) if maximize
                    else (lambda r:  float(metric(r, primary,  math.inf))))
    return have[:max(1, k)]


# ---------- bounds helpers ----------

def _config_lookup(config: Any, key: str):
    """Return a configuration value matching ``key`` with hyphen/underscore fallbacks."""

    mapping = _coerce_mapping(config or {})
    candidates = [key]
    if "-" in key:
        candidates.append(key.replace("-", "_"))
    if "_" in key:
        candidates.append(key.replace("_", "-"))

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand in mapping:
            val = mapping[cand]
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            return val
    return None


def _unwrap_config_value(value: Any) -> Any:
    """Convert W&B config payloads into JSON-serialisable primitives."""

    if isinstance(value, Mapping):
        mapping = _coerce_mapping(value)
        if not mapping:
            return {}
        if set(mapping.keys()) == {"value"}:
            return _unwrap_config_value(mapping["value"])
        unwrapped: Dict[str, Any] = {}
        for key, val in mapping.items():
            if isinstance(key, str) and key.startswith("_wandb"):
                continue
            unwrapped[str(key)] = _unwrap_config_value(val)
        return unwrapped

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_unwrap_config_value(v) for v in value]

    if isinstance(value, np.generic):
        try:
            return value.item()
        except Exception:
            return _normalize_config_value(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", "replace")

    return _normalize_config_value(value)


def _sanitize_run_config(config: Any) -> Dict[str, Any]:
    mapping = _coerce_mapping(config or {})
    sanitized: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(key, str) and key.startswith("_wandb"):
            continue
        sanitized[str(key)] = _unwrap_config_value(value)
    return sanitized


REQUIRED_KEY_ALIASES: Dict[str, Sequence[str]] = {
    "training_method": ("training_method", "method"),
    "gnn_type": ("gnn_type", "gnn-type"),
    "hidden_dim": ("hidden_dim", "hidden-dim"),
    "num_layers": ("num_layers", "num-layers"),
}


def _enforce_required_keys(
    raw_config: Mapping[str, Any],
    sanitized: Dict[str, Any],
) -> Dict[str, Any]:
    best_cfg = dict(sanitized)
    missing: List[str] = []
    for canonical, aliases in REQUIRED_KEY_ALIASES.items():
        value = None
        for alias in aliases:
            value = _config_lookup(raw_config, alias)
            if value is None:
                value = _config_lookup(sanitized, alias)
            if value is not None:
                break
        if value is None:
            missing.append(canonical)
            continue
        best_cfg[canonical] = _normalize_config_value(value)

    if missing:
        raise RuntimeError(
            "Best run config missing required keys: " + ", ".join(sorted(missing))
        )

    if not best_cfg:
        raise RuntimeError("Best run config resolved to an empty payload")

    return best_cfg


def _normalize_config_value(val: Any) -> Any:
    """Convert sweep config values into JSON-serialisable primitives."""

    if isinstance(val, dict) and "value" in val:
        return _normalize_config_value(val["value"])

    if isinstance(val, bool):
        return int(val)

    if isinstance(val, str):
        raw = val.strip()
        lower = raw.lower()
        if lower in {"true", "t", "yes", "y", "on"}:
            return 1
        if lower in {"false", "f", "no", "n", "off"}:
            return 0
        if lower in {"none", "null"}:
            return None
        try:
            if any(ch in raw for ch in (".", "e", "E")):
                fval = float(raw)
                return int(fval) if fval.is_integer() else fval
            return int(raw)
        except Exception:
            return val

    return val


def grab_cfg_vals(runs: List[Any], key: str) -> List[Any]:
    out = []
    for r in runs:
        v = _config_lookup(r.config, key)
        if v is None:
            continue
        norm = _normalize_config_value(v)
        if norm is None:
            continue
        out.append(norm)
    return out


# ---------- sweep spec helpers ----------

NUMERIC_CLAMP_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "mask_ratio": (0.0, 0.95),
    "ema_decay": (0.0, 0.9999),
    "learning_rate": (1e-6, None),
}

CATEGORICAL_FALLBACK_OPTIONS: Dict[str, Sequence[Any]] = {
    "hidden_dim": (256, 384, 512),
    "num_layers": (3, 4, 5),
    "pretrain_batch_size": (64, 128, 256),
    "finetune_batch_size": (128, 256, 512),
}


def _sweep_param_map(sweep: Any) -> Dict[str, Any]:
    try:
        cfg = getattr(sweep, "config", None) or {}
        params = cfg.get("parameters", {}) if isinstance(cfg, dict) else {}
        return params if isinstance(params, dict) else {}
    except Exception:
        return {}


def _load_phase2_template_params(app_dir: str) -> Dict[str, Any]:
    tpl = os.path.join(app_dir, "sweeps", "grid_sweep_phase2.yaml")
    try:
        with open(tpl, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    params = data.get("parameters", {}) if isinstance(data, dict) else {}
    return params if isinstance(params, dict) else {}


def _lookup_param_spec(parameters: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    if not parameters:
        return None
    candidates: Iterable[str] = [key]
    if "-" in key:
        candidates = list(candidates) + [key.replace("-", "_")]
    if "_" in key:
        candidates = list(candidates) + [key.replace("_", "-")]
    seen: Set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        spec = parameters.get(cand)
        if isinstance(spec, dict):
            return spec
    return None


def _extend_numeric_constant(value: float, key: str, spec: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    base = float(value)
    delta = abs(base) * 0.2
    if delta == 0.0:
        delta = max(abs(base) * 0.1, 1e-6)
    lo = base - delta
    hi = base + delta

    if key == "mask_ratio":
        # Mask ratio searches benefit from a gentle band around the observed value.
        lo = min(lo, base - 0.05)
        hi = max(hi, base + 0.05)
    elif key == "learning_rate":
        lo = min(lo, base / 5.0, 5e-5)
        hi = max(hi, base * 3.0, 5e-4)


    if spec:
        try:
            if "min" in spec:
                lo = max(lo, float(spec["min"]))
            if "max" in spec:
                hi = min(hi, float(spec["max"]))
        except Exception:
            pass

    clamp = NUMERIC_CLAMP_BOUNDS.get(key)
    if clamp:
        min_v, max_v = clamp
        if min_v is not None:
            lo = max(lo, float(min_v))
        if max_v is not None:
            hi = min(hi, float(max_v))

    if hi <= lo:
        eps = max(abs(base) * 0.05, 1e-6)
        hi = lo + eps
    return float(lo), float(hi)


def _dedupe_options(options: Iterable[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for opt in options:
        norm = _normalize_config_value(opt)
        if norm is None or norm in seen:
            continue
        seen.add(norm)
        out.append(opt)
    return out


def _maybe_sort_numeric_options(options: List[Any]) -> List[Any]:
    if not options:
        return options
    try:
        norms = [_normalize_config_value(o) for o in options]
        if all(isinstance(n, (int, float)) for n in norms):
            pairs = sorted(zip(norms, options), key=lambda t: float(t[0]))
            return [opt for _norm, opt in pairs]
    except Exception:
        return options
    return options


def _iterable_from_option(value: Any) -> Optional[Iterable[Any]]:
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, (set, frozenset)):
        return list(value)
    return None


def _extend_categorical_constant(
    value: Any,
    key: str,
    specs: Sequence[Optional[Dict[str, Any]]],
) -> List[Any]:
    norm_value = _normalize_config_value(value)
    option_sources: List[Iterable[Any]] = []
    spec_option_pool: List[Any] = []
    for spec in specs:
        if not spec:
            continue
        vals = None
        if "values" in spec:
            vals = _iterable_from_option(spec["values"])
        elif "value" in spec:
            vals = _iterable_from_option(spec["value"])
        if vals is not None:
            vals_list = list(vals)
            option_sources.append(vals_list)
            spec_option_pool.extend(vals_list)

    fallback = CATEGORICAL_FALLBACK_OPTIONS.get(key)
    if fallback and len(_dedupe_options(spec_option_pool)) <= 1:
        option_sources.append(fallback)

    options: List[Any] = _dedupe_options(opt for src in option_sources for opt in src)

    normalized_options = {_normalize_config_value(o) for o in options}
    if norm_value not in normalized_options:
        options.insert(0, value)
        options = _dedupe_options(options)
        normalized_options = {_normalize_config_value(o) for o in options}

    if not options:
        if norm_value in (0, 1):
            options = [norm_value, 1 - norm_value]
        else:
            return [value]

    options = _maybe_sort_numeric_options(options)

    norm_options_list = [_normalize_config_value(o) for o in options]
    try:
        idx = norm_options_list.index(norm_value)
    except ValueError:
        idx = 0

    chosen: List[Any] = [options[idx]]
    higher = options[idx + 1] if idx + 1 < len(options) else None
    lower = options[idx - 1] if idx - 1 >= 0 else None
    if higher is not None:
        chosen.append(higher)
    elif lower is not None:
        chosen.append(lower)
    elif len(options) > 1:
        chosen.append(options[1])
    elif norm_value in (0, 1):
        # Binary toggles benefit from offering both possibilities.
        chosen.append(1 - norm_value)
        
    deduped = _dedupe_options(chosen)
    return deduped if deduped else [value]


def percentile_band(arr: List[Any]) -> Optional[Tuple[float, float]]:
    xs = [x for x in arr if isinstance(x, (int, float))]
    if not xs:
        return None
    lo, hi = np.percentile(xs, 10), np.percentile(xs, 90)
    return float(lo), float(hi)


def uniq_vals(arr: List[Any]) -> List[Any]:
    s = set()
    for v in arr:
        norm = _normalize_config_value(v)
        if norm is None:
            continue
        s.add(norm)
    return sorted(s, key=lambda x: (str(type(x)), str(x)))

# Task + metric plan (robust to missing val_rmse)
def pick_primary_metric(runs, task: str, args) -> Tuple[str, bool]:
    # (name, maximize?) ordered by preference
    if task == "regression":
        prefs = [("val_rmse", False), ("rmse", False), ("rmse_mean", False),
                    ("probe_rmse_mean", False), ("metric", False)]
    else:
        prefs = [("val_auc", True), ("pr_auc", True), ("roc_auc", True),
                    ("metric", True)]
    for name, mx in prefs:
        if any(metric(r, name) is not None for r in runs):
            return name, mx
    # fallback to the task default even if empty; choose_best will raise
    return (args.reg_primary if task=="regression" else args.clf_primary,
            False if task=="regression" else True)

# ---------- main ----------

def main():
    APP_DIR  = need_env("APP_DIR")
    GRID_DIR = need_env("GRID_DIR")
    ENTITY   = need_env("WANDB_ENTITY")
    PROJECT  = need_env("WANDB_PROJECT")

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_id", "--sweep-id", dest="sweep_id",
                    help="entity/project/sweepid to read from; "
                         "defaults to $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_SWEEP_ID1")
    ap.add_argument("--out", "--out-csv", dest="out",
                default=os.path.join(GRID_DIR, "best_grid_config.json"),
                help="Write best run CONFIG to this JSON")
    ap.add_argument("--phase2_yaml", "--phase2-yaml", dest="phase2_yaml",
                    default=os.path.join(GRID_DIR, "grid_sweep_phase2.yaml"),
                    help="Emit/overwrite Phase-2 sweep YAML here (default: $GRID_DIR/grid_sweep_phase2.yaml)")
    # phase-2 data roots (externalizable via CI YAML/env)
    ap.add_argument("--phase2_unlabeled_dir", "--phase2-unlabeled-dir",
                    dest="phase2_unlabeled_dir",
                    default=os.path.join(APP_DIR, "data", "ZINC-canonicalized"))
    ap.add_argument("--phase2_labeled_dir", "--phase2-labeled-dir",
                    dest="phase2_labeled_dir",
                    default=os.path.join(APP_DIR, "data", "katielinkmoleculenet_benchmark", "train"))

    # selection behavior
    ap.add_argument("--task", "--task-type", dest="task", choices=["auto","regression","classification"], default="auto")
    ap.add_argument("--tie_eps", "--tie-eps", dest="tie_eps", type=float, default=0.01)

    # metric names (override if logs differ)
    ap.add_argument("--reg_primary", "--reg-primary", dest="reg_primary", default="val_rmse")
    ap.add_argument("--reg_tb1", "--reg-tb1",  dest="reg_tb1", default="val_mae")
    ap.add_argument("--clf_primary", "--clf-primary", dest="clf_primary", default="val_auc")
    ap.add_argument("--clf_tb1", "--clf-tb1", dest="clf_tb1", default="val_brier")
    ap.add_argument("--clf_tb2", "--clf-tb2", dest="clf_tb2", default="val_pr_auc")

    # Phase-2 derivation
    ap.add_argument("--emit_bounds", "--emit-bounds", dest="emit_bounds", action="store_true",
                    help="Also derive top-K bounds and update Phase-2 YAML")
    ap.add_argument("--topk", "--top-k", dest="topk", type=int, default=20)
    ap.add_argument("--phase2_method", "--phase2-method", dest="phase2_method", default="bayes", choices=["bayes","random"])
    ap.add_argument("--phase2_metric", "--phase2-metric", dest="phase2_metric", default=None,
                    help="Override Phase-2 metric name; defaults to task primary")
    ap.add_argument(
        "--extend-fixed",
        dest="extend_fixed",
        action="store_true",
        default=True,
        help="Expand parameters that are fixed across top-K runs into small ranges/lists (default: enabled)",
    )
    ap.add_argument(
        "--no-extend-fixed",
        dest="extend_fixed",
        action="store_false",
        help="Disable automatic expansion of fixed top-K parameters",
    )

    args = ap.parse_args()

    sweep_id = args.sweep_id or f"{ENTITY}/{PROJECT}/{need_env('WANDB_SWEEP_ID1')}"
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    runs  = list(sweep.runs)
    if not runs:
        raise RuntimeError(f"No runs found in sweep {sweep_id}")

    # Task + metric plan
    task = args.task if args.task != "auto" else detect_task(runs)
    primary, maximize = pick_primary_metric(runs, task, args)
    tiebreakers: List[Tuple[str,bool]] = []
    if task == "regression":
        if args.reg_tb1: tiebreakers.append((args.reg_tb1, False))
    else:
        if args.clf_tb1: tiebreakers.append((args.clf_tb1, False))
        if args.clf_tb2: tiebreakers.append((args.clf_tb2, True))

    # Pick best
    best = choose_best(runs, primary, maximize, args.tie_eps, tiebreakers)

    # Log and write best config
    msg = f"[export_best] Best run={best.name} task={task} {primary}={metric(best, primary)}"
    for n,_mx in tiebreakers:
        v = metric(best, n)
        if v is not None:
            msg += f"  {n}={v}"
    print(msg)

    raw_config = getattr(best, "config", {}) or {}
    sanitized_config = _sanitize_run_config(raw_config)
    best_cfg = _enforce_required_keys(raw_config, sanitized_config)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2, sort_keys=True)

    summary_keys = ["training_method", "gnn_type", "hidden_dim", "num_layers"]
    summary = {k: best_cfg.get(k) for k in summary_keys if k in best_cfg}
    print(
        f"[export_best] Wrote best config to {args.out}: "
        f"{json.dumps(summary, sort_keys=True)}"
    )

    # Optionally derive narrowed Phase-2 ranges and write YAML for Step #3
    if args.emit_bounds:
        top = collect_topk(runs, primary, maximize, args.topk)
        sweep_params = _sweep_param_map(sweep)
        template_params = _load_phase2_template_params(APP_DIR) if args.extend_fixed else {}

        # Numeric → p10–p90; discrete → unique sets
        numeric_keys = ["mask_ratio", "ema_decay", "learning_rate"]
        set_keys = [
            # model
            "hidden_dim","num_layers","gnn_type","contiguity","add_3d",
            # aug (contrastive only, harmless to include)
            "aug_rotate","aug_mask_angle","aug_dihedral",
            "aug_bond_deletion","aug_atom_masking","aug_subgraph_removal",
            # training length/scale
            "pretrain_batch_size","finetune_batch_size",
            "pretrain_epochs","finetune_epochs",
            # task/data
            "task_type","label_col",
            # dataset caps (if used in Phase-1)
            "sample_unlabeled","sample_labeled","n_rows_per_file",
            # caching + loader performance knobs
            "cache-datasets","cache-dir",
            "num-workers","prefetch-factor","persistent-workers",
            "pin-memory","bf16","devices","use-wandb",
        ]

        params: Dict[str, Any] = {}

        for k in numeric_keys:
            band = percentile_band(grab_cfg_vals(top, k))
            if band:
                lo, hi = band
                if lo == hi:
                    if args.extend_fixed:
                        spec = _lookup_param_spec(sweep_params, k) or _lookup_param_spec(template_params, k)
                        lo, hi = _extend_numeric_constant(float(lo), k, spec)
                    else:
                        lo = float(lo) * 0.9
                        hi = max(float(hi) * 1.1, 1e-6)
                params[k] = {"min": float(lo), "max": float(hi)}

        for k in set_keys:
            arr = grab_cfg_vals(top, k)
            if not arr:
                continue
            vals = uniq_vals(arr)
            if len(vals) == 1:
                if args.extend_fixed:
                    specs = (
                        _lookup_param_spec(sweep_params, k),
                        _lookup_param_spec(template_params, k),
                    )
                    extended = _extend_categorical_constant(vals[0], k, specs)
                    if len(extended) == 1:
                        params[k] = {"value": extended[0]}
                    else:
                        params[k] = {"values": extended}
                else:
                    params[k] = {"value": vals[0]}
            else:
                params[k] = {"values": vals}

        params["labeled_dir"]      = {"value": "${env:PHASE2_LABELED_DIR}"}
        params["unlabeled_dir"]    = {"value": "${env:PHASE2_UNLABELED_DIR}"}

        # Normalise caching/performance knobs if the sweep logs were sparse.
        def _ensure_param(key: str, default=None):
            if key in params:
                return
            v = _config_lookup(best.config, key)
            if v is None and "-" in key:
                v = _config_lookup(best.config, key.replace("-", "_"))
            if v is None:
                v = default
            if v is None:
                return
            params[key] = {"value": _normalize_config_value(v)}

        env_cache_root = os.environ.get("SWEEP_CACHE_DIR")
        if env_cache_root:
            params["cache-dir"] = {"value": "${env:SWEEP_CACHE_DIR}"}
        else:
            _ensure_param("cache-dir", default=_config_lookup(best.config, "cache-dir") or _config_lookup(best.config, "cache_dir"))

        _ensure_param("cache-datasets", default=1)
        _ensure_param("num-workers", default=4)
        _ensure_param("prefetch-factor", default=2)
        _ensure_param("persistent-workers", default=0)
        _ensure_param("pin-memory", default=0)
        _ensure_param("bf16", default=1)
        _ensure_param("devices", default=1)
        _ensure_param("use-wandb", default=1)

        # safety in case Schnet is chosen and 3D is not set
        try:
            gnn_val = (params.get("gnn_type", {}) or {}).get("value")
            if gnn_val == "schnet3d":
                params["add_3d"] = {"value": 1}
        except Exception:
            pass
        

        phase2_metric_name = args.phase2_metric or primary
        phase2_goal        = "maximize" if maximize else "minimize"

        # Clean Phase-2 space to match JEPA-only Bayes with ranges

        # --- Phase-2 policy tweaks to match JEPA-only Bayes with ranges ---
        # --- Winner-aware Phase-2 policy (JEPA or Contrastive) with ranges ---
        # Winner comes from env (set by run-grid after paired-effect); default to JEPA.
        winner = os.environ.get("METHOD_WINNER", "jepa")

        # If JEPA won, drop contrastive-only knobs (aug_* and temperature).
        # If Contrastive won, keep them (and optionally drop JEPA-only knobs if you add any later).
        if winner == "jepa":
            for k in list(params.keys()):
                if k.startswith("aug_") or k in ("temperature", "seed", "seeds"):
                    params.pop(k, None)
        elif winner == "contrastive":
            for k in list(params.keys()):
                if k in ("mask_ratio", "ema_decay"):
                    params.pop(k, None)

        # Suggest distributions for continuous ranges so Bayes can interpolate well.
        if isinstance(params.get("mask_ratio"), dict) and "min" in params["mask_ratio"]:
            params["mask_ratio"].setdefault("distribution", "uniform")
        if isinstance(params.get("learning_rate"), dict) and "min" in params["learning_rate"]:
            params["learning_rate"].setdefault("distribution", "log_uniform_values")

        # JEPA or Contrastive per Phase-1 winner
        training_method_param = {"value": winner}

        # Early termination to speed up Bayes search
        early_terminate_cfg = {"type": "hyperband", "min_iter": 3}

        # Initialise optional W&B run for grid search
        wb = maybe_init_wandb(
            enable=True,
            project=os.environ.get("WANDB_PROJECT", PROJECT),
            tags=["export_best"],
            config={
                "sweep_id": sweep_id,
                "task": "auto",
                "tie_eps": args.tie_eps,
                "phase2_emit_bounds": bool(args.emit_bounds),
                "phase2_method": args.phase2_method,
                "phase2_metric": args.phase2_metric or "",
                # anything else we want as metadata…
            },
            api_key=os.environ.get("WANDB_API_KEY"),
        )

        spec = {
            "program": "${env:APP_DIR}/scripts/train_jepa.py",
            "command": [
                "${interpreter}", "${program}", "sweep-run",
                # "--unlabeled-dir", args.phase2_unlabeled_dir, # sending it as normal params
                #  "--labeled-dir",   args.phase2_labeled_dir, # sending it as normal params
                "${args}"
            ],
            "method": args.phase2_method,                 # expect "bayes"
            "metric": {"name": phase2_metric_name, "goal": phase2_goal},
            "early_terminate": early_terminate_cfg,       # optional (matches pic)
            "parameters": {
                "training_method": training_method_param, # JEPA or Contrastive (winner)
                **params
            }
        }

        # log the winner to the run’s summary (one-shot; no step noise)
        # one-shot summary on this export step's run
        if wb:
            s = {
                "best_run_name": best.name,
                "best_run_id":   best.id,
            }
            pv = metric(best, primary)
            if pv is not None:
                s[primary] = float(pv)
            for n, _mx in tiebreakers:
                v = metric(best, n)
                if v is not None:
                    # try float; fall back to str for non-numeric types
                    try: s[n] = float(v)
                    except Exception: s[n] = str(v)

            wb.summary.update(s)
            wb.finish()

        tpl_phase2 = os.path.realpath(
            os.path.join(APP_DIR, "sweeps", "grid_sweep_phase2.yaml")
        )
        target_phase2 = os.path.realpath(args.phase2_yaml)
        same_as_template = target_phase2 == tpl_phase2
        try:
            same_as_template = same_as_template or os.path.samefile(
                target_phase2, tpl_phase2
            )
        except FileNotFoundError:
            same_as_template = False
        sweeps_suffix = os.sep + os.path.join("sweeps", "grid_sweep_phase2.yaml")
        if same_as_template or target_phase2.endswith(sweeps_suffix):
            raise RuntimeError(
                "Refusing to overwrite tracked Phase-2 template at "
                f"{tpl_phase2}. Set GRID_DIR or pass --phase2-yaml to a "
                "writable workspace (e.g., $GRID_DIR/grid_sweep_phase2.yaml)."
            )

        os.makedirs(os.path.dirname(args.phase2_yaml) or ".", exist_ok=True)
        with open(args.phase2_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False)
            
        print(f"[export_best] Wrote Phase-2 sweep YAML to {args.phase2_yaml}")


if __name__ == "__main__":
    main()
