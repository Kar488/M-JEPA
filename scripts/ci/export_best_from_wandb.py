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

import argparse, copy, csv, json, math, os, time
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from utils.logging import maybe_init_wandb
from utils.wandb_filters import silence_pydantic_field_warnings

silence_pydantic_field_warnings()

import numpy as np
import requests
import wandb
import yaml
from urllib.parse import urlparse

try:
    from reports.wandb_utils import resolve_wandb_http_timeout
except ImportError:  # pragma: no cover - optional dependency
    def resolve_wandb_http_timeout(default: Union[int, float]) -> Union[int, float]:
        """Fallback when reporting helpers are unavailable."""

        return default


# ---------- env helpers ----------

def need_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env: {name}")
    return v


def _ensure_wandb_env() -> None:
    if getattr(wandb, "__spec__", None) is None:
        # During tests ``wandb`` is replaced with a lightweight stub that does not
        # require authentication. Skip the credential guard so fake clients can
        # operate without a real API key.
        return

    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set; cannot authenticate with Weights & Biases.")
        raise SystemExit(1)

    base_url = os.environ.get("WANDB_BASE_URL")
    if base_url:
        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            print(
                "WANDB_BASE_URL is set but invalid. Expected a full URL such as "
                "https://api.wandb.ai; please correct or unset it."
            )
            raise SystemExit(1)


def _coerce_env_bool(value: Optional[str]) -> Optional[bool]:
    """Interpret common truthy/falsey environment values.

    Returns ``True``/``False`` for recognised tokens and ``None`` otherwise.
    """

    if value is None:
        return None

    normalized = value.strip().lower()
    if not normalized:
        return None

    truthy = {"1", "true", "yes", "on"}
    falsey = {"0", "false", "no", "off"}

    if normalized in truthy:
        return True
    if normalized in falsey:
        return False

    return None


def _phase2_prefers_cpu() -> bool:
    """Decide whether the Phase-2 export should emit CPU-friendly defaults.

    GitHub runners only expose CPUs, so we prefer conservative loader knobs
    whenever we detect a CI context.  Vast jobs typically provision GPUs; those
    should inherit the GPU-ready defaults instead.  Both behaviours can be
    overridden explicitly via ``CI_FORCE_CPU_PHASE2`` or
    ``CI_FORCE_GPU_PHASE2``.
    """

    force_cpu = _coerce_env_bool(os.environ.get("CI_FORCE_CPU_PHASE2"))
    if force_cpu is True:
        return True

    force_gpu = _coerce_env_bool(os.environ.get("CI_FORCE_GPU_PHASE2"))
    if force_gpu is True:
        return False

    ci_flag = _coerce_env_bool(os.environ.get("CI"))
    if ci_flag is True:
        return True

    gh_actions = _coerce_env_bool(os.environ.get("GITHUB_ACTIONS"))
    if gh_actions is True:
        return True

    return False


def _init_wandb_api(max_attempts: int = 3, timeout: int = 60) -> wandb.Api:
    _ensure_wandb_env()

    resolved_timeout = resolve_wandb_http_timeout(timeout)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return wandb.Api(timeout=resolved_timeout)
        except TypeError:
            print(
                "[export_best] wandb.Api does not accept a timeout argument; retrying without it",
                flush=True,
            )
            try:
                return wandb.Api()
            except Exception as err:
                last_error = err
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            wandb.errors.CommError,
            wandb.errors.AuthenticationError,
        ) as err:
            last_error = err

        if last_error is None:
            continue

        if attempt < max_attempts:
            sleep_for = min(30.0, 2.0 * attempt)
            print(
                f"[export_best] W&B client init failed ({type(last_error).__name__}); "
                f"retrying in {sleep_for:.1f}s..."
            )
            time.sleep(sleep_for)
            last_error = None
            continue
        break

    message = "Failed to initialize the W&B client after multiple attempts. "
    if last_error:
        message += f"Last error: {last_error}"
    print(message.strip())
    raise SystemExit(1)


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


def _qualify_sweep_id(raw: str, entity: str, project: str) -> str:
    token = (raw or "").strip()
    if not token:
        return token
    if "/" in token:
        return token
    base = f"{entity}/{project}" if entity else project
    return f"{base}/{token}" if base else token


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


def _coerce_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            return float(token)
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


RMSE_KEYS: Tuple[str, ...] = (
    "val_rmse",
    "validation.rmse",
    "metrics/val_rmse",
    "rmse",
    "metrics/rmse",
    "rmse_mean",
    "metrics/rmse_mean",
)

R2_KEYS: Tuple[str, ...] = (
    "val_r2",
    "validation.r2",
    "metrics/val_r2",
)

R2_MEAN_KEYS: Tuple[str, ...] = (
    "val_r2_mean",
    "metrics/val_r2_mean",
    "validation.r2_mean",
)

R2_STD_KEYS: Tuple[str, ...] = (
    "val_r2_std",
    "metrics/val_r2_std",
    "validation.r2_std",
)

R2_CI95_KEYS: Tuple[str, ...] = (
    "val_r2_ci95",
    "metrics/val_r2_ci95",
    "validation.r2_ci95",
)

BRIER_KEYS: Tuple[str, ...] = (
    "val_brier",
    "validation.brier",
    "metrics/val_brier",
)

BRIER_MEAN_KEYS: Tuple[str, ...] = (
    "val_brier_mean",
    "metrics/val_brier_mean",
    "validation.brier_mean",
)

BRIER_STD_KEYS: Tuple[str, ...] = (
    "val_brier_std",
    "metrics/val_brier_std",
    "validation.brier_std",
)

BRIER_CI95_KEYS: Tuple[str, ...] = (
    "val_brier_ci95",
    "metrics/val_brier_ci95",
    "validation.brier_ci95",
)


def _resolve_metric_value(run: Any, candidates: Sequence[str]) -> Optional[float]:
    for key in candidates:
        value = metric(run, key)
        coerced = _coerce_to_float(value)
        if coerced is not None:
            return coerced
    return None


def _serialise_config_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            try:
                return json.dumps(_unwrap_config_value(value), sort_keys=True)
            except Exception:
                return str(value)
    return value


def _build_run_record(run: Any, sweep_id: str, is_winner: bool = False) -> Dict[str, Any]:
    config = _sanitize_run_config(getattr(run, "config", {}) or {})

    # Persist the resolved device count when the sweep config omitted it but
    # the launcher exposed a CUDA mask. This keeps phase2_winner_config.csv in
    # sync with the actual GPU allocation used for the winner.
    if "devices" not in config:
        inferred_devices = _visible_device_count_from_env()
        if inferred_devices is not None:
            config["devices"] = inferred_devices

    record: Dict[str, Any] = {
        "sweep_id": sweep_id,
        "run_id": getattr(run, "id", ""),
        "run_name": getattr(run, "name", ""),
        "state": getattr(run, "state", ""),
        "is_winner": bool(is_winner),
    }
    method = config.get("training_method")
    if isinstance(method, str):
        record["training_method"] = method
    else:
        summary_method = metric(run, "training_method")
        if isinstance(summary_method, str):
            record["training_method"] = summary_method

    # Export secondary metrics in raw form alongside aggregated statistics so
    # sweep-level analysis can audit per-run trajectories without altering the
    # selection logic.
    rmse_val = _resolve_metric_value(run, RMSE_KEYS)
    if rmse_val is not None:
        record["metric_rmse"] = rmse_val
    r2_val = _resolve_metric_value(run, R2_KEYS)
    if r2_val is not None:
        record["metric_r2"] = r2_val
        record["val_r2"] = r2_val
    for key_list, field in (
        (R2_MEAN_KEYS, "val_r2_mean"),
        (R2_STD_KEYS, "val_r2_std"),
        (R2_CI95_KEYS, "val_r2_ci95"),
    ):
        value = _resolve_metric_value(run, key_list)
        if value is not None:
            record[field] = value

    brier_val = _resolve_metric_value(run, BRIER_KEYS)
    if brier_val is not None:
        record["val_brier"] = brier_val
    for key_list, field in (
        (BRIER_MEAN_KEYS, "val_brier_mean"),
        (BRIER_STD_KEYS, "val_brier_std"),
        (BRIER_CI95_KEYS, "val_brier_ci95"),
    ):
        value = _resolve_metric_value(run, key_list)
        if value is not None:
            record[field] = value

    for key, value in config.items():
        record[f"config.{key}"] = _serialise_config_value(value)

    return record


def _write_records_csv(path: str, records: Sequence[Dict[str, Any]]) -> int:
    if not path or not records:
        return 0

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames: List[str] = sorted({key for rec in records for key in rec.keys()})
    if not fieldnames:
        return 0

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {field: rec.get(field, "") for field in fieldnames}
            writer.writerow(row)

    return len(records)


def _visible_device_count_from_env() -> Optional[int]:
    """Best-effort CUDA_VISIBLE_DEVICES length, ignoring empty tokens."""

    mask = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not mask:
        return None
    tokens = [tok for tok in mask.split(",") if tok.strip()]
    return len(tokens) if tokens else None


def _has_any_metric(runs, names: Sequence[str]) -> bool:
    return any(metric(r, name) is not None for name in names for r in runs)


def detect_task(runs) -> str:
    auc_keys = ("val_auc", "metrics/val_auc", "validation.auc")
    have_auc = _has_any_metric(runs, auc_keys)
    have_rmse = _has_any_metric(runs, RMSE_KEYS)
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

    if "add_3d" not in best_cfg:
        add_3d_option = _config_lookup(raw_config, "add_3d_options")
        if add_3d_option is None:
            add_3d_option = _config_lookup(sanitized, "add_3d_options")
        if isinstance(add_3d_option, (list, tuple)) and len(add_3d_option) == 1:
            add_3d_option = add_3d_option[0]
        if add_3d_option is not None:
            best_cfg["add_3d"] = _normalize_config_value(add_3d_option)

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

def _primary_preferences(task: str, args: Any) -> List[Tuple[Tuple[str, ...], bool]]:
    def _prepend_override(pref_list: List[Tuple[Tuple[str, ...], bool]], name: Optional[str], maximize: bool) -> None:
        token = (name or "").strip()
        if token:
            pref_list.insert(0, ((token,), maximize))

    if task == "regression":
        prefs: List[Tuple[Tuple[str, ...], bool]] = [
            (("val_rmse", "metrics/val_rmse", "validation.rmse"), False),
            (("rmse", "metrics/rmse"), False),
            (("rmse_mean", "metrics/rmse_mean"), False),
            (("probe_rmse_mean",), False),
            (("metric",), False),
        ]
        _prepend_override(prefs, getattr(args, "reg_primary", None), False)
        return prefs

    prefs = [
        (("val_auc", "metrics/val_auc", "validation.auc"), True),
        (("pr_auc", "metrics/pr_auc", "validation.pr_auc"), True),
        (("roc_auc", "metrics/roc_auc"), True),
        (("metric",), True),
    ]
    _prepend_override(prefs, getattr(args, "clf_primary", None), True)
    return prefs


def _find_present_metric(runs, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if any(metric(r, name) is not None for r in runs):
            return name
    return None


# Task + metric plan (robust to missing val_rmse)
def pick_primary_metric(runs, task: str, args) -> Tuple[str, bool]:
    prefs = _primary_preferences(task, args)

    for names, mx in prefs:
        found = _find_present_metric(runs, names)
        if found:
            return found, mx

    # fallback to the task default even if empty; choose_best will raise
    return (
        args.reg_primary if task == "regression" else args.clf_primary,
        False if task == "regression" else True,
    )

# ---------- main ----------

def _collect_sweep_ids(primary: str, include_sweeps_raw: Optional[Sequence[str]], entity: str, project: str) -> List[str]:
    include_sweeps_raw = include_sweeps_raw or []
    include_sweeps: List[str] = []

    def _maybe_add(raw: Optional[str]) -> None:
        qualified = _qualify_sweep_id(raw, entity, project)
        if qualified and qualified not in include_sweeps:
            include_sweeps.append(qualified)

    # Always include the primary sweep, then append any explicit include flags.
    _maybe_add(primary)
    for raw in include_sweeps_raw:
        _maybe_add(raw)

    # Mirror Phase-1 policy: automatically fold in both JEPA/contrastive sweeps
    # when CI exported them via WANDB_SWEEP_ID{1,2}. This ensures metrics CSVs
    # enumerate both methods even if the caller forgets to pass --include-sweep.
    #
    # However, when the caller explicitly provides include sweeps we should not
    # also inject the environment defaults; stale WANDB_SWEEP_ID* bindings from
    # older runs can otherwise leak into the export and pollute the metrics CSV
    # with unrelated runs.
    if not include_sweeps_raw:
        _maybe_add(os.environ.get("WANDB_SWEEP_ID1"))
        _maybe_add(os.environ.get("WANDB_SWEEP_ID2"))

    return include_sweeps


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
    ap.add_argument(
        "--metrics-csv",
        dest="metrics_csv",
        default=os.path.join(GRID_DIR, "phase1_export", "stage-outputs", "phase1_runs.csv"),
        help="Export per-run hyperparameters and metrics to this CSV (default: $GRID_DIR/phase1_export/stage-outputs/phase1_runs.csv)",
    )
    ap.add_argument(
        "--winner-csv",
        dest="winner_csv",
        default=os.path.join(GRID_DIR, "phase2_export", "stage-outputs", "phase2_winner_config.csv"),
        help="Export the selected Phase-2 seed configuration to this CSV",
    )
    ap.add_argument(
        "--include-sweep",
        dest="include_sweeps",
        action="append",
        default=None,
        help=(
            "Additional sweep identifiers to include when exporting the per-run "
            "CSV (repeatable). Accepts either bare sweep IDs or fully qualified "
            "entity/project/id paths."
        ),
    )
    # phase-2 data roots (externalizable via CI YAML/env)
    ap.add_argument(
        "--phase2_unlabeled_dir",
        "--phase2-unlabeled-dir",
        dest="phase2_unlabeled_dir",
        default=os.path.join(APP_DIR, "data", "ZINC-canonicalized"),
        help="Directory of the source unlabeled dataset (e.g., ZINC-canonicalized)",
    )
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
    sweep_ids = _collect_sweep_ids(sweep_id, args.include_sweeps, ENTITY, PROJECT)

    api = _init_wandb_api()

    sweep = api.sweep(sweep_id)
    primary_runs = list(sweep.runs)
    if not primary_runs:
        raise RuntimeError(f"No runs found in sweep {sweep_id}")

    per_sweep_runs: Dict[str, List[Any]] = {sweep_id: primary_runs}
    for extra_sweep in sweep_ids:
        if extra_sweep in per_sweep_runs:
            continue
        try:
            per_sweep_runs[extra_sweep] = list(api.sweep(extra_sweep).runs)
        except Exception as exc:  # pragma: no cover - network/API variability
            print(
                f"[export_best][warn] failed to load sweep {extra_sweep}: {exc}",
                flush=True,
            )
            continue

    print(
        "[export_best] discovered sweeps with run counts: "
        + ", ".join(
            f"{sid}={len(per_sweep_runs.get(sid, []))}" for sid in sweep_ids
        ),
        flush=True,
    )

    # Task + metric plan
    task = args.task if args.task != "auto" else detect_task(primary_runs)
    primary, maximize = pick_primary_metric(primary_runs, task, args)
    tiebreakers: List[Tuple[str,bool]] = []
    if task == "regression":
        if args.reg_tb1: tiebreakers.append((args.reg_tb1, False))
    else:
        if args.clf_tb1: tiebreakers.append((args.clf_tb1, False))
        if args.clf_tb2: tiebreakers.append((args.clf_tb2, True))

    # Pick best
    best = choose_best(primary_runs, primary, maximize, args.tie_eps, tiebreakers)

    winner_method: Optional[str] = None
    best_cfg_method = _normalize_config_value(_config_lookup(getattr(best, "config", {}), "training_method"))
    if isinstance(best_cfg_method, str):
        winner_method = best_cfg_method
    env_winner = os.environ.get("METHOD_WINNER")
    if env_winner:
        winner_method = env_winner.strip()

    # Collect per-run metrics and annotate the winner
    best_id = getattr(best, "id", "")
    best_name = getattr(best, "name", "")
    metrics_records: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for sweep_key, sweep_runs in per_sweep_runs.items():
        for run in sweep_runs:
            run_id = getattr(run, "id", None)
            dedup_key = (sweep_key, run_id or "")
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            try:
                is_winner = False
                if run_id and run_id == best_id:
                    is_winner = True
                elif not best_id and getattr(run, "name", None) == best_name:
                    is_winner = True

                record = _build_run_record(run, sweep_key, is_winner=is_winner)
                if winner_method:
                    record["phase1_winner_method"] = winner_method
                    record["is_phase1_winner_method"] = (
                        _normalize_config_value(record.get("training_method")) == _normalize_config_value(winner_method)
                    )
                metrics_records.append(record)
            except Exception as exc:  # pragma: no cover - defensive guard
                print(
                    f"[export_best][warn] failed to serialise run {run_id} from {sweep_key}: {exc}",
                    flush=True,
                )
                continue

    if args.metrics_csv and metrics_records:
        rows_written = _write_records_csv(args.metrics_csv, metrics_records)
        print(
            f"[export_best] wrote Phase-1 sweep metrics to {args.metrics_csv} "
            f"({rows_written} rows)",
            flush=True,
        )

        legacy_metrics = os.path.join(GRID_DIR, "phase1_runs.csv")
        if os.path.abspath(args.metrics_csv) != os.path.abspath(legacy_metrics):
            try:
                _write_records_csv(legacy_metrics, metrics_records)
                print(
                    f"[export_best] mirrored Phase-1 metrics to legacy path {legacy_metrics}",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                print(
                    f"[export_best][warn] unable to mirror metrics to {legacy_metrics}: {exc}",
                    flush=True,
                )

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

    if args.winner_csv:
        winner_records = [_build_run_record(best, sweep_id, is_winner=True)]
        winner_rows = _write_records_csv(args.winner_csv, winner_records)
        if winner_rows:
            print(
                f"[export_best] wrote Phase-2 winner metrics to {args.winner_csv}",
                flush=True,
            )
        legacy_winner = os.path.join(GRID_DIR, "phase2_winner_config.csv")
        if os.path.abspath(args.winner_csv) != os.path.abspath(legacy_winner):
            try:
                _write_records_csv(legacy_winner, winner_records)
                print(
                    f"[export_best] mirrored winner metrics to legacy path {legacy_winner}",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    f"[export_best][warn] unable to mirror winner metrics to {legacy_winner}: {exc}",
                    flush=True,
                )

    # Optionally derive narrowed Phase-2 ranges and write YAML for Step #3
    if args.emit_bounds:
        top = collect_topk(primary_runs, primary, maximize, args.topk)
        sweep_params = _sweep_param_map(sweep)
        template_params = _load_phase2_template_params(APP_DIR)

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

        params: Dict[str, Any] = copy.deepcopy(template_params) if template_params else {}
        params.pop("training_method", None)

        prefers_cpu = _phase2_prefers_cpu()

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

        template_overrides = {
            "pretrain_batch_size",
            "finetune_batch_size",
            "pretrain_epochs",
            "max_pretrain_batches",
            "finetune_epochs",
            "sample_unlabeled",
        }
        for key in template_overrides:
            if key in template_params:
                params[key] = template_params[key]

        params["labeled_dir"]      = {"value": "${env:PHASE2_LABELED_DIR}"}
        params["unlabeled_dir"]    = {"value": "${env:PHASE2_UNLABELED_DIR}"}

        def _force_param_value(key: str, value: Any) -> None:
            params[key] = {"value": _normalize_config_value(value)}

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
            _force_param_value("cache-dir", "${env:SWEEP_CACHE_DIR}")
        else:
            cache_default = (
                _config_lookup(best.config, "cache-dir")
                or _config_lookup(best.config, "cache_dir")
                or "cache/graphs"
            )
            _force_param_value("cache-dir", cache_default)

        _force_param_value("cache-datasets", 1)
        _ensure_param("num-workers", default=4)
        _ensure_param("prefetch-factor", default=2)
        if prefers_cpu:
            _force_param_value("persistent-workers", 0)
            _force_param_value("pin-memory", 0)
            _force_param_value("devices", 1)
        else:
            # Respect explicit GPU overrides from the sweep, only correcting
            # configs that inherited CPU defaults during phase-1 exploration.
            pin_memory_cfg = _normalize_config_value(
                _config_lookup(best.config, "pin-memory")
            )
            persistent_workers_cfg = _normalize_config_value(
                _config_lookup(best.config, "persistent-workers")
            )
            devices_cfg = _normalize_config_value(
                _config_lookup(best.config, "devices")
            )

            _ensure_param("persistent-workers")
            _ensure_param("pin-memory")
            _ensure_param("devices")

            def _as_int(val: Any) -> Optional[int]:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return None

            inherited_cpu_defaults = (
                (_as_int(devices_cfg) in {None, 1})
                and (_as_int(pin_memory_cfg) in {None, 0})
                and (_as_int(persistent_workers_cfg) in {None, 0})
            )

            if inherited_cpu_defaults:
                _force_param_value("persistent-workers", 1)
                _force_param_value("pin-memory", 1)
                _force_param_value("devices", 2)
        _force_param_value("bf16", 1)
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

        if prefers_cpu:
            spec["parameters"]["persistent-workers"] = {"value": 0}
            spec["parameters"]["pin-memory"] = {"value": 0}
            spec["parameters"]["devices"] = {"value": 1}
        else:
            spec_params = spec["parameters"]
            spec_params.setdefault("persistent-workers", {"value": 1})
            spec_params.setdefault("pin-memory", {"value": 1})
            spec_params.setdefault("devices", {"value": 2})

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
