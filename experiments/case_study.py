"""Case study using real Tox21 toxicity labels.

This module demonstrates how JEPA embeddings can prioritise molecules by
ranking predictions on the Tox21 dataset. A small encoder is pretrained on
unlabelled molecules, a classification head is fitted on a chosen toxicity task
and the most toxic predictions are compared against a random exclusion
baseline.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import os
import pickle
import random
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from data.mdataset import EDGE_BASE_DIM, EDGE_TOTAL_DIM

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _HAS_RDKIT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_RDKIT = False

try:  # pragma: no cover - optional dependency
    from data.scaffold_split import scaffold_split_indices
except Exception:  # pragma: no cover - fallback when RDKit is absent
    scaffold_split_indices = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from scripts.bench import BenchmarkRule, resolve_metric_threshold
except Exception:  # pragma: no cover - fallback when scripts package unavailable
    BenchmarkRule = None  # type: ignore[assignment]
    resolve_metric_threshold = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from utils.checkpoint import (
        compute_state_dict_hash,
        extract_encoder_hash,
        load_state_dict_forgiving,
        safe_load_checkpoint,
    )
    try:  # pragma: no cover - optional dependency
        from utils.checkpoint import _prepare_state_dict_for_module  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fall back when helper unavailable
        _prepare_state_dict_for_module = None  # type: ignore[assignment]
except Exception:  # pragma: no cover - fallback for environments without checkpoint helpers
    load_state_dict_forgiving = None  # type: ignore[assignment]
    safe_load_checkpoint = None  # type: ignore[assignment]
    compute_state_dict_hash = None  # type: ignore[assignment]
    extract_encoder_hash = None  # type: ignore[assignment]
    _prepare_state_dict_for_module = None  # type: ignore[assignment]

from models.ema import EMA
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor

try:  # pragma: no cover - optional dependency
    from models.factory import build_encoder as _IMPORTED_BUILD_ENCODER
except Exception:  # pragma: no cover - fallback when factory is unavailable
    _IMPORTED_BUILD_ENCODER = None  # type: ignore[assignment]
from utils.seed import set_seed
from utils.metrics import expected_calibration_error

import inspect

if TYPE_CHECKING:  # pragma: no cover - typing only
    from data.mdataset import GraphDataset as GraphDatasetT

logger = logging.getLogger(__name__)


def _ci_diag_enabled() -> bool:
    value = os.getenv("CI_DIAG", "")
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no"}


def _ci_log(message: str, **payload: Any) -> None:
    if not _ci_diag_enabled():
        return
    if payload:
        try:
            serialised = json.dumps(payload, sort_keys=True)
        except Exception:
            serialised = ", ".join(f"{k}={v}" for k, v in sorted(payload.items()))
        logger.info("[ci][info] %s %s", message, serialised)
    else:
        logger.info("[ci][info] %s", message)


def _sanitize_binary_labels(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Coerce binary labels and drop sentinel values.

    The Tox21 CSVs occasionally encode missing labels as negative numbers or
    other non-binary values. Treat any non-finite/negative entry as missing and
    discard rows that are not strict ``{0, 1}`` labels.

    Args:
        df: Input dataframe containing the label column.
        label_col: Name of the label column to sanitise.

    Returns:
        A tuple of (clean_dataframe, drop_stats) where ``drop_stats`` records
        how many rows were removed for each reason.
    """

    if label_col not in df.columns:
        return df, {"dropped_negative": 0, "dropped_non_binary": 0, "dropped_na": 0}

    filtered = df.copy()
    filtered[label_col] = pd.to_numeric(filtered[label_col], errors="coerce")

    drop_stats = {"dropped_negative": 0, "dropped_non_binary": 0, "dropped_na": 0}

    negative_mask = filtered[label_col] < 0
    drop_stats["dropped_negative"] = int(negative_mask.sum())
    if drop_stats["dropped_negative"]:
        filtered.loc[negative_mask, label_col] = np.nan

    non_binary_mask = filtered[label_col].notna() & ~filtered[label_col].isin([0, 1])
    drop_stats["dropped_non_binary"] = int(non_binary_mask.sum())
    if drop_stats["dropped_non_binary"]:
        filtered.loc[non_binary_mask, label_col] = np.nan

    before_drop = len(filtered)
    filtered = filtered.dropna(subset=[label_col])
    drop_stats["dropped_na"] = before_drop - len(filtered)

    return filtered, drop_stats


# ``training.supervised`` and ``training.unsupervised`` are heavy modules that are
# frequently monkeypatched in tests.  ``run_tox21_case_study`` only needs
# ``train_linear_head`` and ``train_jepa`` so import them defensively to honour
# test stubs.
try:  # pragma: no cover - exercised mainly in tests
    try:
        supervised_mod = importlib.import_module("training.supervised")
        unsupervised_mod = importlib.import_module("training.unsupervised")
    except ModuleNotFoundError as exc:
        # When the real ``training`` package is unavailable (for instance, when
        # tests register lightweight stubs directly in ``sys.modules``) we fall
        # back to a minimal namespace package so ``importlib`` can resolve the
        # dotted module path.  Creating the stub pre-emptively would shadow the
        # real package, so only install it after the initial import fails.
        if exc.name and not exc.name.startswith("training"):
            raise
        if "training" not in sys.modules:
            sys.modules["training"] = types.ModuleType("training")
        supervised_mod = importlib.import_module("training.supervised")
        unsupervised_mod = importlib.import_module("training.unsupervised")
except Exception as exc:  # pragma: no cover - fail fast if even the stub is missing
    raise ImportError(
        "train_linear_head and train_jepa are required to run the Tox21 case study"
    ) from exc

train_linear_head = getattr(supervised_mod, "train_linear_head", None)
train_jepa = getattr(unsupervised_mod, "train_jepa", None)
if train_linear_head is None or train_jepa is None:
    raise ImportError(
        "train_linear_head and train_jepa are required to run the Tox21 case study"
    )

_ORIGINAL_TRAIN_LINEAR_HEAD = train_linear_head
_ORIGINAL_TRAIN_JEPA = train_jepa


def _resolve_training_callable(name: str, fallback: Optional[Callable[..., Any]]) -> Callable[..., Any]:
    """Return a fresh training callable, honouring runtime monkeypatches.

    ``experiments.case_study`` caches references to ``train_linear_head`` and
    ``train_jepa`` at import time to avoid repeatedly importing the heavy
    ``training`` package.  Individual tests often replace the corresponding
    modules inside :mod:`sys.modules` without reloading this file, so we need to
    re-discover the latest callable when we invoke the helpers.
    """

    module_name = "training.supervised" if name == "train_linear_head" else "training.unsupervised"
    module = sys.modules.get(module_name)
    module_candidate = getattr(module, name, None) if module is not None else None

    if callable(fallback):
        original = (
            _ORIGINAL_TRAIN_LINEAR_HEAD if name == "train_linear_head" else _ORIGINAL_TRAIN_JEPA
        )
        if fallback is not original:
            return fallback

    if callable(module_candidate):
        return module_candidate

    if callable(fallback):
        return fallback

    raise ImportError(f"{name} is unavailable; ensure training stubs are imported")


def _train_linear_head_callable() -> Callable[..., Any]:
    return _resolve_training_callable("train_linear_head", train_linear_head)


def _train_jepa_callable() -> Callable[..., Any]:
    return _resolve_training_callable("train_jepa", train_jepa)


def _build_encoder_callable() -> Optional[Callable[..., Any]]:
    """Return the latest ``build_encoder`` implementation if available.

    ``experiments.case_study`` imports :func:`models.factory.build_encoder` at
    module import time, but many tests swap the ``models`` package with lightweight
    stubs *after* the module has been loaded.  We therefore resolve the callable at
    runtime so that monkeypatched factories are honoured instead of the cached
    import.
    """

    module = sys.modules.get("models.factory")
    candidate = getattr(module, "build_encoder", None) if module is not None else None
    if callable(candidate):
        return candidate
    if callable(_IMPORTED_BUILD_ENCODER):
        return _IMPORTED_BUILD_ENCODER
    return None

try:  # pragma: no cover - optional during tests
    from training.supervised import stratified_split as _lib_stratified_split  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - provide a lightweight fallback
    _lib_stratified_split = None


def _fallback_stratified_split(
    indices: Iterable[int],
    labels: Iterable[Any],
    *,
    train_frac: float,
    val_frac: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Simple stratified split for binary labels.

    Mirrors the behaviour of :func:`training.supervised.stratified_split` while
    avoiding the import-time dependency on the full training module.  The
    function operates purely on Python lists/Numpy arrays so that tests can
    supply lightweight stubs.
    """

    idx_list = list(indices)
    if not idx_list:
        return [], [], []

    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != len(idx_list):
        raise ValueError("labels must align with indices for stratified split")

    mask_finite = np.isfinite(labels_arr)
    observed = labels_arr[mask_finite]
    if observed.size == 0:
        observed = np.zeros((0,), dtype=int)

    unique = np.unique(observed.astype(int, copy=False)) if observed.size else np.array([])
    if unique.size < 2:
        random.shuffle(idx_list)
        n_total = len(idx_list)
        train_end = int(train_frac * n_total)
        val_end = int((train_frac + val_frac) * n_total)
        return idx_list[:train_end], idx_list[train_end:val_end], idx_list[val_end:]

    buckets: Dict[int, List[int]] = {int(k): [] for k in unique.tolist()}
    remainder: List[int] = []
    for idx in idx_list:
        lbl = labels_arr[idx]
        if not np.isfinite(lbl):
            remainder.append(idx)
            continue
        buckets.setdefault(int(lbl), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for values in buckets.values():
        random.shuffle(values)
        n_class = len(values)
        if n_class == 0:
            continue
        train_end = int(train_frac * n_class)
        val_end = int((train_frac + val_frac) * n_class)
        train_idx.extend(values[:train_end])
        val_idx.extend(values[train_end:val_end])
        test_idx.extend(values[val_end:])

    random.shuffle(remainder)
    test_idx.extend(remainder)

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)
    return train_idx, val_idx, test_idx


stratified_split = _lib_stratified_split or _fallback_stratified_split

def _load_real_graphdataset():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    mod_name = "data.mdataset"              # the real module name
    file_path = data_dir / "mdataset.py"

    # 1) Ensure 'data' package exists and points at your repo's data/ dir
    if "data" not in sys.modules:
        pkg = types.ModuleType("data")
        pkg.__path__ = [str(data_dir)]
        sys.modules["data"] = pkg
    else:
        # make sure its __path__ points to your repo
        sys.modules["data"].__path__ = [str(data_dir)]

    # 2) Build spec for the correct qualified name, create module, and
    #    register it in sys.modules BEFORE exec_module (needed for dataclasses)
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module.GraphDataset

from utils.graph_ops import (
    _ensure_edge_attr_np_or_torch as ensure_edge_attr,
    _encode_graph_flex,
    _pool_graph_emb,
)
try:
    from utils.bond_feats import attach_bond_features_from_smiles
except Exception:  # pragma: no cover - optional dependency
    def attach_bond_features_from_smiles(graph, smiles, *_, **__):
        """Fallback when RDKit bond featurisation is unavailable.

        The ``*_, **__`` placeholders mirror the real helper's signature so the
        RDKit-free path accepts optional keyword arguments such as
        ``target_edge_dim`` without raising ``TypeError``.
        """
        return graph


HIGHER_IS_BETTER = {
    "roc_auc",
    "pr_auc",
    "accuracy",
    "acc",
    "ap",
}

_METRIC_ALIASES = {
    "auc": "roc_auc",
    "auroc": "roc_auc",
    "roc_auc": "roc_auc",
    "pr_auc": "pr_auc",
    "ap": "pr_auc",
}


@dataclass
class EvaluationResult:
    """Summary of a single evaluation pass for the Tox21 case study."""

    name: str
    encoder_source: str
    mean_true: float
    mean_random: float
    mean_pred: float
    baseline_means: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_metric: Optional[str] = None
    benchmark_threshold: Optional[float] = None
    met_benchmark: Optional[bool] = None
    manifest_path: Optional[str] = None


@dataclass
class EncoderLoadSummary:
    """Diagnostic information about encoder weight loading."""

    matched_ratio: float
    matched: int
    total: int
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    resized: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    shape_mismatch: List[str] = field(default_factory=list)


@dataclass
class CaseStudyResult:
    """Container returned by :func:`run_tox21_case_study`."""

    evaluations: List[EvaluationResult]
    threshold_rule: Optional["BenchmarkRule"] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    encoder_hash: Optional[str] = None
    baseline_encoder_hash: Optional[str] = None
    encoder_load: Dict[str, Any] = field(default_factory=dict)
    calibrator_state: Optional[Dict[str, Any]] = None
    split_summary: Dict[str, Any] = field(default_factory=dict)


def _to_list(x: Any) -> List[Any]:
    """Convert arrays/tensors/sequences to plain Python lists."""

    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
    except Exception:
        pass

    if hasattr(x, "tolist"):
        try:
            return x.tolist()  # type: ignore[return-value]
        except Exception:
            pass

    return np.asarray(x).tolist()


def _canonical_metric_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return _METRIC_ALIASES.get(str(name).lower(), str(name).lower())


def _metric_is_higher_better(name: Optional[str]) -> bool:
    canonical = _canonical_metric_name(name)
    return canonical in HIGHER_IS_BETTER


def _compute_met_benchmark(metric_name: Optional[str], value: Optional[float], threshold: Optional[float]) -> Optional[bool]:
    if metric_name is None or value is None or threshold is None:
        return None
    if np.isnan(value):  # type: ignore[arg-type]
        return None
    if _metric_is_higher_better(metric_name):
        return bool(value >= threshold)
    return bool(value <= threshold)


def _shape_tuple(value: Any) -> Optional[Tuple[int, ...]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except Exception:
        return None


def _fmt_shape(shape: Optional[Tuple[int, ...]]) -> str:
    if shape is None:
        return "<none>"
    return str(tuple(int(dim) for dim in shape))


def _summarise_keys(
    keys: List[str],
    module_shapes: Dict[str, Optional[Tuple[int, ...]]],
    incoming_shapes: Dict[str, Optional[Tuple[int, ...]]],
    limit: int = 10,
) -> List[str]:
    items: List[str] = []
    for key in keys[:limit]:
        exp_shape = _fmt_shape(module_shapes.get(key))
        inc_shape = _fmt_shape(incoming_shapes.get(key))
        if inc_shape == "<none>":
            items.append(f"{key} (expected={exp_shape})")
        else:
            items.append(f"{key} (expected={exp_shape} got={inc_shape})")
    if len(keys) > limit:
        items.append(f"...(+{len(keys) - limit} more)")
    return items


def _summarise_unexpected(
    keys: List[str],
    incoming_shapes: Dict[str, Optional[Tuple[int, ...]]],
    limit: int = 10,
) -> List[str]:
    items: List[str] = []
    for key in keys[:limit]:
        inc_shape = _fmt_shape(incoming_shapes.get(key))
        if inc_shape == "<none>":
            items.append(key)
        else:
            items.append(f"{key} (shape={inc_shape})")
    if len(keys) > limit:
        items.append(f"...(+{len(keys) - limit} more)")
    return items


def _load_encoder_strict(
    module: torch.nn.Module,
    raw_state: Mapping[str, Any],
    *,
    allow_shape_coercion: bool,
    verify_match_threshold: float,
    hidden_dim: int,
    edge_dim: Optional[int] = None,
    checkpoint_hidden_dim: Optional[Any] = None,
    checkpoint_edge_dim: Optional[Any] = None,
    ckpt_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(raw_state, Mapping):
        raise TypeError("Encoder state must be a mapping for strict load")

    module_state = module.state_dict()
    total = len(module_state)
    module_shapes = {k: _shape_tuple(v) for k, v in module_state.items()}
    incoming_shapes = {k: _shape_tuple(v) for k, v in raw_state.items()}

    unexpected = sorted(k for k in raw_state.keys() if k not in module_state)
    prepared_state: Mapping[str, Any] | Dict[str, Any]
    prepared_state = dict(raw_state)
    resized: List[str] = []
    dropped: List[str] = []
    shape_mismatch: List[str] = []

    if allow_shape_coercion:
        if _prepare_state_dict_for_module is not None:
            prepared_candidate, resized, dropped = _prepare_state_dict_for_module(module, raw_state)
            if prepared_candidate is not None:
                prepared_state = prepared_candidate
        else:
            logger.warning(
                "allow_shape_coercion=True but alignment helper unavailable; proceeding without shape adjustment."
            )
        if resized or dropped:
            logger.warning(
                "Shape coercion applied during encoder load: resized=%d dropped=%d",
                len(resized),
                len(dropped),
            )
    else:
        prepared_state = dict(raw_state)
        for key in list(prepared_state.keys()):
            if key not in module_shapes:
                continue
            exp_shape = module_shapes.get(key)
            inc_shape = incoming_shapes.get(key)
            if exp_shape != inc_shape:
                shape_mismatch.append(f"{key}: {_fmt_shape(inc_shape)} -> {_fmt_shape(exp_shape)}")
                prepared_state.pop(key, None)

    try:
        load_result = module.load_state_dict(prepared_state, strict=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load encoder weights from {ckpt_path or '<checkpoint>'}: {exc}"
        ) from exc

    missing = sorted(getattr(load_result, "missing_keys", []) or [])
    unexpected_after = sorted(getattr(load_result, "unexpected_keys", []) or [])
    unexpected_combined = sorted(dict.fromkeys(unexpected + unexpected_after))

    matched = total - len(missing)
    ratio = matched / total if total else 1.0

    checkpoint_hash = None
    if compute_state_dict_hash is not None:
        try:
            checkpoint_hash = compute_state_dict_hash(raw_state)
        except Exception:
            logger.exception("Failed to compute encoder hash during load")

    logger.info(
        "[enc_load] matched=%.1f%% (matched/total=%d/%d) missing=%d, unexpected=%d; hidden_dim=%s; hash=%s",
        ratio * 100.0,
        matched,
        total,
        len(missing),
        len(unexpected_combined),
        hidden_dim,
        checkpoint_hash or "<unknown>",
    )

    if shape_mismatch and not allow_shape_coercion:
        preview = ", ".join(shape_mismatch[:10])
        model_edge_dim = edge_dim if edge_dim is not None else "<unknown>"
        ck_hidden = checkpoint_hidden_dim if checkpoint_hidden_dim is not None else "<unknown>"
        ck_edge = checkpoint_edge_dim if checkpoint_edge_dim is not None else "<unknown>"
        raise RuntimeError(
            "[enc_load] mismatch: model hidden_dim=%s edge_dim=%s checkpoint hidden_dim=%s edge_dim=%s. "
            "Set allow_shape_coercion=true to override. First mismatches: %s"
            % (hidden_dim, model_edge_dim, ck_hidden, ck_edge, preview)
        )

    if ratio < verify_match_threshold:
        diff_parts: List[str] = []
        if missing:
            diff_parts.append(
                "missing=" + ", ".join(_summarise_keys(missing, module_shapes, incoming_shapes))
            )
        if unexpected_combined:
            diff_parts.append(
                "unexpected="
                + ", ".join(_summarise_unexpected(unexpected_combined, incoming_shapes))
            )
        if shape_mismatch:
            diff_parts.append("shape_mismatch=" + ", ".join(shape_mismatch[:10]))
        raise RuntimeError(
            "Encoder checkpoint mismatch: matched_ratio=%.3f < %.3f (%s). "
            "Set allow_shape_coercion=true to override."
            % (ratio, verify_match_threshold, "; ".join(diff_parts) or "no detail")
        )

    summary: Dict[str, Any] = {
        "matched_ratio": ratio,
        "matched": matched,
        "total": total,
        "missing": missing,
        "unexpected": unexpected_combined,
        "shape_mismatch": shape_mismatch,
        "resized": resized,
        "dropped": dropped,
    }
    if checkpoint_hash:
        summary["hash"] = checkpoint_hash
    if ckpt_path:
        summary["checkpoint"] = os.path.abspath(ckpt_path)
    summary["allow_shape_coercion"] = bool(allow_shape_coercion)
    summary["verify_match_threshold"] = float(verify_match_threshold)
    return summary


def _load_manifest_payload(manifest_path: Optional[str]) -> Dict[str, Any]:
    if not manifest_path:
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        logger.warning("Failed to read encoder manifest from %s", manifest_path, exc_info=True)
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_payload_to_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    hyper = payload.get("hyperparameters")
    manifest_cfg = dict(hyper) if isinstance(hyper, Mapping) else {}
    featurizer_cfg = payload.get("featurizer")
    if isinstance(featurizer_cfg, Mapping):
        for key, value in featurizer_cfg.items():
            manifest_cfg.setdefault(key, value)
    return manifest_cfg


def _load_manifest_config(manifest_path: Optional[str]) -> Dict[str, Any]:
    payload = _load_manifest_payload(manifest_path)
    return _manifest_payload_to_config(payload)


def _resolve_manifest_baseline(
    payload: Mapping[str, Any]
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not isinstance(payload, Mapping):
        return None, None, None, None

    baseline_hash: Optional[str] = None
    hash_source: Optional[str] = None
    hashes = payload.get("hashes")
    if isinstance(hashes, Mapping):
        candidate = hashes.get("encoder")
        if isinstance(candidate, str) and candidate:
            baseline_hash = candidate
            hash_source = "manifest.hashes.encoder"

    if baseline_hash is None:
        manifest_hash = payload.get("encoder_hash")
        if isinstance(manifest_hash, str) and manifest_hash:
            baseline_hash = manifest_hash
            hash_source = "manifest.encoder_hash"

    baseline_path: Optional[str] = None
    path_source: Optional[str] = None
    paths = payload.get("paths")
    if isinstance(paths, Mapping):
        for key in ("encoder", "encoder_symlink", "checkpoint", "encoder_checkpoint"):
            candidate = paths.get(key)
            if isinstance(candidate, str) and candidate:
                baseline_path = candidate
                path_source = f"manifest.paths.{key}"
                break

    if baseline_path is None:
        candidate = payload.get("encoder_checkpoint")
        if isinstance(candidate, str) and candidate:
            baseline_path = candidate
            path_source = "manifest.encoder_checkpoint"

    return baseline_hash, hash_source, baseline_path, path_source


def _extract_state_config(state: Optional[dict]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    cfg = state.get("encoder_cfg")
    return cfg if isinstance(cfg, dict) else {}


def _resolve_threshold_rule(dataset_name: Optional[str], task_name: Optional[str]) -> Optional["BenchmarkRule"]:
    if resolve_metric_threshold is None or dataset_name is None:
        return None
    try:
        return resolve_metric_threshold(dataset_name, task_name)
    except KeyError:
        return None


def _maybe_set_eval(candidate: Any) -> None:
    """Switch a module or nested collection of modules into eval mode."""

    if candidate is None:
        return
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        for member in candidate:
            _maybe_set_eval(member)
        return
    eval_attr = getattr(candidate, "eval", None)
    if callable(eval_attr):
        eval_attr()


def _forward_head(head: Any, batch: torch.Tensor) -> torch.Tensor:
    """Invoke a prediction head safely, tolerating callables and stubs."""

    if head is None:
        return torch.zeros((batch.shape[0], 1), dtype=batch.dtype, device=batch.device)
    if isinstance(head, Sequence) and not isinstance(head, (torch.nn.Module, str, bytes, bytearray)):
        outputs = [_forward_head(member, batch) for member in head]
        stacked = torch.stack(
            [torch.as_tensor(o, dtype=batch.dtype, device=batch.device) for o in outputs], dim=0
        )
        return stacked.mean(dim=0)
    if callable(head):
        result = head(batch)
    else:
        raise RuntimeError("Prediction head is not callable")
    if isinstance(result, (list, tuple)):
        result = (
            result[0]
            if result
            else torch.zeros((batch.shape[0], 1), dtype=batch.dtype, device=batch.device)
        )
    if isinstance(result, torch.Tensor):
        return result
    return torch.as_tensor(result, dtype=batch.dtype, device=batch.device)


def _predict_logits_probs_in_chunks(
    dataset,
    indices: List[int],
    encoder,
    head,
    device: str,
    edge_dim: int,
    batch_size: int = 256,
    diag_hook: Optional[Callable[[int, int], None]] = None,
):
    _maybe_set_eval(encoder)
    _maybe_set_eval(head)
    device_t = torch.device(device) if not isinstance(device, torch.device) else device
    logits_list: List[torch.Tensor] = []
    probs_list: List[torch.Tensor] = []
    batch_count = 0
    molecule_count = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            if not chunk:
                continue
            graph_embs: List[torch.Tensor] = []
            for idx in chunk:
                graph = dataset.graphs[idx]
                graph = ensure_edge_attr(graph, edge_dim, device=device)
                node_emb = _encode_graph_flex(encoder, graph, device_t)
                if isinstance(node_emb, tuple):
                    node_emb = node_emb[0]
                if not isinstance(node_emb, torch.Tensor):
                    node_emb = torch.as_tensor(node_emb, dtype=torch.float32, device=device_t)
                else:
                    node_emb = node_emb.to(device_t)
                node_emb = torch.nan_to_num(node_emb, nan=0.0, posinf=0.0, neginf=0.0)
                graph_emb = _pool_graph_emb(node_emb, graph)
                if not isinstance(graph_emb, torch.Tensor):
                    graph_emb = torch.as_tensor(graph_emb, dtype=torch.float32, device=device_t)
                else:
                    graph_emb = graph_emb.to(device_t)
                if graph_emb.ndim == 0:
                    graph_emb = graph_emb.reshape(1, 1)
                elif graph_emb.ndim == 1:
                    graph_emb = graph_emb.unsqueeze(0)
                graph_embs.append(graph_emb)
            if not graph_embs:
                continue
            batch = torch.cat(graph_embs, dim=0).to(device_t)
            logits = _forward_head(head, batch)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            if logits.ndim == 1 or logits.shape[-1] == 1:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)
            probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
            logits_list.append(logits.detach().cpu())
            probs_list.append(probs.detach().cpu())
            batch_count += 1
            molecule_count += len(chunk)
    if diag_hook is not None:
        try:
            diag_hook(batch_count, molecule_count)
        except Exception:
            logger.debug("Diagnostics hook failed", exc_info=True)
    if not logits_list:
        return torch.empty((0, 1)), torch.empty((0, 1))
    return torch.cat(logits_list, dim=0), torch.cat(probs_list, dim=0)


def _evaluate_case_study(
    dataset,
    encoder,
    head,
    all_labels: np.ndarray,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    triage_pct: float,
    calibrate: bool,
    device: str,
    edge_dim: int,
    seed: int,
    baseline_embeddings: Optional[dict[str, str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
):
    val_idx_arr = np.asarray(val_idx, dtype=int)
    test_idx_arr = np.asarray(test_idx, dtype=int)

    if not isinstance(val_idx_arr, np.ndarray):
        val_idx_arr = np.asarray(list(val_idx_arr), dtype=int)
    if not isinstance(test_idx_arr, np.ndarray):
        test_idx_arr = np.asarray(list(test_idx_arr), dtype=int)

    val_indices = _to_list(val_idx_arr.reshape(-1))
    test_indices = _to_list(test_idx_arr.reshape(-1))

    batch_diag = diagnostics.setdefault("batch_counts", {}) if diagnostics is not None else None

    def _make_batch_hook(split: str) -> Optional[Callable[[int, int], None]]:
        if batch_diag is None:
            return None

        def _hook(batch_count: int, molecule_count: int) -> None:
            batch_diag[split] = {
                "batches": int(batch_count),
                "molecules": int(molecule_count),
            }

        return _hook

    def _as_head_members(candidate: Any) -> List[Any]:
        if isinstance(candidate, torch.nn.Module):
            return [candidate]
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (torch.nn.Module, str, bytes)
        ):
            return list(candidate)
        if candidate is None:
            return []
        return [candidate]

    head_members: List[Any] = _as_head_members(head)
    if not head_members:
        logger.debug("No explicit prediction head supplied; defaulting to stub entry")
        head_members = [None]

    def _mean_tensors(
        tensors: List[torch.Tensor],
        split: str,
        kind: str,
    ) -> torch.Tensor:
        if not tensors:
            return torch.empty((0, 1))
        if len(tensors) == 1:
            return tensors[0]
        lengths = {int(t.shape[0]) for t in tensors}
        if len(lengths) > 1:
            min_len = min(lengths)
            logger.warning(
                "Head ensemble produced mismatched %s lengths on %s split; truncating to %d",
                kind,
                split,
                min_len,
            )
            tensors = [t[:min_len] for t in tensors]
        stacked = torch.stack([t.to(torch.float32) for t in tensors], dim=0)
        return stacked.mean(dim=0)

    per_head_probs: Dict[str, List[torch.Tensor]] = {}

    def _collect_predictions(split: str, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_members: List[torch.Tensor] = []
        probs_members: List[torch.Tensor] = []
        base_hook = _make_batch_hook(split)
        for idx, module in enumerate(head_members):
            hook = base_hook if idx == 0 else None
            logits, probs = _predict_logits_probs_in_chunks(
                dataset,
                indices,
                encoder,
                module,
                device,
                edge_dim,
                diag_hook=hook,
            )
            logits_members.append(logits)
            probs_members.append(probs)
        per_head_probs[split] = list(probs_members)
        return _mean_tensors(logits_members, split, "logits"), _mean_tensors(probs_members, split, "probabilities")

    val_logits, val_probs = _collect_predictions("val", val_indices)
    test_logits, test_probs = _collect_predictions("test", test_indices)

    def _select_positive_probabilities(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 1:
            return arr
        class_dim = int(arr.shape[-1]) if arr.shape else 0
        leading = int(np.prod(arr.shape[:-1])) if arr.ndim > 1 else arr.size
        if class_dim <= 0:
            dtype = arr.dtype if arr.dtype != np.dtype("O") else float
            return np.zeros((leading,), dtype=dtype)
        flat = arr.reshape(leading, class_dim)
        if class_dim == 1:
            return flat[:, 0]
        return flat[:, -1]

    def _select_calibration_features(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        class_dim = int(arr.shape[-1]) if arr.shape else 0
        leading = int(np.prod(arr.shape[:-1])) if arr.ndim > 1 else arr.size
        if class_dim <= 0:
            return np.zeros((leading, 1), dtype=float)
        flat = arr.reshape(leading, class_dim)
        if class_dim == 1:
            return flat.astype(float)
        if class_dim == 2:
            diff = flat[:, 1] - flat[:, 0]
            return diff.reshape(-1, 1)
        return flat[:, [-1]]

    val_logits_np = val_logits.cpu().numpy() if val_logits.numel() else np.zeros((0, 1))
    test_logits_np = test_logits.cpu().numpy() if test_logits.numel() else np.zeros((0, 1))

    val_logits_feat = _select_calibration_features(val_logits_np)
    test_logits_feat = _select_calibration_features(test_logits_np)

    val_probs_np = _select_positive_probabilities(val_probs.cpu().numpy())
    test_probs_np = _select_positive_probabilities(test_probs.cpu().numpy())

    calibrator_info: Dict[str, Any] = {
        "enabled": bool(calibrate),
        "fit_split": "val",
    }
    feature_dim = int(val_logits_feat.shape[1]) if val_logits_feat.ndim == 2 else 1
    if calibrate or (val_logits_np.ndim >= 2 and val_logits_np.shape[-1] > 1):
        calibrator_info["feature_dim"] = feature_dim
    calibrated_probs = test_probs_np
    if calibrate:
        calibrator_info["status"] = "skipped"
        try:
            val_y = all_labels[val_idx_arr].astype(float)
            mask = (~np.isnan(val_y)) & np.isfinite(val_logits_feat[:, 0])
            yv = val_y[mask].astype(int)
            Xv = np.nan_to_num(val_logits_feat[mask], nan=0.0, posinf=1e6, neginf=-1e6)
            calibrator_info["n_candidates"] = int(mask.sum())
            if yv.size > 1 and np.unique(yv).size > 1:
                platt = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
                platt.fit(Xv, yv)
                Xt = np.nan_to_num(test_logits_feat, nan=0.0, posinf=1e6, neginf=-1e6)
                calibrated_probs = platt.predict_proba(Xt)[:, 1]
                calibrator_info.update(
                    {
                        "status": "fitted",
                        "type": "platt",
                        "class_weight": "balanced",
                        "n_samples": int(yv.size),
                        "coef": platt.coef_.reshape(-1).astype(float).tolist(),
                        "intercept": float(platt.intercept_.reshape(-1)[0]),
                    }
                )
            else:
                calibrator_info.update(
                    {
                        "status": "insufficient_variance",
                        "reason": "need_two_classes",
                        "n_samples": int(yv.size),
                    }
                )
        except Exception as exc:  # pragma: no cover - calibration optional
            logger.warning("Calibration skipped due to error: %s", exc)
            calibrated_probs = test_probs_np
            calibrator_info.update({"status": "error", "error": str(exc)})
    else:
        calibrator_info["status"] = "disabled"

    expected_len = int(test_idx_arr.size)
    calibrated_probs = np.asarray(calibrated_probs, dtype=float).reshape(-1)
    if expected_len == 0:
        calibrated_probs = np.zeros((0,), dtype=float)
    else:
        if calibrated_probs.size == 0:
            logger.warning(
                "No calibrated probabilities were produced for %d test molecules; defaulting to zeros.",
                expected_len,
            )
            calibrated_probs = np.zeros(expected_len, dtype=float)
        elif calibrated_probs.size != expected_len:
            logger.warning(
                "Resizing calibrated probabilities from %d to match %d test molecules.",
                calibrated_probs.size,
                expected_len,
            )
            try:
                calibrated_probs = np.resize(calibrated_probs, expected_len).reshape(-1)
            except Exception as exc:
                logger.warning(
                    "Resizing calibrated probabilities failed (%s); padding with zeros instead.",
                    exc,
                )
                calibrated_probs = np.zeros(expected_len, dtype=float)
        if calibrated_probs.size != expected_len:
            logger.warning(
                "Calibrated probabilities still mismatched after resizing; padding with zeros.",
            )
            calibrated_probs = np.zeros(expected_len, dtype=float)

    if triage_pct <= 0:
        k = 0
    else:
        k = max(1, int(triage_pct * test_idx_arr.size))
    k = min(k, test_idx_arr.size) if test_idx_arr.size else 0

    order = np.argsort(-calibrated_probs)[:k] if k > 0 else np.array([], dtype=int)
    mask_pred = np.ones(test_idx_arr.shape[0], dtype=bool)
    if order.size > 0:
        mask_pred[order] = False
    remaining_pred = test_idx_arr[mask_pred]

    def _mean_for_indices(indices: np.ndarray) -> float:
        if indices.size == 0:
            return 0.0
        values = np.asarray(all_labels[indices], dtype=float)
        if values.size == 0:
            return 0.0
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return float("nan")
        return float(values[finite_mask].mean())

    mean_pred = _mean_for_indices(remaining_pred)

    rng = np.random.default_rng(seed)
    rand_choice = rng.choice(test_idx_arr, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    mask_rand = np.ones(test_idx_arr.shape[0], dtype=bool)
    if rand_choice.size > 0:
        mask_rand[np.isin(test_idx_arr, rand_choice)] = False
    remaining_rand = test_idx_arr[mask_rand]
    mean_rand = _mean_for_indices(remaining_rand)

    mean_true = _mean_for_indices(test_idx_arr)

    metrics: Dict[str, float] = {
        # Default metrics to NaN so reporting utilities can drop degenerate
        # evaluations instead of treating them as real scores. These
        # placeholders are overwritten below whenever a metric is computed
        # successfully.
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
    }

    y_true = all_labels[test_idx_arr].astype(float)
    mask_valid = np.isfinite(y_true)
    num_valid = int(mask_valid.sum())
    unique_valid_labels: Optional[np.ndarray] = None
    if num_valid > 0:
        unique_valid_labels = np.unique(y_true[mask_valid])

    if num_valid < 2:
        if num_valid == 0:
            logger.warning("TEST split empty or without finite labels. Skipping AUC/Brier/ECE.")
        else:
            logger.warning("TEST split has fewer than two finite labels. Skipping AUC/Brier/ECE.")
    else:
        y_true_m = y_true[mask_valid]
        if unique_valid_labels.size < 2:
            logger.warning("TEST split degenerate (one class/empty). Skipping AUC/Brier/ECE.")
        else:
            try:
                y_pred_m = calibrated_probs[mask_valid]
            except IndexError:
                logger.warning(
                    "Failed to align calibrated probabilities with %d valid test labels; substituting zeros.",
                    num_valid,
                )
                y_pred_m = np.zeros(num_valid, dtype=float)
            else:
                if y_pred_m.size != num_valid:
                    logger.warning(
                        "Probability array length %d mismatched with %d valid labels; resizing.",
                        y_pred_m.size,
                        num_valid,
                    )
                    if y_pred_m.size == 0:
                        y_pred_m = np.zeros(num_valid, dtype=float)
                    else:
                        y_pred_m = np.resize(y_pred_m, num_valid)

            pp = np.nan_to_num(y_pred_m, nan=0.5, posinf=1.0, neginf=0.0)
            yy = y_true_m.astype(int)
            try:
                roc_auc_val = float(roc_auc_score(yy, pp))
            except Exception:
                roc_auc_val = float("nan")
            if math.isnan(roc_auc_val) and np.unique(yy).size >= 2:
                spread = float(np.nanmax(pp) - np.nanmin(pp)) if pp.size else 0.0
                if spread <= 1e-12:
                    roc_auc_val = 0.5
            metrics["roc_auc"] = roc_auc_val
            try:
                metrics["pr_auc"] = float(average_precision_score(yy, pp))
            except Exception:
                metrics["pr_auc"] = float("nan")
            try:
                metrics["brier"] = float(brier_score_loss(yy, pp))
            except Exception:
                metrics["brier"] = float("nan")
            try:
                metrics["ece"] = float(expected_calibration_error(pp, yy, n_bins=10))
            except Exception:
                metrics["ece"] = float("nan")

    roc_auc_is_nan = math.isnan(metrics["roc_auc"])
    if (
        roc_auc_is_nan
        and unique_valid_labels is not None
        and unique_valid_labels.size == 1
    ):
        _ci_log("roc_auc_nan_single_class", unique_label=float(unique_valid_labels[0]))

    if (
        roc_auc_is_nan
        and unique_valid_labels is not None
        and unique_valid_labels.size >= 2
        and num_valid >= 2
        and per_head_probs.get("test")
    ):
        member_aucs: List[float] = []
        test_size = int(test_idx_arr.size)
        for probs_tensor in per_head_probs.get("test", []):
            try:
                member_probs = _select_positive_probabilities(probs_tensor.cpu().numpy())
            except Exception:
                continue
            member_probs = np.asarray(member_probs, dtype=float).reshape(-1)
            if test_size == 0:
                member_probs = np.zeros((0,), dtype=float)
            elif member_probs.size == 0:
                member_probs = np.zeros(test_size, dtype=float)
            elif member_probs.size != test_size:
                try:
                    member_probs = np.resize(member_probs, test_size).reshape(-1)
                except Exception:
                    member_probs = np.zeros(test_size, dtype=float)
            member_probs = np.nan_to_num(member_probs, nan=0.5, posinf=1.0, neginf=0.0)
            member_valid = member_probs[mask_valid]
            if member_valid.size != num_valid:
                continue
            try:
                auc_val = float(roc_auc_score(y_true_m.astype(int), member_valid))
            except Exception:
                auc_val = float("nan")
            if math.isnan(auc_val) and np.unique(y_true_m.astype(int)).size >= 2:
                spread = (
                    float(np.nanmax(member_valid) - np.nanmin(member_valid))
                    if member_valid.size
                    else 0.0
                )
                if spread <= 1e-12:
                    auc_val = 0.5
            if not math.isnan(auc_val):
                member_aucs.append(auc_val)
        if member_aucs:
            metrics["roc_auc"] = float(sum(member_aucs) / len(member_aucs))
            if diagnostics is not None:
                fallback = diagnostics.setdefault("head_ensemble", {})
                fallback["roc_auc_member_average"] = {
                    "count": len(member_aucs),
                    "values": [float(v) for v in member_aucs],
                }

    baseline_means: Dict[str, float] = {}
    if baseline_embeddings:
        try:
            from sklearn.linear_model import Ridge
        except Exception:  # pragma: no cover - optional dependency
            Ridge = None  # type: ignore[assignment]
        if Ridge is not None:
            train_idx_arr = np.asarray(train_idx, dtype=int)
            val_idx_arr = np.asarray(val_idx, dtype=int)
            train_val_idx = np.concatenate([train_idx_arr, val_idx_arr])
            train_val_idx = np.unique(train_val_idx)
            y_train_val = all_labels[train_val_idx]
            for name, path in baseline_embeddings.items():
                try:
                    if path.lower().endswith(".npy"):
                        X = np.load(path)
                    else:
                        X = pd.read_csv(path).to_numpy()
                except Exception as exc:
                    logger.warning("Failed to load baseline embeddings %s from %s: %s", name, path, exc)
                    continue
                if X.shape[0] != all_labels.shape[0]:
                    raise ValueError(
                        f"Embeddings for {name} have {X.shape[0]} rows, expected {all_labels.shape[0]}"
                    )
                reg = Ridge(alpha=1.0, random_state=seed).fit(X[train_val_idx], y_train_val)
                pred = reg.predict(X)
                pred_test = pred[test_idx_arr]
                top = np.argsort(-pred_test)[:k]
                mask = np.ones(test_idx_arr.shape[0], dtype=bool)
                if top.size > 0:
                    mask[top] = False
                remain = test_idx_arr[mask]
                baseline_means[name] = _mean_for_indices(remain)

    return mean_true, mean_rand, mean_pred, baseline_means, metrics, calibrator_info


def _import_graphdataset():
    from data.mdataset import GraphDataset
    return GraphDataset

def run_tox21_case_study(
    csv_path: str,
    task_name: str,
    smiles_col: str = "smiles",
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 20,
    lr: float = 1e-3,
    pretrain_lr: Optional[float] = None,
    head_lr: Optional[float] = None,
    encoder_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    class_weights: Any = "auto",
    pos_class_weight: Any = None,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: Optional[float] = None,
    gnn_type: str = "mpnn",
    add_3d: bool = False,
    contiguous: bool = False,
    mask_ratio: float = 0.15,
    contrastive: bool = False,
    triage_pct: float = 0.0,
    calibrate: bool = True,
    use_pos_weight: bool = True,
    device: str = "cpu",
    baseline_embeddings: dict[str, str] | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    bf16: bool = False,
    pretrain_time_budget_mins: int = 0,
    finetune_time_budget_mins: int = 0,
    finetune_patience: Optional[int] = None,
    devices: int = 1,
    *,
    dataset_name: str = "tox21",
    encoder_checkpoint: Optional[str] = None,
    encoder_manifest: Optional[str] = None,
    strict_encoder_config: bool = False,
    bf16_head: Optional[bool] = None,
    encoder_source_override: Optional[str] = None,
    evaluation_mode: str = "pretrain_frozen",
    allow_shape_coercion: Optional[bool] = None,
    allow_equal_hash: bool = False,
    verify_match_threshold: float = 0.98,
    cli_hidden_dim_provided: bool = True,
    cli_num_layers_provided: bool = True,
    cli_gnn_type_provided: bool = True,
    full_finetune: Optional[bool] = None,
    freeze_encoder: bool = False,
    unfreeze_top_layers: int = 0,
    tox21_head_batch_size: int = 256,
    head_ensemble_size: int = 1,
    head_scheduler: Optional[str] = None,
    cache_dir: Optional[str] = None,
    explain_mode: Optional[str] = None,
    explain_config: Optional[Dict[str, Any]] = None,
) -> CaseStudyResult:
    """Run the Tox21 case study and return structured evaluation results."""

    logger.info("Running Tox21 case study on %s", csv_path)
    ci_diag = _ci_diag_enabled()
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in {csv_path}")
    if task_name not in df.columns:
        raise ValueError(f"Task column '{task_name}' not found in {csv_path}")

    diagnostics: Dict[str, Any] = {}

    df = df[[smiles_col, task_name]]
    df, label_drop_stats = _sanitize_binary_labels(df, task_name)
    diagnostics["label_filter_counts"] = {
        **label_drop_stats,
        "retained": len(df),
    }
    dropped_total = label_drop_stats.get("dropped_na", 0)
    if label_drop_stats.get("dropped_negative", 0):
        logger.info(
            "Dropping %d rows with negative %s labels (treated as missing)",
            label_drop_stats["dropped_negative"],
            task_name,
        )
    if label_drop_stats.get("dropped_non_binary", 0):
        logger.info(
            "Dropping %d rows with non-binary %s labels",
            label_drop_stats["dropped_non_binary"],
            task_name,
        )
    if dropped_total:
        logger.info(
            "Filtered dataset to %d rows after removing missing %s labels",
            len(df),
            task_name,
        )
    smiles_list = _to_list(df[smiles_col].astype(str))
    labels_list = _to_list(df[task_name].astype(float))
    logger.debug("Loaded %d molecules", len(smiles_list))

    try:
        ensemble_size = int(head_ensemble_size)
    except Exception:
        logger.warning(
            "Failed to parse head_ensemble_size=%s; defaulting to 1",
            head_ensemble_size,
            exc_info=True,
        )
        ensemble_size = 1
    if ensemble_size <= 0:
        logger.warning(
            "Invalid head_ensemble_size=%s; forcing at least one head",
            head_ensemble_size,
        )
        ensemble_size = 1
    diagnostics["head_ensemble_size"] = int(ensemble_size)
    full_finetune_requested = full_finetune
    full_finetune_effective = (
        bool(full_finetune_requested)
        if full_finetune_requested is not None
        else False
    )
    if ci_diag:
        diagnostics["csv_path"] = os.path.abspath(csv_path)
        diagnostics["encoder_checkpoint"] = (
            os.path.abspath(encoder_checkpoint)
            if encoder_checkpoint
            else None
        )
        _ci_log(
            "tox21 encoder checkpoint",
            encoder_checkpoint=diagnostics["encoder_checkpoint"],
            encoder_manifest=os.path.abspath(encoder_manifest)
            if encoder_manifest
            else None,
        )
    else:
        diagnostics["csv_path"] = os.path.abspath(csv_path)
        diagnostics["encoder_checkpoint"] = (
            os.path.abspath(encoder_checkpoint) if encoder_checkpoint else None
        )

    set_seed(seed)

    try:
        pretrain_lr_value = (
            float(pretrain_lr) if pretrain_lr is not None else 1e-4
        )
    except Exception:
        logger.warning(
            "Failed to parse pretrain_lr=%s; defaulting to 1e-4",
            pretrain_lr,
            exc_info=True,
        )
        pretrain_lr_value = 1e-4
    diagnostics["pretrain_lr"] = float(pretrain_lr_value)

    GraphDatasetCls = _load_real_graphdataset()
    gnn_type_lower = (gnn_type or "").lower()
    requires_3d = gnn_type_lower in {"schnet3d", "schnet"}
    requested_add_3d = bool(add_3d)
    effective_add_3d = requested_add_3d or requires_3d

    cache_dir_path: Optional[Path] = None
    if cache_dir:
        try:
            cache_dir_path = Path(cache_dir).expanduser()
            cache_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug(
                "Failed to prepare tox21 cache directory %s", cache_dir, exc_info=True
            )
            cache_dir_path = None

    try:
        head_batch_size = int(tox21_head_batch_size)
    except Exception:
        head_batch_size = 256
    head_batch_size = max(1, head_batch_size)

    manifest_payload = _load_manifest_payload(encoder_manifest)
    manifest_cfg = _manifest_payload_to_config(manifest_payload)

    baseline_hash: Optional[str] = None
    baseline_hash_source: Optional[str] = None
    baseline_path: Optional[str] = None
    baseline_path_source: Optional[str] = None
    if manifest_payload:
        (
            manifest_hash,
            manifest_hash_source,
            manifest_path_candidate,
            manifest_path_source,
        ) = _resolve_manifest_baseline(manifest_payload)
        if manifest_hash:
            baseline_hash = str(manifest_hash)
            baseline_hash_source = manifest_hash_source
        if manifest_path_candidate:
            baseline_path = str(manifest_path_candidate)
            baseline_path_source = manifest_path_source

    if baseline_path is None:
        env_candidate = os.getenv("PRETRAIN_ENCODER_PATH")
        if env_candidate:
            baseline_path = env_candidate
            baseline_path_source = "env.PRETRAIN_ENCODER_PATH"

    if baseline_path is None:
        pretrain_dir_env = os.getenv("PRETRAIN_DIR")
        if pretrain_dir_env:
            baseline_path = os.path.join(pretrain_dir_env, "encoder.pt")
            baseline_path_source = "env.PRETRAIN_DIR"

    if baseline_path:
        candidate_path = str(baseline_path)
        if encoder_manifest and not os.path.isabs(candidate_path):
            manifest_dir = os.path.dirname(os.path.abspath(encoder_manifest))
            candidate_path = os.path.abspath(os.path.join(manifest_dir, candidate_path))
        else:
            candidate_path = os.path.abspath(candidate_path)
        baseline_path = candidate_path

    state_cfg: Dict[str, Any] = {}
    enc_state: Optional[Dict[str, Any]] = None
    encoder_checkpoint_state: Optional[Any] = None
    loaded_head_state: Optional[Dict[str, Any]] = None
    if encoder_checkpoint:
        if safe_load_checkpoint is None:
            raise ImportError("Checkpoint loading utilities are unavailable")
        state, _ = safe_load_checkpoint(
            primary=encoder_checkpoint,
            ckpt_dir=None,
            default_name="encoder.pt",
            map_location=device,
            allow_missing=False,
        )
        encoder_checkpoint_state = state
        state_cfg = _extract_state_config(state)
        if isinstance(state, dict):
            enc_state = state.get("encoder", state)
            loaded_head_state = state.get("head")
        else:
            enc_state = state

    if (
        baseline_hash is None
        and baseline_path
        and safe_load_checkpoint is not None
        and extract_encoder_hash is not None
    ):
        try:
            baseline_state, _ = safe_load_checkpoint(
                primary=baseline_path,
                ckpt_dir=None,
                default_name=os.path.basename(baseline_path) or "encoder.pt",
                map_location=device,
                allow_missing=False,
            )
        except Exception:
            logger.warning(
                "Failed to load baseline encoder checkpoint from %s to compute hash.",
                baseline_path,
                exc_info=True,
            )
        else:
            try:
                baseline_candidate = extract_encoder_hash(baseline_state)
            except Exception:
                baseline_candidate = None
            if baseline_candidate is not None:
                baseline_hash = baseline_candidate
                baseline_hash_source = f"{baseline_path_source or 'checkpoint'}.hash"

    if (
        baseline_hash is None
        and encoder_checkpoint_state is not None
        and extract_encoder_hash is not None
    ):
        try:
            baseline_candidate = extract_encoder_hash(encoder_checkpoint_state)
        except Exception:
            baseline_candidate = None
        if baseline_candidate is not None:
            baseline_hash = baseline_candidate
            baseline_hash_source = "encoder_checkpoint.hash"

    if baseline_hash is not None and isinstance(baseline_hash, bytes):
        baseline_hash = baseline_hash.decode("utf-8", errors="ignore")

    def _maybe_int_value(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except Exception:
            return None

    def _maybe_bool_value(value: Any) -> Optional[bool]:
        if value is None:
            return None
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

    checkpoint_hidden_dim = None
    checkpoint_edge_dim = None
    checkpoint_add_3d = None
    for candidate in (state_cfg.get("hidden_dim"), manifest_cfg.get("hidden_dim")):
        coerced = _maybe_int_value(candidate)
        if coerced is not None:
            checkpoint_hidden_dim = coerced
            break
    for candidate in (state_cfg.get("edge_dim"), manifest_cfg.get("edge_dim")):
        coerced = _maybe_int_value(candidate)
        if coerced is not None:
            checkpoint_edge_dim = coerced
            break
    for candidate in (state_cfg.get("add_3d"), manifest_cfg.get("add_3d")):
        coerced = _maybe_bool_value(candidate)
        if coerced is not None:
            checkpoint_add_3d = coerced
            break

    cache_hidden_marker = hidden_dim if hidden_dim is not None else checkpoint_hidden_dim

    dataset: "GraphDatasetT"
    dataset_cache_path: Optional[Path] = None
    dataset_cache_hit = False
    schema_digest: Optional[str] = None

    def _materialise_dataset(
        add_3d_flag: bool,
        hidden_marker: Optional[int],
    ) -> Tuple["GraphDatasetT", Optional[Path], bool, Optional[str]]:
        dataset_local: "GraphDatasetT" | None = None
        cache_path_local: Optional[Path] = None
        cache_hit_local = False
        schema_digest_local: Optional[str] = None
        if cache_dir_path is not None:
            csv_stem = Path(csv_path).stem or "dataset"
            task_part = str(task_name or dataset_name or "task")
            hidden_key = f"hd{int(hidden_marker)}" if hidden_marker is not None else "hdunk"
            schema_parts = [f"3d{int(add_3d_flag)}", hidden_key]
            fingerprint = "|".join(schema_parts)
            schema_digest_local = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:8]
            cache_name = (
                f"tox21_{csv_stem}_{task_part}_"
                f"{'_'.join(schema_parts)}_h{schema_digest_local}.pkl"
            )
            cache_path_local = cache_dir_path / cache_name
            if cache_path_local.exists():
                try:
                    with cache_path_local.open("rb") as handle:
                        graphs, labels_cached, smiles_cached = pickle.load(handle)
                    dataset_local = GraphDatasetCls(graphs, labels_cached, smiles_cached)
                    cache_hit_local = True
                except Exception:
                    logger.warning(
                        "Failed to load tox21 dataset cache %s; regenerating graphs.",
                        cache_path_local,
                        exc_info=True,
                    )
                    dataset_local = None
        if dataset_local is None:
            dataset_local = GraphDatasetCls.from_smiles_list(
                smiles_list,
                labels=labels_list,
                add_3d=add_3d_flag,
            )
            if cache_dir_path is not None and cache_path_local is not None:
                try:
                    payload = (
                        dataset_local.graphs,
                        dataset_local.labels.tolist() if dataset_local.labels is not None else None,
                        dataset_local.smiles,
                    )
                    tmp_path = cache_path_local.with_suffix(cache_path_local.suffix + ".tmp")
                    with tmp_path.open("wb") as handle:
                        pickle.dump(payload, handle)
                    tmp_path.replace(cache_path_local)
                except Exception:
                    logger.debug(
                        "Failed to materialise tox21 dataset cache at %s",
                        cache_path_local,
                        exc_info=True,
                    )
        return dataset_local, cache_path_local, cache_hit_local, schema_digest_local

    final_hidden_dim: Optional[int] = None
    final_num_layers: Optional[int] = None
    final_gnn_type: Optional[str] = None
    final_dropout: Optional[float] = None
    final_edge_dim: Optional[int] = None
    input_dim = 0
    edge_dim = 0
    attempted_add_3d_materialisation = False

    while True:
        dataset, dataset_cache_path, dataset_cache_hit, schema_digest = _materialise_dataset(
            effective_add_3d,
            cache_hidden_marker,
        )
        if len(dataset) == 0:
            raise ValueError("No valid molecules could be parsed from the dataset.")

        dataset_has_pos = any(getattr(graph, "pos", None) is not None for graph in dataset.graphs)

        try:
            g0 = dataset.graphs[0]
            input_dim = g0.x.shape[1]
            current_edge_dim = int(g0.edge_attr.shape[1]) if getattr(g0, "edge_attr", None) is not None else 0
        except Exception:
            current_edge_dim = 0
            if dataset.graphs and getattr(dataset.graphs[0], "x", None) is not None:
                input_dim = dataset.graphs[0].x.shape[1]

        final_cfg: Dict[str, Any] = {}
        if cli_gnn_type_provided and gnn_type is not None:
            final_cfg["gnn_type"] = gnn_type
        if cli_hidden_dim_provided and hidden_dim is not None:
            final_cfg["hidden_dim"] = hidden_dim
        if cli_num_layers_provided and num_layers is not None:
            final_cfg["num_layers"] = num_layers
        final_cfg["add_3d"] = bool(effective_add_3d)
        final_cfg["edge_dim"] = current_edge_dim
        cli_cfg = final_cfg.copy()
        mismatch_report: Dict[str, Tuple[Any, Any]] = {}
        allow_shape_normalisation = (
            bool(allow_shape_coercion)
            if allow_shape_coercion is not None
            else bool(state_cfg or manifest_cfg)
        )
        auto_shape_normalisation = allow_shape_coercion is None and allow_shape_normalisation
        for source in (state_cfg, manifest_cfg):
            for key in ("gnn_type", "hidden_dim", "num_layers", "add_3d", "edge_dim"):
                value = source.get(key)
                if value is None:
                    continue
                current_value = final_cfg.get(key)
                if (
                    strict_encoder_config
                    and current_value is not None
                    and str(current_value) != str(value)
                ):
                    raise ValueError(
                        f"Encoder configuration mismatch for {key}: CLI={current_value} checkpoint={value}"
                    )
                if key == "add_3d":
                    coerced_bool = _maybe_bool_value(value)
                    if coerced_bool is not None:
                        value = coerced_bool
                elif key == "edge_dim":
                    coerced_int = _maybe_int_value(value)
                    if coerced_int is not None:
                        value = coerced_int
                if allow_shape_normalisation:
                    if current_value is None or str(current_value) == str(value):
                        final_cfg[key] = value
                    else:
                        original = cli_cfg.get(key)
                        if original is not None and key not in mismatch_report:
                            mismatch_report[key] = (original, value)
                        final_cfg[key] = value
                elif current_value is None:
                    final_cfg[key] = value
        if mismatch_report and allow_shape_normalisation:
            details = ", ".join(
                f"{name} {before}→{after}" for name, (before, after) in mismatch_report.items()
            )
            if auto_shape_normalisation:
                logger.info(
                    "Auto shape coercion enabled; normalising encoder configuration from checkpoint metadata: %s",
                    details,
                )
            else:
                logger.info(
                    "Normalising encoder configuration from checkpoint metadata: %s",
                    details,
                )

        hidden_dim_fallback = hidden_dim if hidden_dim is not None else 256
        num_layers_fallback = num_layers if num_layers is not None else 3
        gnn_type_fallback = gnn_type or "mpnn"

        resolved_hidden = final_cfg.get("hidden_dim", hidden_dim_fallback)
        resolved_layers = final_cfg.get("num_layers", num_layers_fallback)
        resolved_gnn = final_cfg.get("gnn_type", gnn_type_fallback)

        final_edge_dim_candidate = final_cfg.get("edge_dim", current_edge_dim)
        if final_edge_dim_candidate is not None:
            try:
                final_edge_dim_candidate = int(final_edge_dim_candidate)
            except Exception:
                final_edge_dim_candidate = current_edge_dim

        desired_add_3d = bool(final_cfg.get("add_3d", effective_add_3d))

        dataset_edge_dim_expected = (
            int(final_edge_dim_candidate) if final_edge_dim_candidate is not None else 0
        )
        edge_dim_mismatch = (
            checkpoint_edge_dim is not None
            and dataset_edge_dim_expected != int(checkpoint_edge_dim)
        )
        if edge_dim_mismatch:
            if allow_shape_coercion is False:
                raise RuntimeError(
                    "Encoder featurizer mismatch: checkpoint edge_dim=%s dataset edge_dim=%s. "
                    "Set allow_shape_coercion=true to override."
                    % (checkpoint_edge_dim, dataset_edge_dim_expected)
                )
            logger.warning(
                "Encoder edge_dim mismatch detected (checkpoint=%s dataset=%s); proceeding with "
                "shape coercion enabled.",
                checkpoint_edge_dim,
                dataset_edge_dim_expected,
            )
            diagnostics.setdefault("warnings", []).append("edge_dim_coerced")
            diagnostics.setdefault("allow_shape_coercion_effective", True)
            if allow_shape_coercion is None:
                diagnostics.setdefault("allow_shape_coercion_auto", True)
        add_3d_mismatch = (
            checkpoint_add_3d is not None and bool(checkpoint_add_3d) != bool(desired_add_3d)
        )
        if add_3d_mismatch:
            if allow_shape_coercion is False:
                raise RuntimeError(
                    "Encoder featurizer mismatch: checkpoint add_3d=%s requested add_3d=%s. "
                    "Set allow_shape_coercion=true to override."
                    % (checkpoint_add_3d, bool(desired_add_3d))
                )
            logger.warning(
                "Encoder add_3d mismatch detected (checkpoint=%s requested=%s); proceeding with "
                "shape coercion enabled.",
                checkpoint_add_3d,
                bool(desired_add_3d),
            )
            diagnostics.setdefault("warnings", []).append("add_3d_coerced")
            diagnostics.setdefault("allow_shape_coercion_effective", True)
            if allow_shape_coercion is None:
                diagnostics.setdefault("allow_shape_coercion_auto", True)

        final_gnn_candidate = str(resolved_gnn)
        final_hidden_candidate = int(resolved_hidden)
        final_layers_candidate = int(resolved_layers)
        final_requires_3d = final_gnn_candidate.lower() in {"schnet3d", "schnet"}
        if final_requires_3d:
            desired_add_3d = True

        if desired_add_3d and not dataset_has_pos:
            if not attempted_add_3d_materialisation and not effective_add_3d:
                logger.info(
                    "Regenerating Tox21 dataset with add_3d=%s to satisfy encoder metadata (previous=%s)",
                    desired_add_3d,
                    effective_add_3d,
                )
                attempted_add_3d_materialisation = True
                effective_add_3d = bool(desired_add_3d)
                cache_hidden_marker = hidden_dim if hidden_dim is not None else final_hidden_candidate
                continue

            if final_requires_3d:
                raise RuntimeError(
                    "3D coordinates were requested but could not be generated. "
                    "Ensure RDKit is installed with 3D conformer support or disable add_3d."
                )

            logger.warning(
                "3D coordinates requested but unavailable; continuing without add_3d features."
            )
            diagnostics.setdefault("warnings", []).append(
                "add_3d_unavailable"
            )
            desired_add_3d = False
            effective_add_3d = False
            final_cfg["add_3d"] = False

        if bool(desired_add_3d) != bool(effective_add_3d):
            logger.info(
                "Updating Tox21 dataset add_3d flag to %s based on encoder metadata (previous=%s)",
                desired_add_3d,
                effective_add_3d,
            )
            effective_add_3d = bool(desired_add_3d)
            cache_hidden_marker = hidden_dim if hidden_dim is not None else final_hidden_candidate
            continue

        final_hidden_dim = final_hidden_candidate
        final_num_layers = final_layers_candidate
        final_gnn_type = final_gnn_candidate
        final_edge_dim = final_edge_dim_candidate
        effective_add_3d = bool(desired_add_3d)
        edge_dim = dataset_edge_dim_expected
        break

    assert final_hidden_dim is not None
    assert final_num_layers is not None
    assert final_gnn_type is not None

    if dropout is not None:
        try:
            final_dropout = float(dropout)
        except Exception:
            final_dropout = float(0.1)
    else:
        final_dropout = 0.1

    if dataset_cache_path is not None:
        logger.info(
            "[cache] selected_cache=%s (hit=%s add_3d=%s hidden_dim=%s schema_hash=%s)",
            dataset_cache_path,
            "yes" if dataset_cache_hit else "no",
            bool(effective_add_3d),
            final_hidden_dim,
            schema_digest if schema_digest is not None else "<unknown>",
        )
        diagnostics["cache"] = {
            "path": str(dataset_cache_path),
            "hit": bool(dataset_cache_hit),
            "add_3d": bool(effective_add_3d),
            "hidden_dim": int(final_hidden_dim),
            "schema_hash": schema_digest,
        }
        if ci_diag:
            _ci_log(
                "tox21 dataset cache",
                path=str(dataset_cache_path),
                hit=bool(dataset_cache_hit),
                add_3d=bool(effective_add_3d),
                hidden_dim=int(final_hidden_dim),
                schema_hash=schema_digest,
            )
    diagnostics["dropout"] = float(final_dropout) if final_dropout is not None else None

    if ci_diag:
        labels_arr = dataset.labels
        task_count = 0
        if labels_arr is not None:
            task_count = 1 if labels_arr.ndim == 1 else int(labels_arr.shape[1])
        diagnostics.update(
            {
                "task_count": task_count,
                "num_molecules": int(len(dataset)),
            }
        )

    if final_gnn_type.lower() in {"schnet3d", "schnet"} and all(
        getattr(g, "pos", None) is None for g in dataset.graphs
    ):
        raise ValueError(
            "SchNet-style encoders require 3D coordinates, but none were generated. "
            "Ensure RDKit is installed with 3D conformer support."
        )

    diagnostics["add_3d_requested"] = requested_add_3d
    diagnostics["add_3d_effective"] = bool(effective_add_3d)

    final_edge_dim_value = final_edge_dim
    gnn_type_lower = final_gnn_type.lower()
    if gnn_type_lower in {
        "gine",
        "gin_edge",
        "gin+edge",
        "edge_mpnn",
        "mpnn_edge",
        "edge",
        "dmpnn",
        "chemprop",
        "attentivefp",
        "attnfp",
    } and (final_edge_dim_value is None or final_edge_dim_value <= 0):
        final_edge_dim_value = 1

    logger.info(
        "[enc_cfg] gnn_type=%s hidden_dim=%s num_layers=%s add_3d=%s edge_dim=%s",
        final_gnn_type,
        final_hidden_dim,
        final_num_layers,
        bool(effective_add_3d),
        final_edge_dim_value,
    )

    diagnostics["encoder_config"] = {
        "gnn_type": final_gnn_type,
        "hidden_dim": int(final_hidden_dim),
        "num_layers": int(final_num_layers),
        "add_3d": bool(effective_add_3d),
        "edge_dim": int(final_edge_dim_value) if final_edge_dim_value is not None else None,
    }

    edge_dim = int(final_edge_dim_value) if final_edge_dim_value is not None else 0
    final_edge_dim = final_edge_dim_value

    target_edge_dim = EDGE_TOTAL_DIM if effective_add_3d else EDGE_BASE_DIM

    for i, graph in enumerate(dataset.graphs):
        smi = getattr(graph, "smiles", None) or (
            getattr(dataset, "smiles", None)[i] if hasattr(dataset, "smiles") else None
        )
        if not smi:
            continue
        graph.smiles = smi
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is None or getattr(edge_attr, "shape", (0, 0))[1] == 0:
            attach_bond_features_from_smiles(
                graph,
                smi,
                target_edge_dim=target_edge_dim,
            )

    all_labels = dataset.labels.astype(float)
    num_total = len(dataset)

    threshold_rule = _resolve_threshold_rule(dataset_name, task_name)


    def _split_summary(indices: List[int]) -> Dict[str, int]:
        if not indices:
            return {"size": 0, "finite": 0, "positives": 0}
        arr = all_labels[np.asarray(indices, dtype=int)]
        mask = np.isfinite(arr)
        finite = int(mask.sum())
        positives = int(np.nansum(arr[mask]))
        return {
            "size": int(len(indices)),
            "finite": finite,
            "positives": positives,
        }

    used_scaffold_split = bool(scaffold_split_indices and _HAS_RDKIT)
    if used_scaffold_split:
        train_split, val_split, test_split = scaffold_split_indices(
            smiles_list,
            train_frac=train_fraction,
            val_frac=val_fraction,
            seed=seed,
        )
        train_idx = _to_list(np.asarray(train_split, dtype=int))
        val_idx = _to_list(np.asarray(val_split, dtype=int))
        test_idx = _to_list(np.asarray(test_split, dtype=int))
        logger.info(
            "Scaffold split: train=%d val=%d test=%d",
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )
    else:
        logger.warning("RDKit scaffold split unavailable; using stratified random split.")
        indices = list(range(num_total))
        rand_state = random.getstate()
        np_state = np.random.get_state()
        train_idx, val_idx, test_idx = stratified_split(
            indices,
            dataset.labels,
            train_frac=train_fraction,
            val_frac=val_fraction,
        )
        random.setstate(rand_state)
        np.random.set_state(np_state)

    def _has_label_diversity(indices: List[int]) -> bool:
        if not indices:
            return False
        arr = all_labels[np.asarray(indices, dtype=int)]
        mask = np.isfinite(arr)
        observed = arr[mask]
        if observed.size < 2:
            return False
        unique = np.unique(observed.astype(int, copy=False))
        return unique.size >= 2

    if used_scaffold_split:
        missing_diversity: List[str] = []
        if not _has_label_diversity(val_idx):
            missing_diversity.append("val")
        if not _has_label_diversity(test_idx):
            missing_diversity.append("test")
        if missing_diversity:
            logger.warning(
                "Scaffold split produced %s split(s) without label diversity; falling back to stratified split.",
                ",".join(missing_diversity),
            )
            indices = list(range(num_total))
            rand_state = random.getstate()
            np_state = np.random.get_state()
            train_idx, val_idx, test_idx = stratified_split(
                indices,
                dataset.labels,
                train_frac=train_fraction,
                val_frac=val_fraction,
            )
            random.setstate(rand_state)
            np.random.set_state(np_state)
            diagnostics["split_strategy"] = "stratified_fallback"
        else:
            diagnostics["split_strategy"] = "scaffold"
    else:
        diagnostics["split_strategy"] = "stratified"

    split_summary = {
        "train": _split_summary(train_idx),
        "val": _split_summary(val_idx),
        "test": _split_summary(test_idx),
    }
    diagnostics["split_counts"] = split_summary
    for split_name, stats in split_summary.items():
        logger.info(
            "Split %s: size=%d finite=%d positives=%d",
            split_name,
            stats.get("size", 0),
            stats.get("finite", 0),
            stats.get("positives", 0),
        )
    if ci_diag:
        _ci_log(
            "tox21 dataset split",
            num_molecules=diagnostics.get("num_molecules", 0),
            task_count=diagnostics.get("task_count", 0),
            train=split_summary.get("train", {}),
            val=split_summary.get("val", {}),
            test=split_summary.get("test", {}),
        )

    input_dim = dataset.graphs[0].x.shape[1]
    edge_dim = 0
    try:
        g0 = dataset.graphs[0]
        if getattr(g0, "edge_attr", None) is not None:
            edge_dim = int(g0.edge_attr.shape[1])
    except Exception:
        edge_dim = 0

    build_encoder_fn = _build_encoder_callable()
    if build_encoder_fn is not None:
        encoder = build_encoder_fn(
            gnn_type=final_gnn_type,
            input_dim=input_dim,
            hidden_dim=final_hidden_dim,
            num_layers=final_num_layers,
            edge_dim=edge_dim,
            dropout=final_dropout,
        )
    else:  # pragma: no cover - exercised only when factory import fails
        logger.warning(
            "models.factory.build_encoder unavailable; falling back to GNNEncoder."
        )
        encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=final_hidden_dim,
            num_layers=final_num_layers,
            gnn_type=final_gnn_type,
        )

    requested_mode = evaluation_mode
    normalized_mode = requested_mode.lower().replace("-", "_")
    requested_fine_tuned_alias = normalized_mode == "fine_tuned"
    display_mode = normalized_mode
    if requested_fine_tuned_alias:
        normalized_mode = "end_to_end"
        display_mode = "fine_tuned"
    valid_modes = {"pretrain_frozen", "frozen_finetuned", "end_to_end"}
    if normalized_mode not in valid_modes:
        logger.warning(
            "Unknown evaluation_mode '%s'; defaulting to pretrain_frozen.",
            evaluation_mode,
        )
        normalized_mode = "pretrain_frozen"
        display_mode = "pretrain_frozen"
    auto_full_finetune = False
    auto_pretrain = False
    if normalized_mode == "end_to_end" and full_finetune_requested is None:
        if not requested_fine_tuned_alias or not encoder_checkpoint:
            reason = "no explicit full_finetune flag"
            if not encoder_checkpoint:
                reason += "; encoder checkpoint not supplied"
            logger.info(
                "Enabling full fine-tuning for end_to_end evaluation (%s).",
                reason,
            )
            full_finetune_effective = True
            auto_full_finetune = True
        else:
            logger.info(
                "Fine-tuned checkpoint supplied without explicit full_finetune flag; keeping encoder frozen.",
            )

    freeze_encoder_requested = bool(freeze_encoder)
    if freeze_encoder_requested and full_finetune_effective:
        logger.info(
            "Freeze-encoder flag set; disabling full fine-tuning for evaluation mode '%s'.",
            display_mode,
        )
        full_finetune_effective = False
        auto_full_finetune = False

    if (
        normalized_mode in {"frozen_finetuned", "end_to_end"}
        and not encoder_checkpoint
        and not full_finetune_effective
    ):
        logger.warning(
            "evaluation_mode '%s' requires a checkpoint; falling back to pretrain_frozen.",
            normalized_mode,
        )
        normalized_mode = "pretrain_frozen"
        display_mode = "pretrain_frozen"

    head_from_checkpoint = isinstance(loaded_head_state, dict) and bool(loaded_head_state)
    train_head_missing = normalized_mode == "end_to_end" and not head_from_checkpoint
    train_probe = normalized_mode != "end_to_end" or train_head_missing
    if full_finetune_effective:
        train_probe = True

    patience_value = finetune_patience
    if patience_value is None:
        env_patience = os.getenv("TOX21_FINETUNE_PATIENCE", "").strip()
        if env_patience:
            try:
                patience_value = int(env_patience)
            except ValueError:
                logger.warning(
                    "Invalid TOX21_FINETUNE_PATIENCE=%s; falling back to default patience.",
                    env_patience,
                )
                patience_value = None
    if patience_value is None:
        patience_value = 12 if full_finetune_effective else 10

    logger.info(
        "Tox21 evaluation configuration: mode=%s head_from_checkpoint=%s train_head=%s full_finetune=%s epochs=%d patience=%s",
        display_mode,
        "yes" if head_from_checkpoint else "no",
        "yes" if train_probe else "no",
        "yes" if full_finetune_effective else "no",
        int(finetune_epochs if train_probe else 0),
        str(patience_value if train_probe else "<skip>"),
    )

    head_trained = False
    head_training_steps = 0.0
    head_loader_batches = 0.0
    head_epoch_batches = 0.0
    random_head_used = False

    should_pretrain = normalized_mode == "pretrain_frozen" and not encoder_checkpoint
    if normalized_mode == "end_to_end" and not encoder_checkpoint:
        should_pretrain = True
        auto_pretrain = True
        if not auto_full_finetune:
            logger.info(
                "No encoder checkpoint supplied for end_to_end evaluation; running JEPA pretraining before fine-tuning.",
            )
    encoder_source = normalized_mode
    eval_name = normalized_mode

    encoder_load_info: Dict[str, Any] = {}
    encoder_hash: Optional[str] = None

    if baseline_path or baseline_hash:
        baseline_path_display = baseline_path or "<unset>"
        logger.info(
            "[tox21] baseline_hash_source=%s baseline_hash=%s baseline_path_source=%s baseline_path=%s",
            baseline_hash_source or "<unknown>",
            baseline_hash or "<unknown>",
            baseline_path_source or "<unknown>",
            baseline_path_display,
        )
    allow_shape_requested: Optional[bool] = allow_shape_coercion
    allow_shape_effective = bool(allow_shape_coercion) if allow_shape_coercion is not None else False
    auto_shape_retry = False

    if enc_state and encoder_checkpoint:
        try:
            encoder_load_info = _load_encoder_strict(
                encoder,
                enc_state,
                allow_shape_coercion=allow_shape_effective,
                verify_match_threshold=float(verify_match_threshold),
                hidden_dim=int(final_hidden_dim),
                edge_dim=edge_dim,
                checkpoint_hidden_dim=checkpoint_hidden_dim,
                checkpoint_edge_dim=checkpoint_edge_dim,
                ckpt_path=encoder_checkpoint,
            )
        except RuntimeError as exc:
            message = str(exc)
            hint_present = "allow_shape_coercion" in message
            if allow_shape_coercion is None and hint_present:
                logger.warning(
                    "Encoder checkpoint shapes mismatched (strict load failed: %s); "
                    "retrying with allow_shape_coercion enabled.",
                    message,
                )
                allow_shape_effective = True
                auto_shape_retry = True
                encoder_load_info = _load_encoder_strict(
                    encoder,
                    enc_state,
                    allow_shape_coercion=True,
                    verify_match_threshold=float(verify_match_threshold),
                    hidden_dim=int(final_hidden_dim),
                    edge_dim=edge_dim,
                    checkpoint_hidden_dim=checkpoint_hidden_dim,
                    checkpoint_edge_dim=checkpoint_edge_dim,
                    ckpt_path=encoder_checkpoint,
                )
            else:
                raise
        encoder_hash = encoder_load_info.get("hash") if isinstance(encoder_load_info, dict) else None
        source_label = (
            str(encoder_source_override)
            if encoder_source_override is not None
            else encoder_source
        )
        logger.info(
            "[tox21] eval_checkpoint_source=%s eval_checkpoint_hash=%s eval_checkpoint_path=%s",
            source_label,
            encoder_hash or "<unknown>",
            os.path.abspath(encoder_checkpoint),
        )
    elif encoder_checkpoint and not enc_state:
        logger.warning(
            "Encoder checkpoint %s contained no weights; using random initialisation",
            encoder_checkpoint,
        )

    fine_tuned_modes = {"frozen_finetuned", "end_to_end"}
    if (
        (full_finetune_effective or normalized_mode in fine_tuned_modes)
        and encoder_hash
        and baseline_hash
        and str(encoder_hash) == str(baseline_hash)
    ):
        if not allow_equal_hash:
            raise RuntimeError(
                "Fine-tuned evaluation aborted: encoder hash matches baseline hash. "
                "Set allow_equal_hash=true to bypass."
            )
        logger.warning(
            "allow_equal_hash=true; continuing despite identical encoder hashes (%s).",
            encoder_hash,
        )

    diagnostics["auto_full_finetune"] = bool(auto_full_finetune)
    diagnostics["auto_pretrain"] = bool(auto_pretrain)
    diagnostics["full_finetune"] = bool(full_finetune_effective)
    diagnostics["full_finetune_requested"] = full_finetune_requested
    diagnostics["freeze_encoder_requested"] = freeze_encoder_requested
    diagnostics["freeze_encoder_effective"] = bool(
        freeze_encoder_requested or not full_finetune_effective
    )
    diagnostics["encoder_load"] = encoder_load_info
    diagnostics["encoder_hash"] = encoder_hash
    diagnostics["baseline_encoder_hash"] = baseline_hash
    diagnostics["baseline_hash_source"] = baseline_hash_source
    diagnostics["baseline_checkpoint"] = baseline_path
    diagnostics["baseline_path_source"] = baseline_path_source
    diagnostics["allow_shape_coercion_requested"] = (
        None if allow_shape_requested is None else bool(allow_shape_requested)
    )
    diagnostics["allow_shape_coercion_effective"] = bool(allow_shape_effective)
    diagnostics["allow_shape_coercion_auto"] = bool(auto_shape_retry)
    diagnostics["verify_match_threshold"] = float(verify_match_threshold)

    def _freeze_encoder_params() -> None:
        params_fn = getattr(encoder, "parameters", None)
        if callable(params_fn):
            try:
                for param in params_fn():
                    param.requires_grad = False
            except Exception:
                pass

    if not should_pretrain:
        _freeze_encoder_params()

    if encoder_source_override:
        encoder_source = str(encoder_source_override)
        eval_name = str(encoder_source_override)
    elif full_finetune_effective:
        encoder_source = "full_finetune"
        eval_name = "full_finetune"
    if should_pretrain:
        ema_encoder_fn = _build_encoder_callable()
        if ema_encoder_fn is None:
            logger.warning(
                "models.factory.build_encoder unavailable during pretraining; "
                "falling back to GNNEncoder.",
            )
            ema_encoder = GNNEncoder(
                input_dim=input_dim,
                hidden_dim=final_hidden_dim,
                num_layers=final_num_layers,
                gnn_type=final_gnn_type,
            )
        else:
            ema_encoder = ema_encoder_fn(
                gnn_type=final_gnn_type,
                input_dim=input_dim,
                hidden_dim=final_hidden_dim,
                num_layers=final_num_layers,
                edge_dim=edge_dim,
            )
        ema_helper = EMA(encoder, decay=0.99)
        predictor = MLPPredictor(embed_dim=final_hidden_dim, hidden_dim=final_hidden_dim * 2)

        train_fn = _train_jepa_callable()
        if contrastive:
            try:
                from training.unsupervised import train_contrastive

                if callable(train_contrastive):
                    train_fn = train_contrastive
                else:
                    raise TypeError("train_contrastive is not callable")
            except Exception:
                logger.warning(
                    "Contrastive pretraining requested but unavailable; falling back to JEPA",
                )

        train_fn(
            dataset=dataset,
            encoder=encoder,
            ema_encoder=ema_encoder,
            predictor=predictor,
            ema=ema_helper,
            epochs=pretrain_epochs,
            batch_size=64,
            mask_ratio=mask_ratio,
            contiguous=contiguous,
            lr=pretrain_lr_value,
            device=device,
            devices=devices,
            reg_lambda=1e-4,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            bf16=bf16,
            time_budget_mins=pretrain_time_budget_mins,
        )
    else:
        logger.info(
            "Skipping JEPA pretraining (mode=%s); using provided encoder weights.",
            normalized_mode,
        )

    if train_probe and should_pretrain and not full_finetune_effective:
        # After optional pretraining we freeze the encoder so that the linear head
        # evaluates the representation quality without further backbone updates.
        _freeze_encoder_params()

    encoder = encoder.to(device)

    try:
        head_lr_value = float(head_lr) if head_lr is not None else float(lr)
    except Exception:
        logger.warning("Failed to parse head_lr=%s; falling back to lr=%s", head_lr, lr)
        head_lr_value = float(lr)
    try:
        encoder_lr_value = float(encoder_lr) if encoder_lr is not None else None
    except Exception:
        logger.warning("Failed to parse encoder_lr=%s; treating as unset", encoder_lr, exc_info=True)
        encoder_lr_value = None
    if full_finetune_effective and encoder_lr_value is None:
        encoder_lr_value = 3e-4

    encoder_lr_display = (
        f"{float(encoder_lr_value):.2e}" if encoder_lr_value is not None else "<unset>"
    )
    logger.info(
        "[tox21] learning rates: pretrain=%.2e head=%.2e encoder=%s",
        float(pretrain_lr_value),
        float(head_lr_value),
        encoder_lr_display,
    )

    weight_decay_value: Optional[float] = None
    if weight_decay is not None:
        try:
            weight_decay_value = float(weight_decay)
        except Exception:
            logger.warning(
                "Failed to coerce weight_decay=%s to float; ignoring weight decay", weight_decay, exc_info=True
            )
            weight_decay_value = None

    def _build_probe_head_and_optimizer() -> Tuple[
        Optional[torch.nn.Module], Optional[torch.optim.Optimizer], Optional[Any]
    ]:
        if weight_decay_value is None:
            return None, None, None
        head = torch.nn.Linear(int(final_hidden_dim), 1)
        optimizer = torch.optim.AdamW(  # type: ignore[call-arg]
            [{"params": head.parameters(), "lr": head_lr_value}],
            lr=head_lr_value,
            weight_decay=weight_decay_value,
        )
        scheduler: Optional[Any] = None
        scheduler_mode = (str(head_scheduler).strip().lower() if head_scheduler else "")
        if scheduler_mode in {"cosine", "cosineannealing", "cosine_annealing"}:
            try:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(finetune_epochs)),
                )
            except Exception:
                logger.warning("Failed to initialise cosine scheduler; continuing without", exc_info=True)
                scheduler = None
        if scheduler_mode:
            diagnostics.setdefault("head_scheduler", scheduler_mode)
        return head, optimizer, scheduler

    bf16_linear = bf16_head if bf16_head is not None else bf16

    train_idx_arr = np.asarray(train_idx, dtype=int)
    val_idx_arr = np.asarray(val_idx, dtype=int)
    test_idx_arr = np.asarray(test_idx, dtype=int)

    manual_class_weight: Optional[Dict[int, float]] = None
    automatic_class_weight = bool(use_pos_weight)
    pos_weight_override: Optional[float] = None
    if pos_class_weight is not None:
        candidate: Any = pos_class_weight
        if isinstance(pos_class_weight, Mapping):
            keys_to_try = []
            if task_name is not None:
                keys_to_try.extend(
                    [
                        str(task_name),
                        str(task_name).lower(),
                        str(task_name).upper(),
                    ]
                )
            keys_to_try.extend(["default", "Default", "*", "all"])
            for key in keys_to_try:
                if key in pos_class_weight:
                    candidate = pos_class_weight[key]
                    break
            else:
                candidate = None
        if candidate is not None:
            try:
                pos_weight_override = float(candidate)
            except Exception:
                logger.warning(
                    "Failed to parse pos_class_weight=%s; ignoring override",
                    candidate,
                    exc_info=True,
                )
                pos_weight_override = None
    diagnostics["pos_class_weight_override"] = (
        float(pos_weight_override) if pos_weight_override is not None else None
    )
    linear_train_fn = _train_linear_head_callable()
    if isinstance(class_weights, str):
        mode = class_weights.strip()
        if not mode:
            automatic_class_weight = bool(use_pos_weight)
        elif mode.startswith("{"):
            try:
                parsed = json.loads(mode)
            except Exception:
                logger.warning("Failed to parse class_weights JSON; falling back to auto", exc_info=True)
            else:
                if isinstance(parsed, Mapping):
                    try:
                        manual_class_weight = {int(k): float(v) for k, v in parsed.items()}
                        automatic_class_weight = False
                    except Exception:
                        logger.warning(
                            "Failed to coerce class_weights JSON mapping; falling back to auto", exc_info=True
                        )
                        manual_class_weight = None
                        automatic_class_weight = bool(use_pos_weight)
                else:
                    logger.warning("class_weights JSON must be a mapping; falling back to auto")
        else:
            mode_l = mode.lower()
            if mode_l in {"auto", "balanced"}:
                automatic_class_weight = True
            elif mode_l in {"none", "off"}:
                automatic_class_weight = False
            else:
                logger.warning("Unknown class_weights mode '%s'; using auto", mode)
                automatic_class_weight = True
    elif isinstance(class_weights, Mapping):
        try:
            manual_class_weight = {int(k): float(v) for k, v in class_weights.items()}
            automatic_class_weight = False
        except Exception:
            logger.warning("Failed to coerce class_weights mapping; falling back to auto", exc_info=True)
            manual_class_weight = None
            automatic_class_weight = bool(use_pos_weight)
    else:
        automatic_class_weight = bool(use_pos_weight)

    if ci_diag:
        diagnostics["split_counts"] = {
            "train": _split_summary(train_idx),
            "val": _split_summary(val_idx),
            "test": _split_summary(test_idx),
        }
        _ci_log(
            "tox21 dataset split",
            num_molecules=diagnostics.get("num_molecules", 0),
            task_count=diagnostics.get("task_count", 0),
            train=diagnostics["split_counts"].get("train", {}),
            val=diagnostics["split_counts"].get("val", {}),
            test=diagnostics["split_counts"].get("test", {}),
        )

    extra_args: Dict[str, Any] = {}
    if manual_class_weight is None and pos_weight_override is not None:
        if pos_weight_override <= 0:
            logger.warning(
                "pos_class_weight must be positive; ignoring override value %.3f",
                pos_weight_override,
            )
        else:
            manual_class_weight = {0: 1.0, 1: float(pos_weight_override)}
            automatic_class_weight = False

    if manual_class_weight is not None:
        extra_args["class_weight"] = manual_class_weight
        diagnostics["class_weight_manual"] = {
            int(k): float(v) for k, v in manual_class_weight.items()
        }
    elif automatic_class_weight and train_idx_arr.size > 0:
        train_labels = all_labels[train_idx_arr]
        mask = ~np.isnan(train_labels)
        n_pos = int(np.nansum(train_labels[mask]))
        n_all = int(mask.sum())
        n_neg = max(0, n_all - n_pos)
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)
            if "pos_weight" in inspect.signature(linear_train_fn).parameters:
                extra_args["pos_weight"] = pos_weight
            elif "class_weight" in inspect.signature(linear_train_fn).parameters:
                extra_args["class_weight"] = {0: 1.0, 1: float(n_neg / max(1, n_pos))}

    explain_mode_norm = str(explain_mode).strip() if explain_mode else None
    explain_cfg_payload: Optional[Dict[str, Any]] = None
    if explain_config:
        explain_cfg_payload = dict(explain_config)
    if explain_cfg_payload is not None:
        explain_cfg_payload.setdefault("task_name", task_name)

    clf_metrics: Dict[str, Any]
    if train_probe:
        # Linear-probe modes train a fresh classifier on top of a frozen encoder to
        # measure representation quality without updating the backbone.
        linear_kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "encoder": encoder,
            "task_type": "classification",
            "epochs": finetune_epochs,
            "lr": head_lr_value,
            "batch_size": head_batch_size,
            "device": device,
            "devices": devices,
            "patience": int(patience_value),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
            "bf16": bf16_linear,
            "time_budget_mins": finetune_time_budget_mins,
            "use_scaffold": bool(scaffold_split_indices and _HAS_RDKIT),
            "freeze_encoder": not full_finetune_effective,
            "head_lr": head_lr_value,
            "encoder_lr": encoder_lr_value,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
            "enable_batch_autoscale": False,
            "unfreeze_top_layers": int(unfreeze_top_layers),
            "explain_mode": explain_mode_norm,
            "explain_config": explain_cfg_payload,
            **extra_args,
        }
        if full_finetune_effective:
            linear_kwargs["early_stop_metric"] = "val_auc"

        ensemble_heads: List[torch.nn.Module] = []
        ensemble_metrics: List[Dict[str, Any]] = []

        def _aggregate_stat(key: str) -> float:
            values: List[float] = []
            for payload in ensemble_metrics:
                raw = payload.get(key)
                if raw is None:
                    continue
                try:
                    values.append(float(raw))
                except Exception:
                    continue
            if not values:
                return 0.0
            return float(sum(values) / len(values))

        for member in range(ensemble_size):
            run_kwargs = dict(linear_kwargs)
            head_seed, opt_seed, sched_seed = (None, None, None)
            if train_probe:
                head_seed, opt_seed, sched_seed = _build_probe_head_and_optimizer()
            if head_seed is not None:
                run_kwargs["head"] = head_seed
            if opt_seed is not None:
                run_kwargs["optimizer"] = opt_seed
            if sched_seed is not None:
                run_kwargs["scheduler"] = sched_seed
            if member > 0:
                set_seed(int(seed) + member)
            metrics_payload = linear_train_fn(**run_kwargs)
            ensemble_metrics.append(metrics_payload)

            head_obj = metrics_payload.get("head")
            if head_obj is None:
                fallback_head = run_kwargs.get("head")
                if fallback_head is not None:
                    logger.warning(
                        "train_linear_head did not return a head module; falling back to the provided head instance."
                    )
                    head_obj = fallback_head
            if isinstance(head_obj, torch.nn.parallel.DistributedDataParallel):
                head_obj = head_obj.module
            if head_obj is None:
                raise RuntimeError("train_linear_head did not return a head module")
            head_obj = head_obj.to(device)
            ensemble_heads.append(head_obj)

        set_seed(int(seed))

        if not ensemble_heads:
            raise RuntimeError("No heads were trained during Tox21 fine-tuning")

        head = ensemble_heads[0] if len(ensemble_heads) == 1 else ensemble_heads
        last_metrics = dict(ensemble_metrics[-1]) if ensemble_metrics else {}
        last_metrics["head"] = ensemble_heads[-1]
        clf_metrics = last_metrics
        head_trained = True
        head_training_steps = _aggregate_stat("train/batches")
        head_loader_batches = _aggregate_stat("train/loader_batches")
        head_epoch_batches = _aggregate_stat("train/epoch_batches")
        diagnostics["head_ensemble_members_trained"] = len(ensemble_heads)
        metrics_summary: List[Dict[str, float]] = []
        for payload in ensemble_metrics:
            numeric: Dict[str, float] = {}
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    numeric[key] = float(value)
            if numeric:
                metrics_summary.append(numeric)
        if metrics_summary:
            diagnostics["head_ensemble_metrics"] = metrics_summary
    else:
        out_dim = 1
        in_dim = final_hidden_dim
        if isinstance(loaded_head_state, dict):
            for key, value in loaded_head_state.items():
                if key.endswith("weight") and getattr(value, "ndim", 0) == 2:
                    out_dim, in_dim = value.shape
                    break
        head = torch.nn.Linear(int(in_dim), int(out_dim))
        if isinstance(loaded_head_state, dict) and loaded_head_state:
            if load_state_dict_forgiving is not None:
                load_state_dict_forgiving(head, loaded_head_state)
            else:
                head.load_state_dict(loaded_head_state, strict=False)
        elif normalized_mode == "end_to_end":
            random_head_used = True
            logger.warning(
                "End-to-end mode requested but head weights were missing; using a randomly initialised head."
            )
        head = head.to(device)
        clf_metrics = {"head": head}

    encoder.eval()

    def _eval_head_modules(candidate: Any) -> None:
        if isinstance(candidate, torch.nn.Module):
            candidate.eval()
            return
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (torch.nn.Module, str, bytes)
        ):
            for member in candidate:
                if isinstance(member, torch.nn.Module):
                    member.eval()

    _eval_head_modules(head)

    (
        mean_true,
        mean_rand,
        mean_pred,
        baseline_means,
        metrics,
        calibrator_info,
    ) = _evaluate_case_study(
        dataset=dataset,
        encoder=encoder,
        head=head,
        all_labels=all_labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        triage_pct=triage_pct,
        calibrate=calibrate,
        device=device,
        edge_dim=edge_dim,
        seed=seed,
        baseline_embeddings=baseline_embeddings,
        diagnostics=diagnostics if ci_diag else None,
    )

    diagnostics["calibrator"] = calibrator_info
    logger.info(
        "Calibration summary: enabled=%s status=%s split=%s samples=%s",
        calibrator_info.get("enabled"),
        calibrator_info.get("status"),
        calibrator_info.get("fit_split"),
        calibrator_info.get("n_samples", calibrator_info.get("n_candidates")),
    )

    benchmark_metric = threshold_rule.metric if threshold_rule is not None else None
    canonical_metric = _canonical_metric_name(benchmark_metric)
    metric_value = metrics.get(canonical_metric) if canonical_metric else None

    if ci_diag:
        _ci_log(
            "tox21 inference",
            encoder_checkpoint=os.path.abspath(encoder_checkpoint)
            if encoder_checkpoint
            else None,
            batches=diagnostics.get("batch_counts", {}),
        )
        _ci_log("tox21 metrics", metrics=metrics)
    met_benchmark = _compute_met_benchmark(
        benchmark_metric,
        metric_value,
        threshold_rule.threshold if threshold_rule is not None else None,
    )

    def _fmt_metric(value: Any) -> str:
        try:
            num = float(value)
        except Exception:
            return "<nan>"
        if math.isnan(num):
            return "<nan>"
        return f"{num:.4f}"

    roc_auc_val = metrics.get("roc_auc")
    pr_auc_val = metrics.get("pr_auc")
    logger.info(
        "Tox21 evaluation summary: mode=%s head_trained=%s steps=%.1f roc_auc=%s pr_auc=%s",
        display_mode,
        "yes" if head_trained else "no",
        head_training_steps,
        _fmt_metric(roc_auc_val),
        _fmt_metric(pr_auc_val),
    )
    if random_head_used:
        logger.warning(
            "Tox21 evaluation used a randomly initialised head; metrics may be unreliable."
        )

    evaluation = EvaluationResult(
        name=eval_name,
        encoder_source=encoder_source,
        mean_true=float(mean_true),
        mean_random=float(mean_rand),
        mean_pred=float(mean_pred),
        baseline_means={k: float(v) for k, v in baseline_means.items()},
        metrics={k: float(v) for k, v in metrics.items()},
        benchmark_metric=benchmark_metric,
        benchmark_threshold=(
            float(threshold_rule.threshold) if threshold_rule is not None else None
        ),
        met_benchmark=met_benchmark,
        manifest_path=encoder_manifest if encoder_checkpoint else None,
    )

    return CaseStudyResult(
        evaluations=[evaluation],
        threshold_rule=threshold_rule,
        diagnostics=diagnostics,
        encoder_hash=encoder_hash,
        baseline_encoder_hash=baseline_hash,
        encoder_load=dict(encoder_load_info),
        calibrator_state=calibrator_info,
        split_summary=split_summary,
    )


if __name__ == "__main__":
    csv = "samples/tox21_mini.csv"
    if os.path.exists(csv):
        result = run_tox21_case_study(
            csv_path=csv,
            task_name="NR-AR",
            pretrain_epochs=1,
            finetune_epochs=1,
            triage_pct=0.0,
        )
        primary = result.evaluations[0]
        logger.info("Mean true toxicity: %s", primary.mean_true)
        logger.info("Mean toxicity after random exclusion: %s", primary.mean_random)
        logger.info("Mean toxicity after model exclusion: %s", primary.mean_pred)
        for name, val in primary.baseline_means.items():
            logger.info("Mean toxicity after %s exclusion: %s", name, val)
    else:
        logger.error("Tox21 sample CSV not found: %s", csv)

