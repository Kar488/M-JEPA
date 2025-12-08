# flake8: noqa
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import log_effective_gnn

# Ensure PyTorch uses filesystem-backed shared memory objects instead of file
# descriptors for inter-process tensor sharing.  The default "file_descriptor"
# strategy opens a unique FD for every shared tensor created by DataLoader
# workers, which eventually exhausts the per-process file descriptor quota
# during long fine-tuning runs.  Switching to "file_system" prevents the
# descriptor leak that triggers "Too many open files" crashes.
torch.multiprocessing.set_sharing_strategy("file_system")

from data.mdataset import GraphData, GraphDataset
from data.scaffold_split import scaffold_split_indices
from utils.graph_ops import _encode_graph
from utils.pooling import global_mean_pool
from training.supervised import stratified_split

logger = logging.getLogger(__name__)

_GNN_TYPES_REQUIRING_3D = {"schnet3d", "schnet"}

_TOX21_TASKS = {
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
}

_TOX21_EPOCH_FLOOR_DEFAULT = 10


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


def _coerce_float_like(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _resolve_labeled_dataset_source(args: argparse.Namespace) -> str:
    """Return the path to the labelled dataset, normalising dir/file inputs."""

    labeled_dir_raw = getattr(args, "labeled_dir", None)
    labeled_csv_raw = getattr(args, "labeled_csv", None)

    resolved_csv: Optional[Path] = None
    resolved_dir: Optional[Path] = None

    if labeled_csv_raw:
        try:
            candidate = Path(str(labeled_csv_raw)).expanduser().resolve()
        except Exception:
            candidate = Path(str(labeled_csv_raw))
        if candidate.is_file():
            resolved_csv = candidate
            if not labeled_dir_raw or Path(str(labeled_dir_raw)).resolve() == candidate:
                resolved_dir = candidate.parent
        else:
            logger.warning("Declared --labeled-csv=%s does not exist or is not a file", labeled_csv_raw)

    if labeled_dir_raw:
        try:
            candidate_dir = Path(str(labeled_dir_raw)).expanduser().resolve()
        except Exception:
            candidate_dir = Path(str(labeled_dir_raw))
        if candidate_dir.is_file():
            if resolved_csv is None:
                resolved_csv = candidate_dir
            resolved_dir = candidate_dir.parent
        elif candidate_dir.is_dir():
            resolved_dir = candidate_dir
        else:
            logger.warning("Declared --labeled-dir=%s does not exist", labeled_dir_raw)

    if resolved_dir is not None:
        setattr(args, "labeled_dir", str(resolved_dir))
    if resolved_csv is not None:
        setattr(args, "labeled_csv", str(resolved_csv))

    if resolved_csv is not None:
        return str(resolved_csv)
    if resolved_dir is not None:
        return str(resolved_dir)

    raise FileNotFoundError(
        f"Unable to resolve labelled dataset from --labeled-dir={labeled_dir_raw!r} "
        f"or --labeled-csv={labeled_csv_raw!r}"
    )


def _bestcfg_env_skip_keys() -> set[str]:
    raw = os.getenv("BESTCFG_SKIP", "")
    skip = {token.strip() for token in raw.replace(",", " ").split() if token.strip()}
    if os.getenv("BESTCFG_NO_EPOCHS") == "1":
        skip.update({"pretrain_epochs", "finetune_epochs", "epochs"})
    return skip


def _discover_best_config_path(args: argparse.Namespace) -> Optional[Path]:
    candidates: List[Path] = []
    for attr in ("best_config_path", "best_config", "best_config_json"):
        val = getattr(args, attr, None)
        if val:
            candidates.append(Path(str(val)))

    env_hints = [
        os.getenv("BEST_CONFIG_PATH"),
        os.getenv("FINETUNE_BEST_CONFIG"),
        os.getenv("TRAIN_JEPA_BEST_CONFIG"),
        os.getenv("TOX21_BEST_CONFIG"),
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
    skip_env = _bestcfg_env_skip_keys()

    hidden_val = _coerce_int_like(_extract_bestcfg_value(raw, "hidden_dim"))
    if hidden_val is not None and "hidden_dim" not in skip_env:
        overrides["hidden_dim"] = hidden_val

    layers_val = _coerce_int_like(_extract_bestcfg_value(raw, "num_layers"))
    if layers_val is not None and "num_layers" not in skip_env:
        overrides["num_layers"] = layers_val

    gnn_raw = _extract_bestcfg_value(raw, "gnn_type")
    if isinstance(gnn_raw, str) and gnn_raw.strip() and "gnn_type" not in skip_env:
        overrides["gnn_type"] = gnn_raw.strip()

    add_val = _coerce_bool_like(_extract_bestcfg_value(raw, "add_3d"))
    if add_val is not None and "add_3d" not in skip_env:
        overrides["add_3d"] = add_val

    devices_val = _coerce_int_like(_extract_bestcfg_value(raw, "devices"))
    if devices_val is not None and "devices" not in skip_env:
        overrides["devices"] = devices_val

    num_workers_val = _coerce_int_like(_extract_bestcfg_value(raw, "num_workers"))
    if num_workers_val is not None and "num_workers" not in skip_env:
        overrides["num_workers"] = num_workers_val

    prefetch_val = _coerce_int_like(_extract_bestcfg_value(raw, "prefetch_factor"))
    if prefetch_val is not None and "prefetch_factor" not in skip_env:
        overrides["prefetch_factor"] = prefetch_val

    pin_memory_val = _coerce_bool_like(_extract_bestcfg_value(raw, "pin_memory"))
    if pin_memory_val is not None and "pin_memory" not in skip_env:
        overrides["pin_memory"] = pin_memory_val

    persistent_val = _coerce_bool_like(_extract_bestcfg_value(raw, "persistent_workers"))
    if persistent_val is not None and "persistent_workers" not in skip_env:
        overrides["persistent_workers"] = persistent_val

    bf16_val = _coerce_bool_like(_extract_bestcfg_value(raw, "bf16"))
    if bf16_val is not None and "bf16" not in skip_env:
        overrides["bf16"] = bf16_val

    batch_val = _coerce_int_like(_extract_bestcfg_value(raw, "finetune_batch_size"))
    if batch_val is not None and "finetune_batch_size" not in skip_env:
        overrides["batch_size"] = batch_val

    epoch_raw = _extract_bestcfg_value(raw, "finetune_epochs")
    epoch_val = _coerce_int_like(epoch_raw)
    if epoch_val is not None:
        if "finetune_epochs" in skip_env:
            source = str(path)
            logger.info(
                "Skipping Phase-2 best_config finetune_epochs override (%s from %s) due to BESTCFG_SKIP/BESTCFG_NO_EPOCHS.",
                epoch_val,
                source,
            )
        else:
            overrides["epochs"] = epoch_val

    lr_val = _coerce_float_like(_extract_bestcfg_value(raw, "lr"))
    if lr_val is None:
        lr_val = _coerce_float_like(_extract_bestcfg_value(raw, "learning_rate"))
        lr_key = "learning_rate"
    else:
        lr_key = "lr"
    if lr_val is not None and lr_key not in skip_env:
        overrides["lr"] = lr_val

    return overrides, path


def _apply_best_config_overrides(args: argparse.Namespace) -> None:
    best_overrides, best_path = _load_best_config_overrides(args)
    if not best_overrides:
        return

    inherited: List[str] = []

    def _apply(dest: str, value: Any, *, flags: Tuple[str, ...] = ()) -> None:
        if value is None:
            return
        if flags and _flag_was_provided(flags):
            return
        current = getattr(args, dest, None)
        if isinstance(value, bool):
            coerced: Any = bool(value)
        else:
            coerced = value
        if current == coerced:
            return
        if dest == "epochs":
            current_int = _coerce_int_like(current)
            override_int = _coerce_int_like(coerced)
            if (
                current_int is not None
                and override_int is not None
                and override_int < current_int
            ):
                source = str(best_path) if best_path is not None else "best_config"
                logger.info(
                    "Skipping Phase-2 best_config epochs override (%s from %s) because baseline uses %s epochs.",
                    override_int,
                    source,
                    current_int,
                )
                return
        setattr(args, dest, coerced)
        inherited.append(f"{dest}={coerced}")

    _apply("gnn_type", best_overrides.get("gnn_type"), flags=("--gnn-type", "--gnn_type"))
    _apply("hidden_dim", best_overrides.get("hidden_dim"), flags=("--hidden-dim", "--hidden_dim"))
    _apply("num_layers", best_overrides.get("num_layers"), flags=("--num-layers", "--num_layers"))
    _apply("add_3d", best_overrides.get("add_3d"), flags=("--add-3d", "--add_3d"))
    _apply("devices", best_overrides.get("devices"), flags=("--devices",))
    _apply("num_workers", best_overrides.get("num_workers"), flags=("--num-workers", "--num_workers"))
    _apply("prefetch_factor", best_overrides.get("prefetch_factor"), flags=("--prefetch-factor", "--prefetch_factor"))
    _apply("pin_memory", best_overrides.get("pin_memory"), flags=("--pin-memory", "--pin_memory"))
    _apply("persistent_workers", best_overrides.get("persistent_workers"), flags=("--persistent-workers", "--persistent_workers"))
    _apply("bf16", best_overrides.get("bf16"), flags=("--bf16",))
    _apply("batch_size", best_overrides.get("batch_size"), flags=("--batch-size", "--batch_size"))
    _apply("epochs", best_overrides.get("epochs"), flags=("--epochs",))
    _apply("lr", best_overrides.get("lr"), flags=("--lr",))

    if inherited and best_path is not None:
        logger.info(
            "Inheriting Phase-2 best_config overrides from %s: %s",
            best_path,
            ", ".join(inherited),
        )


def _maybe_enforce_tox21_epoch_floor(
    args: argparse.Namespace, label_columns: Iterable[str]
) -> None:
    labels = [label for label in label_columns if label]
    if not labels:
        return
    if not all(label in _TOX21_TASKS for label in labels):
        return
    dataset_hint = str(getattr(args, "labeled_dir", "") or "").lower()
    if "tox21" not in dataset_hint:
        return

    requested_epochs = _coerce_int_like(getattr(args, "epochs", None))
    if requested_epochs is None:
        return
    if _flag_was_provided(("--epochs",)):
        return

    env_floor = os.getenv("TOX21_FINETUNE_EPOCH_FLOOR") or os.getenv(
        "TOX21_FINETUNE_MIN_EPOCHS"
    )
    floor_value = _coerce_int_like(env_floor) if env_floor else None
    if floor_value is None or floor_value <= 0:
        floor_value = _TOX21_EPOCH_FLOOR_DEFAULT

    if requested_epochs >= floor_value:
        return

    setattr(args, "epochs", floor_value)
    setattr(args, "_tox21_epoch_floor_applied", True)
    setattr(args, "_tox21_epoch_floor_value", floor_value)
    setattr(args, "_tox21_epoch_requested", requested_epochs)
    logger.info(
        "[finetune] raising Tox21 epoch floor to %d (requested %s)",
        floor_value,
        requested_epochs,
    )

def _split_label_list(raw: str) -> List[str]:
    tokens: List[str] = []
    normalised = raw.replace("\n", ",")
    for token in normalised.split(","):
        entry = token.strip()
        if entry:
            tokens.append(entry)
    return tokens


def _resolve_label_columns(args: argparse.Namespace) -> List[str]:
    env_spec = str(os.getenv("FINETUNE_LABEL_COLS", "") or "").strip()
    candidates = _split_label_list(env_spec) if env_spec else []

    if not candidates:
        raw_label = getattr(args, "label_col", None)
        if isinstance(raw_label, str):
            candidates = _split_label_list(raw_label)
        elif raw_label:
            candidates = [str(raw_label)]

    expanded: List[str] = []
    for item in candidates:
        lowered = item.strip().lower()
        if lowered in {"all", "*", "__all__", "tox21"}:
            expanded.extend(sorted(_TOX21_TASKS))
        else:
            expanded.append(item.strip())

    if not expanded:
        single = getattr(args, "label_col", None)
        return [single] if single else []

    seen: List[str] = []
    for entry in expanded:
        if entry and entry not in seen:
            seen.append(entry)
    return seen


def _sanitize_dataset_labels(dataset) -> tuple[Any, Dict[str, int]]:
    """Drop rows with missing/sentinel labels from a GraphDataset.

    Treat negative or non-finite labels as missing (common when ``-1`` encodes
    "no label" in MoleculeNet/Tox21 CSVs) and return a filtered copy alongside
    drop statistics.
    """

    labels_attr = getattr(dataset, "labels", None)
    graphs_attr = getattr(dataset, "graphs", None)

    stats = {"size_before": len(dataset), "dropped_negative": 0, "dropped_nonfinite": 0}

    if labels_attr is None or graphs_attr is None:
        stats["size_after"] = len(dataset)
        stats["dropped_total"] = 0
        return dataset, stats

    try:
        labels_arr = np.asarray(labels_attr, dtype=float)
    except Exception:
        logger.debug("Skipping label sanitization; unable to coerce labels array", exc_info=True)
        stats["size_after"] = len(dataset)
        stats["dropped_total"] = 0
        return dataset, stats

    if labels_arr.ndim > 1:
        if labels_arr.shape[1] == 1:
            labels_arr = labels_arr[:, 0]
        else:
            logger.debug(
                "Skipping label sanitization for multi-dimensional labels with shape %s", labels_arr.shape
            )
            stats["size_after"] = len(dataset)
            stats["dropped_total"] = 0
            return dataset, stats

    nonfinite_mask = ~np.isfinite(labels_arr)
    negative_mask = labels_arr < 0
    drop_mask = nonfinite_mask | negative_mask

    stats["dropped_nonfinite"] = int(nonfinite_mask.sum())
    stats["dropped_negative"] = int(negative_mask.sum())
    stats["dropped_total"] = int(drop_mask.sum())

    if stats["dropped_total"] <= 0:
        stats["size_after"] = len(dataset)
        return dataset, stats

    keep_indices = [idx for idx, flag in enumerate(~drop_mask) if flag]
    graphs_clean = [graphs_attr[idx] for idx in keep_indices]
    smiles_attr = getattr(dataset, "smiles", None)
    smiles_clean = [smiles_attr[idx] for idx in keep_indices] if smiles_attr else None
    labels_clean = labels_arr[keep_indices]

    cleaned = GraphDataset(graphs_clean, labels_clean, smiles_clean)
    stats["size_after"] = len(cleaned)
    return cleaned, stats

stage_config: Dict[str, Any] = {}
soft_timeout_exit_code = int(os.environ.get("LINEAR_HEAD_SOFT_TIMEOUT_EXIT", "86"))


def _stage_outputs_dir() -> Optional[Path]:
    stage_dir = os.getenv("STAGE_OUTPUTS_DIR")
    if not stage_dir:
        return None
    try:
        path = Path(stage_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def _record_finetune_stage_outputs(payload: Dict[str, Any]) -> None:
    stage_dir = _stage_outputs_dir()
    if stage_dir is None:
        return
    try:
        suffix = os.getenv("FINETUNE_STAGE_SUFFIX", "").strip()
        if suffix:
            filename = f"finetune_{_sanitize_alias(suffix)}.json"
        else:
            filename = "finetune.json"
        out_path = stage_dir / filename
        tmp_path = out_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(out_path)
    except Exception:
        logger.warning("Failed to write finetune stage outputs", exc_info=True)


def _namespace_snapshot(ns: argparse.Namespace) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for key, value in vars(ns).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            snapshot[key] = value
        elif isinstance(value, (list, tuple)):
            filtered = []
            for item in value:
                if isinstance(item, (str, int, float, bool)) or item is None:
                    filtered.append(item)
            snapshot[key] = filtered
        elif isinstance(value, dict):
            filtered_dict: Dict[str, Any] = {}
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (str, int, float, bool)) or sub_val is None:
                    filtered_dict[sub_key] = sub_val
            if filtered_dict:
                snapshot[key] = filtered_dict
    return snapshot


def _write_fanout_manifest(base_dir: Path, entries: List[Dict[str, Any]]) -> Optional[Path]:
    if not entries:
        return None
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("Unable to create fan-out directory %s", base_dir, exc_info=True)
        return None

    manifest_path = base_dir / "fanout_manifest.json"
    try:
        tmp_path = manifest_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump({"tasks": entries}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(manifest_path)
    except Exception:
        logger.warning("Failed to write finetune fan-out manifest", exc_info=True)
        return None
    return manifest_path


def _sanitize_alias(raw: str) -> str:
    safe = [c if c.isalnum() or c in {"-", "_", "."} else "_" for c in raw]
    text = "".join(safe).strip("._")
    return text or "run"


def _maybe_enable_add_3d(args: argparse.Namespace) -> bool:
    """Ensure SchNet-based encoders always receive 3-D coordinates."""

    gnn_type = str(getattr(args, "gnn_type", "") or "").lower()
    requires_3d = gnn_type in _GNN_TYPES_REQUIRING_3D
    if requires_3d and not getattr(args, "add_3d", False):
        logger.info(
            "GNN '%s' requires 3D coordinates; enabling --add-3d automatically.",
            getattr(args, "gnn_type", gnn_type),
        )
        setattr(args, "add_3d", True)
    return requires_3d


def _ensure_dataset_has_pos(dataset) -> None:
    """Validate that a dataset provides ``pos`` coordinates when required."""

    graphs = getattr(dataset, "graphs", None)
    if not graphs:
        return

    for idx, graph in enumerate(graphs):
        pos = getattr(graph, "pos", None)
        if pos is None:
            num_nodes = 0
            if hasattr(graph, "num_nodes"):
                try:
                    num_nodes = int(graph.num_nodes())
                except Exception:
                    num_nodes = 0
            if not num_nodes:
                x_field = getattr(graph, "x", None)
                try:
                    num_nodes = int(len(x_field)) if x_field is not None else 0
                except Exception:
                    num_nodes = 0
            if num_nodes == 0:
                continue
            raise ValueError(
                "SchNet3D requires 3D coordinates `pos`; graph %d is missing them. "
                "Clear cached datasets or rebuild with --add-3d."
                % idx,
            )
        break


def _iter_trainable_params(model) -> List[nn.Parameter]:
    params = getattr(model, "parameters", None)
    if callable(params):
        try:
            return list(params())
        except Exception:
            return []

    for attr in ("encoder", "module", "backbone"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        sub_params = getattr(sub, "parameters", None)
        if callable(sub_params):
            try:
                return list(sub_params())
            except Exception:
                return []
    return []


def _configure_encoder_trainability(
    encoder: nn.Module,
    *,
    freeze_encoder: bool,
    unfreeze_top_layers: int,
    unfreeze_mode: str = "none",
) -> List[nn.Parameter]:
    """Apply fine-tuning freeze/unfreeze policy and return trainable params."""

    if not isinstance(encoder, nn.Module):
        return []

    params_fn = getattr(encoder, "parameters", None)
    if not callable(params_fn):
        return []

    try:
        all_params = list(params_fn())
    except Exception:
        return []

    mode = str(unfreeze_mode or "none").lower()
    if mode == "full":
        freeze_encoder = False
        unfreeze_top_layers = 0
    elif mode == "partial" and freeze_encoder:
        modules = list(encoder.children()) or [encoder]
        if unfreeze_top_layers <= 0:
            unfreeze_top_layers = max(1, min(len(modules), 1))
    if not freeze_encoder and unfreeze_top_layers <= 0:
        for param in all_params:
            param.requires_grad = True
        return all_params

    for param in all_params:
        param.requires_grad = False

    if not freeze_encoder and unfreeze_top_layers > 0:
        for param in all_params:
            param.requires_grad = True
        return all_params

    if unfreeze_top_layers < 0:
        for param in all_params:
            param.requires_grad = True
        return all_params

    if unfreeze_top_layers == 0:
        return []

    modules = list(encoder.children())
    if not modules:
        modules = [encoder]
    selected = modules[-unfreeze_top_layers:]
    seen: set[int] = set()
    trainable: List[nn.Parameter] = []
    for module in selected:
        params_fn = getattr(module, "parameters", None)
        if not callable(params_fn):
            continue
        try:
            module_params = list(params_fn())
        except Exception:
            continue
        for param in module_params:
            param.requires_grad = True
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                trainable.append(param)
    return trainable


def _cmd_finetune_single(args: argparse.Namespace) -> Dict[str, Any]:
    """Fine‑tune a single labelled task and return stage diagnostics."""
    logger.info("Starting finetune with args: %s", args)

    from utils.checkpoint import compute_state_dict_hash, load_checkpoint, save_checkpoint

    try:
        from ..utils.checkpoint import (
            load_state_dict_forgiving as _load_state_dict_forgiving,  # type: ignore[import-not-found]
        )
        from ..utils.checkpoint import (
            safe_load_checkpoint as _safe_load_checkpoint,  # type: ignore[import-not-found]
        )
    except ImportError:
        # Fallback: absolute imports when run from repo root with PYTHONPATH set
        from utils.checkpoint import (
            load_state_dict_forgiving as _load_state_dict_forgiving,  # type: ignore[import-not-found]
        )
        from utils.checkpoint import (
            safe_load_checkpoint as _safe_load_checkpoint,  # type: ignore[import-not-found]
        )

    # Directories / resume
    args.ckpt_dir = getattr(args, "ckpt_dir", "ckpts/finetune")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    resume_state = {}

    if getattr(args, "resume_ckpt", None):
        # wb may not exist yet; use logger or postpone this log until after wb is created
        logger.info("[finetune] resuming from %s", args.resume_ckpt)
        resume_state = load_checkpoint(args.resume_ckpt)

    if (
        load_directory_dataset is None
        or build_encoder is None
        or train_linear_head is None
    ):
        logger.error("Fine-tuning modules are unavailable.")
        sys.exit(3)

    requires_3d = _maybe_enable_add_3d(args)

    max_finetune_batches = int(getattr(args, "max_finetune_batches", 0) or 0)
    setattr(args, "max_finetune_batches", max_finetune_batches)
    pretrain_cap = int(getattr(args, "max_pretrain_batches", 0) or 0)
    if pretrain_cap > 0 and max_finetune_batches == 0:
        logger.warning(
            "Ignoring --max-pretrain-batches=%d during fine-tuning; use --max-finetune-batches to cap downstream epochs.",
            pretrain_cap,
        )
    if max_finetune_batches > 0:
        logger.info(
            "Fine-tune batches per epoch capped at %d via --max-finetune-batches.",
            max_finetune_batches,
        )

    # Determine seeds: CLI overrides config
    seeds: List[int]
    if args.seeds is not None and len(args.seeds) > 0:
        seeds = args.seeds
    else:
        seeds = CONFIG.get("finetune", {}).get("seeds", [0])  # type: ignore[assignment]

    try:
        dataset_path = _resolve_labeled_dataset_source(args)
    except FileNotFoundError:
        logger.exception("Unable to locate labelled dataset")
        sys.exit(1)

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "labeled_dir": args.labeled_dir,
            "labeled_csv": getattr(args, "labeled_csv", None),
            "gnn_type": args.gnn_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": getattr(args, "dropout", None),
            "task_type": args.task_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "encoder_lr": getattr(args, "encoder_lr", None),
            "head_lr": getattr(args, "head_lr", None),
            "ema_decay": args.ema_decay,
            "seeds": seeds,
            "add_3d": bool(getattr(args, "add_3d", False)),
            "freeze_encoder": bool(getattr(args, "freeze_encoder", True)),
            "unfreeze_top_layers": int(getattr(args, "unfreeze_top_layers", 0) or 0),
            "max_finetune_batches": max_finetune_batches,
            "num_workers": getattr(args, "num_workers", None),
            "pin_memory": getattr(args, "pin_memory", None),
            "persistent_workers": getattr(args, "persistent_workers", None),
            "prefetch_factor": getattr(args, "prefetch_factor", None),
            "bf16": bool(getattr(args, "bf16", False)),
            "use_scaffold": bool(getattr(args, "use_scaffold", False)),
            "dataset_override_reason": os.getenv("FINETUNE_DATASET_OVERRIDE_REASON"),
        },
    )
    log_effective_gnn(args, logger, wb)

    # Load labelled dataset
    label_drop_stats: Dict[str, int] = {}
    try:
        labeled = load_directory_dataset(
            dataset_path,
            label_col=args.label_col,
            add_3d=args.add_3d,
            num_workers=getattr(args, "num_workers", -1),
            cache_dir=getattr(args, "cache_dir", None),
        )  # type: ignore[arg-type]

        if requires_3d:
            _ensure_dataset_has_pos(labeled)

        labeled, label_drop_stats = _sanitize_dataset_labels(labeled)
        if label_drop_stats.get("dropped_total", 0) > 0:
            logger.info(
                "[finetune] filtered %d/%d labelled rows (negative or non-finite labels)",
                label_drop_stats.get("dropped_total", 0),
                label_drop_stats.get("size_before", len(labeled)),
            )
            if wb is not None:
                try:
                    wb.log(
                        {
                            "dataset/labels_dropped": float(label_drop_stats["dropped_total"]),
                            "dataset/labels_dropped_negative": float(
                                label_drop_stats.get("dropped_negative", 0)
                            ),
                            "dataset/labels_dropped_nonfinite": float(
                                label_drop_stats.get("dropped_nonfinite", 0)
                            ),
                        }
                    )
                except Exception:
                    pass

        # Sample a subset of labeled graphs if requested.  Use getattr to
        # handle cases where sample_labeled isn’t provided.
        sample_lb = getattr(args, "sample_labeled", 0)
        if (
            sample_lb
            and hasattr(labeled, "__len__")
            and len(labeled) > sample_lb
            and hasattr(labeled, "random_subset")
        ):
            labeled = labeled.random_subset(sample_lb, seed=42)

        wb.log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset")
        wb.log({"phase": "data_load", "status": "error"})
        sys.exit(1)

    if len(labeled) <= 0:
        logger.error("No labelled rows remain after filtering; aborting fine-tune")
        wb.log({"phase": "data_load", "status": "empty_dataset"})
        sys.exit(1)

    dataset_size = len(labeled)
    dataset_override_reason = (os.getenv("FINETUNE_DATASET_OVERRIDE_REASON", "") or "").strip()
    if dataset_override_reason:
        logger.info("[finetune] dataset override reason=%s", dataset_override_reason)
    logger.info(
        "[finetune] dataset path=%s label_col=%s task=%s samples=%d",
        dataset_path,
        getattr(args, "label_col", "<unset>"),
        getattr(args, "task_type", "<unset>"),
        dataset_size,
    )

    if wb is not None:
        try:
            wb.config.update({"dataset_size": dataset_size}, allow_val_change=True)
        except Exception:
            try:
                wb.config.update({"dataset_size": dataset_size})
            except Exception:
                pass

    initial_batch_size = int(getattr(args, "batch_size", 0) or 0)
    effective_batch_size = initial_batch_size
    target_batches_env = os.getenv("FINETUNE_MIN_TRAIN_BATCHES", "").strip()
    min_batches_target = 8
    if target_batches_env:
        try:
            parsed = int(float(target_batches_env))
        except Exception:
            parsed = 0
        if parsed > 0:
            min_batches_target = max(4, parsed)
    batch_size_autoscaled = False
    batch_size_autoscale_notes: List[Dict[str, int]] = []

    scaffold_requested = bool(getattr(args, "_use_scaffold_provided", False))
    use_scaffold_flag = bool(getattr(args, "use_scaffold", False))
    labeled_lower = str(getattr(args, "labeled_dir", "") or "").lower()
    label_col_name = str(getattr(args, "label_col", "") or "")
    detected_tox21 = "tox21" in labeled_lower or label_col_name in _TOX21_TASKS
    task_type_norm = str(getattr(args, "task_type", "") or "").strip().lower()
    auto_scaffold = False
    if (
        not use_scaffold_flag
        and not scaffold_requested
        and detected_tox21
        and task_type_norm == "classification"
    ):
        use_scaffold_flag = True
        auto_scaffold = True
        setattr(args, "use_scaffold", True)
        logger.info("[finetune] enabling scaffold split for Tox21 fine-tune dataset")

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = (
        None
        if labeled.graphs[0].edge_attr is None
        else labeled.graphs[0].edge_attr.shape[1]
    )
    device = resolve_device(args.device)

    autoscale_env = str(os.getenv("BATCH_AUTOSCALE", "1")).strip().lower()
    enable_batch_autoscale = autoscale_env not in {"0", "false", "no", "off"}

    def _prepare_split_for_seed(seed_value: int) -> Dict[str, List[int]]:
        if dataset_size <= 0:
            return {}

        total = dataset_size
        indices = list(range(total))

        smiles_attr = getattr(labeled, "smiles", None)
        if use_scaffold_flag and smiles_attr is not None:
            try:
                train_idx_np, val_idx_np, test_idx_np = scaffold_split_indices(
                    list(smiles_attr), seed=seed_value
                )
                return {
                    "train_indices": train_idx_np.astype(int).tolist(),
                    "val_indices": val_idx_np.astype(int).tolist(),
                    "test_indices": test_idx_np.astype(int).tolist(),
                }
            except Exception:
                logger.debug(
                    "Scaffold split unavailable for seed %d; falling back to stratified/random split.",
                    seed_value,
                    exc_info=True,
                )

        labels_attr = getattr(labeled, "labels", None)
        if task_type_norm == "classification" and labels_attr is not None:
            labels_arr = np.asarray(labels_attr)
            if labels_arr.ndim > 1:
                if labels_arr.shape[1] == 1:
                    labels_arr = labels_arr[:, 0]
                else:
                    labels_arr = None
            if labels_arr is not None and labels_arr.ndim == 1 and labels_arr.size == total:
                state = random.getstate()
                try:
                    random.seed(seed_value)
                    train_idx, val_idx, test_idx = stratified_split(
                        indices, labels_arr, train_frac=0.8, val_frac=0.1
                    )
                finally:
                    random.setstate(state)
                return {
                    "train_indices": list(train_idx),
                    "val_indices": list(val_idx),
                    "test_indices": list(test_idx),
                }

        rng = np.random.RandomState(seed_value)
        shuffled = indices[:]
        rng.shuffle(shuffled)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        return {
            "train_indices": shuffled[:train_end],
            "val_indices": shuffled[train_end:val_end],
            "test_indices": shuffled[val_end:],
        }

    encoder_cfg_template = {
        "gnn_type": args.gnn_type,
        "hidden_dim": int(args.hidden_dim) if args.hidden_dim is not None else None,
        "num_layers": int(args.num_layers) if args.num_layers is not None else None,
        "add_3d": bool(getattr(args, "add_3d", False)),
        "edge_dim": int(edge_dim) if edge_dim is not None else None,
        "input_dim": int(input_dim),
    }

    def _resolved_encoder_cfg(module: nn.Module) -> Dict[str, Any]:
        cfg = dict(encoder_cfg_template)
        for key in ("hidden_dim", "num_layers"):
            attr = getattr(module, key, None)
            if attr is not None:
                try:
                    cfg[key] = int(attr)
                except Exception:
                    cfg[key] = attr
        gnn_attr = getattr(module, "gnn_type", None)
        if gnn_attr is not None:
            cfg["gnn_type"] = gnn_attr
        return cfg

    # Aggregate metrics across seeds
    metrics_runs: List[Dict[str, float]] = []
    seed_best_paths: Dict[int, str] = {}
    seed_train_steps: Dict[int, float] = {}
    seed_best_metric: Dict[int, float] = {}
    seed_best_step: Dict[int, float] = {}
    seed_best_mode: Dict[int, str] = {}
    seed_baseline_hash: Dict[int, str] = {}

    encoder_unfreeze_mode: Optional[str] = None
    encoder_was_trainable = False
    cumulative_encoder_batches = 0.0

    # --- choose metric & direction (maximize cls metrics, minimize losses/errors) ---
    metric_choice = getattr(args, "metric", None)
    if not metric_choice:
        metric_choice = "val_auc" if args.task_type == "classification" else "val_loss"
        setattr(args, "metric", metric_choice)
        logger.debug("Defaulting fine-tune metric to %s for task=%s", metric_choice, args.task_type)

    metric_name = str(metric_choice or "").lower()
    maximize_metrics = {
        "acc",
        "accuracy",
        "auc",
        "auroc",
        "roc_auc",
        "val_auc",
        "f1",
        "f1_macro",
        "f1_micro",
        "r2",
        "val_r2",
    }
    preferred_mode = "max" if metric_name in maximize_metrics else "min"

    def _lookup_metric(m: dict, name: str):
        """Return float metric value; try common aliases if the exact key is missing."""
        v = m.get(name)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
        aliases = {
            "val_rmse": ["rmse_mean", "rmse"],
            "val_mae": ["mae_mean", "mae"],
            "val_auc": ["auc", "auroc", "roc_auc"],
            "acc": ["accuracy", "val_acc"],
            "accuracy": ["acc", "val_acc"],
            "r2": ["val_r2"],
        }
        for k in aliases.get(name, []):
            v = m.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    def _is_better(curr: float, best: Optional[float], mode: str) -> bool:
        if best is None:
            return True
        return curr > best if mode == "max" else curr < best

    save_every = max(1, int(getattr(args, "save_every", 1)))

    budget_abort = False
    min_headroom = None

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)  # okay if no CUDA; harmless on CPU
        except Exception:
            pass

        split_indices = _prepare_split_for_seed(seed)
        train_split = tuple(split_indices.get("train_indices", []))
        val_split = tuple(split_indices.get("val_indices", []))
        test_split = tuple(split_indices.get("test_indices", []))
        if split_indices:
            logger.info(
                "[finetune] seed=%d using fixed dataset split (train=%d val=%d test=%d)",
                seed,
                len(train_split),
                len(val_split),
                len(test_split),
            )

        approx_train_size = len(train_split)
        if approx_train_size <= 0:
            approx_train_size = max(1, int(round(dataset_size * 0.8)))
        if effective_batch_size > 0 and approx_train_size > 0:
            threshold = effective_batch_size * min_batches_target
            if approx_train_size < threshold:
                candidate_batch_size = int(
                    math.ceil(approx_train_size / float(min_batches_target))
                )
                candidate_batch_size = max(1, candidate_batch_size)
                candidate_batch_size = min(candidate_batch_size, effective_batch_size)
                candidate_batch_size = min(candidate_batch_size, approx_train_size)
                if candidate_batch_size < effective_batch_size:
                    logger.info(
                        "[finetune] reducing batch_size from %d to %d to target ≥%d batches (seed=%d train_split=%d)",
                        effective_batch_size,
                        candidate_batch_size,
                        min_batches_target,
                        seed,
                        approx_train_size,
                    )
                    if wb is not None:
                        try:
                            wb.log(
                                {
                                    "finetune/batch_size_initial": float(
                                        effective_batch_size
                                    ),
                                    "finetune/batch_size_effective": float(
                                        candidate_batch_size
                                    ),
                                }
                            )
                        except Exception:
                            pass
                    effective_batch_size = candidate_batch_size
                    setattr(args, "batch_size", effective_batch_size)
                    batch_size_autoscaled = True
                    batch_size_autoscale_notes.append(
                        {
                            "seed": int(seed),
                            "train_samples": int(approx_train_size),
                            "batch_size": int(effective_batch_size),
                        }
                    )

        encoder = build_encoder(
            gnn_type=args.gnn_type,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            edge_dim=edge_dim,
            dropout=getattr(args, "dropout", None),
        )
        # ensure modules on device
        _maybe_to(encoder, device)

        # Unit tests frequently substitute light-weight encoder stubs without the
        # ``parameters`` iterator that ``nn.Module`` provides.  Guard against that
        # so downstream code can continue treating the encoder as frozen.
        params_attr = getattr(encoder, "parameters", None)
        if params_attr is None or not callable(params_attr):
            setattr(encoder, "parameters", lambda: iter(()))  # type: ignore[attr-defined]

        # Load pretrained encoder weights (from pretrain output)
        if getattr(args, "encoder", None):
            state, _ = _safe_load_checkpoint(
                primary=args.encoder,
                ckpt_dir=None,
                default_name="encoder.pt",
                map_location=device,
                allow_missing=False,
            )
            # state may be {"encoder": ...} or a raw state_dict
            enc_weights = state.get("encoder", state) if isinstance(state, dict) else {}
            if enc_weights:
                logger.info("[finetune] loaded encoder from %s", args.encoder)
                _load_state_dict_forgiving(encoder, enc_weights)
            else:
                logger.warning(
                    "[finetune] no encoder weights found in %s; using random init",
                    args.encoder,
                )

        extra_encoder_ckpt = getattr(args, "load_encoder_checkpoint", None)
        if extra_encoder_ckpt:
            state, _ = _safe_load_checkpoint(
                primary=extra_encoder_ckpt,
                ckpt_dir=None,
                default_name="encoder.pt",
                map_location=device,
                allow_missing=True,
            )
            enc_weights = state.get("encoder", state) if isinstance(state, dict) else {}
            if enc_weights:
                logger.info(
                    "[finetune] loaded additional encoder weights from %s",
                    extra_encoder_ckpt,
                )
                _load_state_dict_forgiving(encoder, enc_weights)
            else:
                logger.warning(
                    "[finetune] encoder checkpoint %s lacked weights; ignoring",
                    extra_encoder_ckpt,
                )

        # If resuming a fine-tune checkpoint, it may contain a fresher encoder
        if "encoder" in resume_state:
            logger.info("Overriding encoder from resume checkpoint")
            _load_state_dict_forgiving(encoder, resume_state["encoder"])

        # Build linear head for fine-tuning
        raw_mode = str(getattr(args, "unfreeze_mode", "none") or "none").lower()
        if raw_mode not in {"none", "partial", "full"}:
            raw_mode = "none"
        freeze_override = getattr(args, "freeze_encoder", None)
        if freeze_override is True:
            freeze_flag = True
            effective_mode = "none"
        elif freeze_override is False:
            freeze_flag = False
            effective_mode = "full"
        else:
            effective_mode = raw_mode
            freeze_flag = effective_mode != "full"

        unfreeze_top = int(getattr(args, "unfreeze_top_layers", 0) or 0)
        if effective_mode == "partial" and unfreeze_top <= 0:
            modules = list(encoder.children()) or [encoder]
            unfreeze_top = max(1, min(len(modules), 1))
            logger.info(
                "Partial unfreeze selected without explicit layer count; defaulting to top %d module(s).",
                unfreeze_top,
            )

        trainable_encoder_params = _configure_encoder_trainability(
            encoder,
            freeze_encoder=freeze_flag,
            unfreeze_top_layers=unfreeze_top,
            unfreeze_mode=effective_mode,
        )
        trainable_param_count = sum(int(p.numel()) for p in trainable_encoder_params)
        encoder_was_trainable = encoder_was_trainable or (trainable_param_count > 0)
        if trainable_param_count > 0:
            logger.info(
                "Encoder fine-tuning enabled (%d trainable parameters, mode=%s, freeze=%s, unfreeze_top_layers=%d)",
                trainable_param_count,
                effective_mode,
                freeze_flag,
                unfreeze_top,
            )
        else:
            logger.info(
                "Encoder frozen during fine-tuning (mode=%s, freeze=%s, unfreeze_top_layers=%d)",
                effective_mode,
                freeze_flag,
                unfreeze_top,
            )
        if effective_mode in {"partial", "full"} and trainable_param_count == 0:
            raise RuntimeError(
                "Encoder unfreeze mode '%s' requested but no trainable parameters were found. Check encoder construction."
                % effective_mode
            )
        if wb is not None:
            try:
                wb.log(
                    {
                        "encoder/trainable_params": float(trainable_param_count),
                        "encoder/freeze_flag": bool(freeze_flag),
                        "encoder/unfreeze_mode": effective_mode,
                    }
                )
            except Exception:
                pass
        encoder_unfreeze_mode = effective_mode

        baseline_hash = None
        try:
            enc_state_snapshot = encoder.state_dict()
        except Exception:
            enc_state_snapshot = None
        if enc_state_snapshot:
            try:
                baseline_hash = compute_state_dict_hash(enc_state_snapshot)
            except Exception:
                logger.exception("Failed to compute baseline encoder hash")
        if baseline_hash:
            logger.info("[baseline_encoder_hash]=%s", baseline_hash)
            seed_baseline_hash[seed] = baseline_hash

        # compute num_classes robustly for classification; for regression we won’t use it
        _in_dim = getattr(encoder, "hidden_dim", getattr(args, "hidden_dim", None))
        assert (
            _in_dim is not None
        ), "hidden dim unknown (encoder.hidden_dim or args.hidden_dim required)"
        if args.task_type == "classification":
            # robust class count
            num_classes = _infer_num_classes(labeled)

            # optional label stats (never break if missing)
            y_arr = _maybe_labels(labeled)
            if y_arr is not None:
                try:
                    # simple example: fraction of positives if binary labels
                    import numpy as _np

                    arr = _np.asarray(y_arr)
                    if arr.ndim > 1:
                        arr = arr[:, 0]
                    pos_frac = float((arr > 0).mean())
                    if "wb" in locals() and wb:
                        wb.log({"dataset/pos_frac": pos_frac})
                except Exception:
                    # Never let metrics logging break training
                    pass

            head = build_linear_head(
                in_dim=_in_dim, num_classes=num_classes, task_type="classification"
            )
        else:
            # regression
            head = build_linear_head(
                in_dim=_in_dim, num_classes=1, task_type="regression"
            )

        if "head" in resume_state and hasattr(head, "load_state_dict"):
            _load_state_dict_forgiving(head, resume_state["head"])

        _maybe_to(head, device)

        loss_name = "BCEWithLogitsLoss" if args.task_type == "classification" else "MSELoss"
        logger.info(
            "[finetune] task_type=%s head=%s loss=%s",
            args.task_type,
            type(head).__name__,
            loss_name,
        )

        # Optimizer & scheduler
        encoder_params = [
            p for p in _iter_trainable_params(encoder) if getattr(p, "requires_grad", False)
        ]
        head_params = [p for p in head.parameters() if p.requires_grad]
        optimizer_groups = []
        raw_head_lr = getattr(args, "head_lr", None)
        head_lr = raw_head_lr if raw_head_lr is not None else args.lr
        try:
            head_lr = float(head_lr)
        except Exception:
            logger.warning("Failed to parse head_lr=%s; falling back to lr=%s", raw_head_lr, args.lr)
            head_lr = float(args.lr)

        raw_encoder_lr = getattr(args, "encoder_lr", None)
        encoder_lr = None
        if raw_encoder_lr is not None:
            try:
                encoder_lr = float(raw_encoder_lr)
            except Exception:
                logger.warning(
                    "Failed to parse encoder_lr=%s; treating as unset", raw_encoder_lr, exc_info=True
                )
                encoder_lr = None
        if encoder_params:
            if encoder_lr is None:
                encoder_lr = 3e-4
                logger.info("Encoder LR defaulting to 3.00e-04")
            optimizer_groups.append({"params": encoder_params, "lr": encoder_lr})
        if head_params:
            optimizer_groups.append({"params": head_params, "lr": head_lr})

        optimizer = (
            torch.optim.AdamW(
                optimizer_groups,
                lr=head_lr,
                weight_decay=1e-4,
            )
            if optimizer_groups
            else None
        )
        if encoder_params:
            logger.info(
                "Optimizer encoder group: %d tensors lr=%.2e",
                len(encoder_params),
                float(encoder_lr or 0.0),
            )
        if head_params:
            logger.info(
                "Optimizer head group: %d tensors lr=%.2e",
                len(head_params),
                float(head_lr),
            )
        if wb is not None:
            try:
                lr_payload = {"optimizer/lr_head": float(head_lr)}
                if encoder_params:
                    lr_payload["optimizer/lr_encoder"] = float(encoder_lr or 0.0)
                wb.log(lr_payload)
            except Exception:
                pass

        encoder_lr_display: Optional[float] = None
        if encoder_params:
            try:
                encoder_lr_display = float(encoder_lr or 0.0)
            except Exception:
                encoder_lr_display = None
        logger.info(
            "[finetune] lr_head=%.2e lr_encoder=%s",
            float(head_lr),
            f"{encoder_lr_display:.2e}" if encoder_lr_display is not None else "<frozen>",
        )
        logger.info("[finetune] unfreeze_mode=%s", effective_mode)

        cache_embeddings = not bool(encoder_params)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        logger.info(
            "CosineAnnealingLR configured with T_max=%d epochs for fine-tuning.",
            args.epochs,
        )

        # If resuming, load head/optim/scheduler from checkpoint
        if "head" in resume_state:
            head.load_state_dict(resume_state["head"], strict=False)
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"], strict=False)
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"], strict=False)

        # epoch to start from (resume file stores last finished epoch)
        start_epoch = int(resume_state.get("epoch", -1)) + 1
        if start_epoch < 0:
            start_epoch = 0

        try:
            wb.log({"phase": f"finetune_{seed}", "status": "start"})

            # per-seed checkpoint dir (avoid overwriting across seeds)
            seed_dir = os.path.join(args.ckpt_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            # initialize best depending on direction
            best_metric: Optional[float] = None
            best_metric_mode = preferred_mode

            wrote_best = False
            last_epoch = start_epoch - 1
            warned_small_loader = False
            for epoch in range(start_epoch, args.epochs):
                metrics = train_linear_head(
                    dataset=labeled,
                    encoder=encoder,
                    # head_type=getattr(args, "head", "linear"),  # <- change innvalid param for supervised?
                    task_type=args.task_type,
                    epochs=1,
                    max_batches=max_finetune_batches,
                    time_budget_mins=getattr(
                        args, "time_budget_mins", 0
                    ),  # ensure it does not crash for unit tests
                    lr=args.lr,
                    batch_size=args.batch_size,
                    device=device,
                    patience=args.patience,
                    devices=args.devices,
                    use_scaffold=use_scaffold_flag,
                    head=head,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    # dataloader & AMP knobs
                    num_workers=getattr(args, "num_workers", -1),
                    pin_memory=getattr(args, "pin_memory", True),
                    persistent_workers=getattr(args, "persistent_workers", True),
                    prefetch_factor=getattr(args, "prefetch_factor", 4),
                    bf16=getattr(args, "bf16", False),
                    encoder_lr=encoder_lr,
                    head_lr=head_lr,
                    freeze_encoder=False,
                    early_stop_metric=getattr(args, "metric", "val_loss"),
                    cache_graph_embeddings=cache_embeddings,
                    enable_batch_autoscale=enable_batch_autoscale,
                    stage_config=stage_config,
                    train_indices=train_split,
                    val_indices=val_split,
                    test_indices=test_split,
                )

                train_batches = float(metrics.get("train/batches", 0.0) or 0.0)
                epoch_batches = float(metrics.get("train/epoch_batches", train_batches) or train_batches)
                loader_batches = float(metrics.get("train/loader_batches", 0.0) or 0.0)

                headroom_val = metrics.get("time/headroom_secs")
                headroom_float = None
                if headroom_val is not None:
                    try:
                        headroom_float = float(headroom_val)
                        if min_headroom is None or headroom_float < min_headroom:
                            min_headroom = headroom_float
                    except Exception:
                        headroom_float = None

                exhausted_flag = 0.0
                raw_exhausted = metrics.get("time/budget_exhausted")
                try:
                    exhausted_flag = float(raw_exhausted or 0.0)
                except Exception:
                    exhausted_flag = 0.0

                if exhausted_flag > 0.0:
                    budget_abort = True
                    remaining_display = headroom_float if headroom_float is not None else float("nan")
                    logger.warning(
                        "[finetune] seed=%d epoch=%d stopping early: wall-clock headroom %.1fs below safety margin.",
                        seed,
                        epoch,
                        remaining_display,
                    )
                    if wb is not None:
                        try:
                            wb.log({"phase": f"finetune_{seed}", "status": "budget_abort"})
                        except Exception:
                            pass
                    break

                if epoch == start_epoch and loader_batches > 0:
                    logger.info(
                        "[finetune] seed=%d train loader reports %d batches per epoch (max_finetune_batches=%d)",
                        seed,
                        int(loader_batches),
                        max_finetune_batches,
                    )
                batch_size_hint = int(getattr(args, "batch_size", 0) or 0)
                if (
                    not warned_small_loader
                    and loader_batches > 0
                    and loader_batches < 6
                    and batch_size_hint >= 256
                ):
                    logger.warning(
                        "[finetune] seed=%d loader has %d batches with batch_size=%d; consider using 128 or 64 for ≥6 steps per epoch.",
                        seed,
                        int(loader_batches),
                        batch_size_hint,
                    )
                    warned_small_loader = True
                if train_batches > 0:
                    logger.info(
                        "[finetune] seed=%d epoch=%d encoder batches=%d",
                        seed,
                        epoch,
                        int(train_batches),
                    )
                if max_finetune_batches == 0 and epoch_batches < 10:
                    logger.warning(
                        "Fine-tune epoch produced only %d batches (seed=%d epoch=%d); representation updates may be limited.",
                        int(epoch_batches),
                        seed,
                        epoch,
                    )
                seed_train_steps[seed] = seed_train_steps.get(seed, 0.0) + epoch_batches
                cumulative_encoder_batches += epoch_batches
                if wb is not None:
                    try:
                        wb_payload = {
                            f"finetune_{seed}/train_batches": train_batches,
                            "encoder/train_batches": train_batches,
                            "encoder/train_batches_epoch": epoch_batches,
                            "encoder/train_batches_total": cumulative_encoder_batches,
                        }
                        if loader_batches > 0:
                            wb_payload[f"finetune_{seed}/loader_batches"] = loader_batches
                        wb.log(wb_payload)
                    except Exception:
                        pass

                trained_head = metrics.pop("head", None)
                if trained_head is not None:
                    head = trained_head

                current = _lookup_metric(metrics, metric_name)
                current_mode = preferred_mode

                if current is None and metric_name != "val_loss":
                    fallback_val = _lookup_metric(metrics, "val_loss")
                    if fallback_val is not None:
                        logger.debug(
                            "Metric '%s' missing in results; falling back to val_loss for checkpointing.",
                            metric_name,
                        )
                        current = fallback_val
                        current_mode = "min"

                if current is not None:
                    if best_metric_mode != current_mode:
                        best_metric = None
                        best_metric_mode = current_mode

                    if _is_better(current, best_metric, current_mode):
                        best_metric = current
                        best_path = os.path.join(seed_dir, "ft_best.pt")
                        best_payload = {"epoch": epoch, "best_metric": best_metric}
                        enc_state = _maybe_state_dict(encoder)
                        head_state = _maybe_state_dict(head)
                        if enc_state is not None:
                            best_payload["encoder"] = enc_state
                            best_payload["encoder_cfg"] = _resolved_encoder_cfg(encoder)
                        if head_state is not None:
                            best_payload["head"] = head_state
                        if optimizer is not None and hasattr(optimizer, "state_dict"):
                            best_payload["optimizer"] = optimizer.state_dict()
                        if scheduler is not None and hasattr(scheduler, "state_dict"):
                            best_payload["scheduler"] = scheduler.state_dict()
                        save_checkpoint(best_path, **best_payload)
                        wrote_best = True
                        seed_best_paths[seed] = best_path
                        seed_best_metric[seed] = float(current)
                        seed_best_step[seed] = float(
                            seed_train_steps.get(seed, 0.0)
                        )
                        seed_best_mode[seed] = str(current_mode)
                        # optional: stable link at the finetune root

                        try:
                            from utils.checkpoint import safe_link_or_copy

                            link = os.path.join(args.ckpt_dir, "head.pt")
                            mode = safe_link_or_copy(best_path, link)
                            logger.info("Updated head.pt (%s) -> %s", mode, best_path)
                        except Exception:
                            logger.warning(
                                "Could not create head.pt symlink", exc_info=True
                            )

                # periodic (and last-epoch) snapshot
                if ((epoch + 1) % save_every == 0) or ((epoch + 1) == args.epochs):
                    save_payload = {"epoch": epoch}
                    for name, obj in (("encoder", encoder), ("head", head)):
                        sd = _maybe_state_dict(obj)
                        if sd is not None:
                            save_payload[name] = sd
                            if name == "encoder":
                                save_payload["encoder_cfg"] = _resolved_encoder_cfg(encoder)
                    if len(save_payload) > 1:
                        save_checkpoint(
                            os.path.join(seed_dir, f"ft_epoch_{epoch+1}.pt"),
                            **save_payload,
                        )

            if budget_abort:
                break

            # Fallback: if no best was recorded, promote last snapshot to best + head.pt
            if not wrote_best:
                logger.info("Attempting to write best")
                try:
                    # find latest epoch file we just wrote
                    snaps = [
                        p
                        for p in os.listdir(seed_dir)
                        if p.startswith("ft_epoch_") and p.endswith(".pt")
                    ]
                    if snaps:
                        snaps.sort(key=lambda s: int(s.split("_")[-1].split(".")[0]))
                        last = os.path.join(seed_dir, snaps[-1])
                        best_path = os.path.join(seed_dir, "ft_best.pt")

                        shutil.copy2(last, best_path)
                        from utils.checkpoint import safe_link_or_copy

                        mode = safe_link_or_copy(
                            best_path, os.path.join(args.ckpt_dir, "head.pt")
                        )
                        logger.info(
                            "Fallback best: head.pt (%s) -> %s", mode, best_path
                        )

                        seed_best_paths[seed] = best_path

                        logger.warning(
                            "No metric '%s' found; promoted %s to ft_best.pt",
                            metric_name,
                            snaps[-1],
                        )
                except Exception:
                    logger.warning("Failed to create fallback ft_best.pt", exc_info=True)  # type: ignore

            wb.log({"phase": f"finetune_{seed}", "status": "success"})
            metrics_runs.append({k: v for k, v in metrics.items() if k != "head"})
        except Exception:
            logger.exception(f"Fine‑tuning failed on seed {seed}")
            wb.log({"phase": f"finetune_{seed}", "status": "error"})
            sys.exit(3)

    total_train_batches = float(sum(seed_train_steps.values()))
    if encoder_was_trainable and total_train_batches <= 0:
        logger.warning(
            "Encoder marked trainable but no optimisation batches were recorded; check dataset splits."
        )
    elif encoder_was_trainable and total_train_batches < 50:
        logger.warning(
            "Encoder fine-tune ran only %d batches across all seeds; consider increasing dataset size, batch size, or epochs.",
            int(total_train_batches),
        )

    export_path: Optional[str] = None
    export_name: Optional[str] = None
    primary_seed = seeds[0] if seeds else None

    summary_seed: Optional[int] = None
    summary_metric_value: Optional[float] = None
    summary_mode = preferred_mode
    summary_score: Optional[float] = None
    for seed, value in seed_best_metric.items():
        if value is None:
            continue
        mode = seed_best_mode.get(seed, preferred_mode)
        try:
            val_float = float(value)
        except Exception:
            continue
        score = val_float if mode == "max" else -val_float
        if summary_score is None or score > summary_score:
            summary_score = score
            summary_seed = seed
            summary_metric_value = val_float
            summary_mode = mode
    if summary_seed is None:
        summary_seed = primary_seed
        summary_metric_value = (
            seed_best_metric.get(summary_seed)
            if summary_seed is not None
            else None
        )
        summary_mode = seed_best_mode.get(summary_seed, preferred_mode)
    summary_step = (
        seed_best_step.get(summary_seed)
        if summary_seed is not None
        else None
    )
    if summary_step is None and summary_seed is not None:
        summary_step = seed_train_steps.get(summary_seed)

    def _fmt_metric(val: Optional[float]) -> str:
        if val is None:
            return "<nan>"
        try:
            return f"{float(val):.4f}"
        except Exception:
            return "<nan>"

    logger.info(
        "[finetune] summary: total_encoder_steps=%.1f best_seed=%s best_step=%.1f metric=%s mode=%s value=%s",
        total_train_batches,
        summary_seed if summary_seed is not None else "<none>",
        float(summary_step or 0.0),
        metric_name,
        summary_mode,
        _fmt_metric(summary_metric_value),
    )
    export_hash: Optional[str] = None
    if encoder_was_trainable and seed_best_paths:
        best_candidate = seed_best_paths.get(primary_seed) if primary_seed is not None else None
        if not best_candidate:
            best_candidate = next(iter(seed_best_paths.values()), None)
        if best_candidate:
            try:
                best_state = load_checkpoint(best_candidate)
                enc_state = (
                    best_state.get("encoder")
                    if isinstance(best_state, dict)
                    else best_state
                )
                if enc_state:
                    alias_source = (
                        os.getenv("EXP_ID")
                        or os.getenv("RUN_ID")
                        or (f"seed{primary_seed}" if primary_seed is not None else "finetune")
                    )
                    # The exported checkpoint is tagged as ``encoder_ft:<alias>`` so
                    # downstream evaluations can distinguish fine-tuned encoders
                    # from the original pretraining lineage.
                    alias = _sanitize_alias(str(alias_source))
                    export_name = f"encoder_ft:{alias}"
                    export_path = os.path.join(args.ckpt_dir, f"encoder_ft_{alias}.pt")
                    encoder_hash = None
                    try:
                        encoder_hash = compute_state_dict_hash(enc_state)
                    except Exception:
                        logger.exception("Failed to compute hash for fine-tuned encoder export")
                    save_checkpoint(
                        export_path,
                        encoder=enc_state,
                        encoder_cfg=_resolved_encoder_cfg(encoder),
                    )
                    try:
                        from utils.checkpoint import safe_link_or_copy

                        link_target = os.path.join(args.ckpt_dir, "encoder_ft.pt")
                        safe_link_or_copy(export_path, link_target)
                    except Exception:
                        logger.debug("Could not create encoder_ft.pt symlink", exc_info=True)
                    logger.info(
                        "Exported fine-tuned encoder to %s (artifact %s)",
                        export_path,
                        export_name,
                    )
                    if wb is not None:
                        try:
                            wb.log(
                                {
                                    "encoder/export_path": export_path,
                                    "encoder/export_artifact": export_name,
                                }
                            )
                        except Exception:
                            pass
                    if encoder_hash:
                        export_hash = encoder_hash
                        logger.info("[encoder_hash]=%s source=finetune_export path=%s", encoder_hash, export_path)
            except Exception:
                logger.exception(
                    "Failed to export fine-tuned encoder from %s", best_candidate
                )

    agg = aggregate_metrics(metrics_runs)
    for k, v in agg.items():
        wb.log({f"metric/{k}": v})
    wb.finish()

    stage_payload: Dict[str, Any] = {
        "encoder_unfreeze_mode": encoder_unfreeze_mode or "none",
        "encoder_trainable": bool(encoder_was_trainable),
        "encoder_train_steps": total_train_batches,
        "seeds": {
            str(seed): {
                "best_checkpoint": seed_best_paths.get(seed),
                "train_batches": float(seed_train_steps.get(seed, 0.0)),
                "best_metric": float(seed_best_metric.get(seed))
                if seed in seed_best_metric
                else None,
                "best_step": float(seed_best_step.get(seed))
                if seed in seed_best_step
                else None,
                "best_mode": seed_best_mode.get(seed),
                "baseline_encoder_hash": seed_baseline_hash.get(seed),
            }
            for seed in seeds
        },
    }
    if stage_config:
        try:
            stage_payload["stage_config"] = dict(stage_config)
        except Exception:
            stage_payload["stage_config"] = stage_config
    if budget_abort:
        stage_payload["time_budget_exhausted"] = True
        if min_headroom is not None:
            try:
                stage_payload["time_budget_headroom_secs"] = float(min_headroom)
            except Exception:
                pass
        stage_payload["time_budget_exit_code"] = int(soft_timeout_exit_code)
    if seed_baseline_hash:
        stage_payload["baseline_encoder_hashes"] = {
            str(k): v for k, v in seed_baseline_hash.items() if v
        }
        primary_hash_seed = summary_seed if summary_seed is not None else primary_seed
        baseline_candidate = None
        if primary_hash_seed is not None:
            baseline_candidate = seed_baseline_hash.get(primary_hash_seed)
        if baseline_candidate is None and seed_baseline_hash:
            baseline_candidate = next(iter(seed_baseline_hash.values()), None)
        if baseline_candidate:
            stage_payload["baseline_encoder_hash"] = baseline_candidate
    if export_path:
        stage_payload["encoder_finetuned"] = {
            "checkpoint": export_path,
            "artifact": export_name,
            "source_seed": primary_seed,
        }
        if export_hash:
            stage_payload["encoder_finetuned"]["hash"] = export_hash

    try:
        seed_list_summary = [int(s) for s in seeds]
    except Exception:
        seed_list_summary = list(seeds)

    dataset_summary: Dict[str, Any] = {
        "path": dataset_path,
        "directory": os.path.abspath(getattr(args, "labeled_dir", ""))
        if getattr(args, "labeled_dir", None)
        else getattr(args, "labeled_dir", None),
        "csv": getattr(args, "labeled_csv", None),
        "label_col": getattr(args, "label_col", None),
        "task_type": getattr(args, "task_type", None),
        "size": int(dataset_size),
        "use_scaffold": bool(use_scaffold_flag),
        "auto_scaffold": bool(auto_scaffold),
        "detected_tox21": bool(detected_tox21),
        "metric": metric_name,
        "override_reason": dataset_override_reason or None,
        "batch_size_initial": int(initial_batch_size) if initial_batch_size else None,
        "batch_size_effective": int(effective_batch_size) if effective_batch_size else None,
        "batch_size_target_batches": int(min_batches_target),
    }
    if label_drop_stats:
        dataset_summary["labels_dropped"] = int(label_drop_stats.get("dropped_total", 0))
        dataset_summary["labels_dropped_negative"] = int(
            label_drop_stats.get("dropped_negative", 0)
        )
        dataset_summary["labels_dropped_nonfinite"] = int(
            label_drop_stats.get("dropped_nonfinite", 0)
        )
    stage_payload["dataset"] = dataset_summary
    stage_payload["seed_list"] = seed_list_summary
    if getattr(args, "ckpt_dir", None):
        try:
            stage_payload["stage_path"] = os.path.abspath(str(args.ckpt_dir))
        except Exception:
            stage_payload["stage_path"] = str(args.ckpt_dir)
    if export_path:
        stage_payload.setdefault("selected_path", export_path)
    stage_payload.setdefault("task_label", getattr(args, "label_col", None))
    diagnostics = stage_payload.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics.setdefault("batch_size_autoscaled", bool(batch_size_autoscaled))
        diagnostics.setdefault(
            "batch_size_autoscale_notes",
            batch_size_autoscale_notes if batch_size_autoscale_notes else None,
        )
        if getattr(args, "_tox21_epoch_floor_applied", False):
            diagnostics.setdefault("tox21_epoch_floor_applied", True)
            diagnostics.setdefault(
                "tox21_epoch_floor_value",
                int(getattr(args, "_tox21_epoch_floor_value", 0) or 0),
            )
            diagnostics.setdefault(
                "tox21_epoch_requested",
                int(getattr(args, "_tox21_epoch_requested", 0) or 0),
            )
    _record_finetune_stage_outputs(stage_payload)

    if budget_abort:
        remaining_msg = (
            f"{min_headroom:.1f}s" if isinstance(min_headroom, (int, float)) else "unknown"
        )
        logger.error(
            "[finetune] exiting with code %d after wall-clock headroom dropped below safety margin (remaining=%s).",
            soft_timeout_exit_code,
            remaining_msg,
        )
        sys.exit(int(soft_timeout_exit_code))

    return stage_payload


def cmd_finetune(args: argparse.Namespace) -> None:
    _apply_best_config_overrides(args)
    label_columns = _resolve_label_columns(args)
    _maybe_enforce_tox21_epoch_floor(args, label_columns)
    if len(label_columns) <= 1:
        if label_columns:
            setattr(args, "label_col", label_columns[0])
        _cmd_finetune_single(args)
        return

    logger.info(
        "[finetune] multi-assay run over %d tasks: %s",
        len(label_columns),
        ", ".join(label_columns),
    )

    base_ckpt_dir = Path(str(getattr(args, "ckpt_dir", "ckpts/finetune")))
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    stage_env = os.getenv("STAGE_OUTPUTS_DIR")
    base_stage_dir = Path(stage_env) if stage_env else base_ckpt_dir / "stage-outputs"

    aggregated: Dict[str, Dict[str, Any]] = {}
    primary_label = label_columns[0]

    fanout_entries: List[Dict[str, Any]] = []
    for label in label_columns:
        sub_args = copy.deepcopy(args)
        sub_args.label_col = label
        safe_label = _sanitize_alias(label)
        sub_ckpt_dir = base_ckpt_dir / safe_label
        sub_args.ckpt_dir = str(sub_ckpt_dir)
        sub_stage_dir = base_stage_dir / safe_label
        os.environ["FINETUNE_STAGE_SUFFIX"] = safe_label
        os.environ["STAGE_OUTPUTS_DIR"] = str(sub_stage_dir)
        sub_stage_dir.mkdir(parents=True, exist_ok=True)
        snapshot = _namespace_snapshot(sub_args)
        snapshot["label_col"] = label
        snapshot["ckpt_dir"] = str(sub_ckpt_dir)
        fanout_entries.append(
            {
                "label": label,
                "safe_label": safe_label,
                "ckpt_dir": str(sub_ckpt_dir),
                "stage_outputs_dir": str(sub_stage_dir),
                "devices": getattr(sub_args, "devices", None),
                "seeds": list(getattr(sub_args, "seeds", []) or []),
                "env": {
                    "FINETUNE_LABEL_COL": label,
                    "FINETUNE_LABEL_COLS": label,
                    "FINETUNE_STAGE_SUFFIX": safe_label,
                    "STAGE_OUTPUTS_DIR": str(sub_stage_dir),
                },
                "args": snapshot,
            }
        )
        logger.info(
            "[finetune] training assay %s (ckpt_dir=%s)",
            label,
            sub_args.ckpt_dir,
        )
        payload = _cmd_finetune_single(sub_args)
        aggregated[label] = payload

    if stage_env is not None:
        os.environ["STAGE_OUTPUTS_DIR"] = stage_env
    else:
        os.environ.pop("STAGE_OUTPUTS_DIR", None)
    os.environ.pop("FINETUNE_STAGE_SUFFIX", None)

    base_stage_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_fanout_manifest(base_stage_dir, fanout_entries)

    primary_payload = aggregated.get(primary_label)
    summary_payload: Dict[str, Any] = {}
    if isinstance(primary_payload, dict):
        summary_payload.update(copy.deepcopy(primary_payload))
    summary_payload["tasks"] = aggregated
    summary_payload["primary_task"] = primary_label
    summary_payload["task_order"] = label_columns
    if manifest_path is not None:
        summary_payload.setdefault("diagnostics", {})
        diagnostics = summary_payload["diagnostics"]
        if isinstance(diagnostics, dict):
            diagnostics.setdefault("fanout_manifest", str(manifest_path))

    diagnostics = summary_payload.setdefault("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        summary_payload["diagnostics"] = diagnostics
    diagnostics.setdefault("task_count", len(label_columns))
    diagnostics.setdefault("task_labels", label_columns)

    if isinstance(primary_payload, dict):
        canonical_entry = primary_payload.get("encoder_finetuned")
        if isinstance(canonical_entry, dict):
            checkpoint = canonical_entry.get("checkpoint")
            if checkpoint and "encoder_checkpoint" not in diagnostics:
                diagnostics["encoder_checkpoint"] = checkpoint

    _record_finetune_stage_outputs(summary_payload)

    canonical_entry = None
    if isinstance(primary_payload, dict):
        canonical_entry = primary_payload.get("encoder_finetuned")
    if isinstance(canonical_entry, dict):
        checkpoint = canonical_entry.get("checkpoint")
        if checkpoint and os.path.isfile(checkpoint):
            try:
                from utils.checkpoint import safe_link_or_copy
            except Exception:
                safe_link_or_copy = None  # type: ignore[assignment]
            target = base_ckpt_dir / "encoder_ft.pt"
            try:
                if safe_link_or_copy is not None:
                    safe_link_or_copy(checkpoint, str(target))
                else:
                    shutil.copy2(checkpoint, target)
            except Exception:
                logger.debug(
                    "Failed to mirror encoder_ft.pt from %s", checkpoint, exc_info=True
                )

    logger.info("[finetune] completed %d Tox21 assays", len(label_columns))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a pretrained encoder by training a linear probe across seeds."""
    logger.info("Starting evaluate with args: %s", args)
    # Reuse finetune implementation with a different default config section
    cmd_finetune(args)


def evaluate_finetuned_head(
    ckpt_path: str, dataset, args: argparse.Namespace, device
) -> Dict[str, float]:
    """Evaluate a fine‑tuned encoder+head on a labelled dataset.

    This is used for the benchmark step when we want to avoid training a new
    head on the test split. The checkpoint is expected to contain both an
    ``encoder`` and ``head`` state dict. Metrics are computed on the entire
    dataset.
    """

    from utils.checkpoint import load_checkpoint
    from utils.metrics import compute_classification_metrics, compute_regression_metrics

    try:
        from ..models.factory import build_encoder  # type: ignore[import-not-found]
        from ..utils.checkpoint import (
            load_state_dict_forgiving as _load_state_dict_forgiving,  # type: ignore[import-not-found]
        )
        from ..utils.pooling import global_mean_pool  # type: ignore[import-not-found]
    except ImportError:
        # Fallback: absolute imports when run from repo root with PYTHONPATH set
        from models.factory import build_encoder  # type: ignore[import-not-found]
        from utils.checkpoint import (
            load_state_dict_forgiving as _load_state_dict_forgiving,  # type: ignore[import-not-found]
        )
        from utils.pooling import global_mean_pool  # type: ignore[import-not-found]

    # module logger (safe even when run outside the injector)
    import logging

    logger = logging.getLogger(__name__)

    # local helper so there are zero naming conflicts with injected globals
    def _to_dev(x, dev):
        try:
            return x.to(dev)
        except Exception:
            return x

    state = load_checkpoint(ckpt_path)
    if "encoder" not in state or "head" not in state:
        logger.warning("Checkpoint missing encoder or head: %s", ckpt_path)
        return {}

    # 1) Try to get the exact encoder config from the finetune ckpt
    enc_cfg = {}
    if isinstance(state, dict):
        enc_cfg = {
            k: v for k, v in (state.get("encoder_cfg") or {}).items() if v is not None
        }

    logger.info("Eval encoder cfg: %s", enc_cfg)

    # 2) If missing, look for a sidecar pretrain encoder next to the head
    #    (we often symlink encoder.pt into the finetune dir)
    import collections
    import os

    import torch

    sidecar = os.path.join(os.path.dirname(ckpt_path or ""), "encoder.pt")
    side_state = None
    if not enc_cfg and os.path.isfile(sidecar):
        try:
            side_state = torch.load(sidecar, map_location="cpu")
        except Exception:
            side_state = None

    # 3) If still missing, *attempt* to infer hidden_dim from state shapes (best-effort)
    def _infer_hidden_dim(sd):
        if not isinstance(sd, dict):
            return None
        c = collections.Counter()
        for k, v in sd.items():
            shp = getattr(v, "shape", None)
            if isinstance(shp, tuple) and len(shp) == 2:
                in_f = shp[1]
                if 64 <= in_f <= 2048 and in_f % 32 == 0:
                    c[in_f] += 1
        return c.most_common(1)[0][0] if c else None

    if not enc_cfg:
        hid = _infer_hidden_dim((state or {}).get("encoder", {})) or _infer_hidden_dim(
            side_state or {}
        )
        if hid:
            enc_cfg["hidden_dim"] = hid

    # 4) Fall back to CLI args for anything still missing
    for k in ("gnn_type", "hidden_dim", "num_layers", "add_3d", "dropout"):
        if k not in enc_cfg and hasattr(args, k):
            enc_cfg[k] = getattr(args, k)

    # 4.1) Derive edge_dim from the dataset (needed for edge_mpnn)
    try:
        g0 = dataset.graphs[0]
        _edge_dim = (
            None
            if getattr(g0, "edge_attr", None) is None
            else int(g0.edge_attr.shape[1])
        )
    except Exception:
        _edge_dim = None
    if _edge_dim is None or _edge_dim <= 0:
        _edge_dim = 1
    enc_cfg.setdefault("edge_dim", _edge_dim)

    # 4.5) Ensure required input_dim is present (infer from dataset if needed)
    in_dim = None
    for attr in ("input_dim", "node_feat_dim", "n_node_features"):
        val = getattr(dataset, attr, None)
        if isinstance(val, int) and val > 0:
            in_dim = val
            break
    if in_dim is None:
        try:
            bx, _, _, _ = dataset.get_batch([0])
            in_dim = int(bx.shape[-1])
        except Exception:
            in_dim = None
    if in_dim is not None:
        enc_cfg["input_dim"] = in_dim

    # 5) Finally build the encoder with the best config we have (filter unknown keys)

    import inspect

    enc_cfg = enc_cfg or {}
    try:
        sig_params = set(inspect.signature(build_encoder).parameters.keys())
    except (ValueError, TypeError):
        sig_params = {
            "gnn_type",
            "hidden_dim",
            "num_layers",
            "input_dim",
        }  # conservative fallback
    # normalize & filter
    norm = dict(enc_cfg)
    gt = norm.get("gnn_type")
    if isinstance(gt, str):
        norm["gnn_type"] = gt.lower()

    if norm.get("gnn_type") == "edge_mpnn" and norm.get("edge_dim") is None:
        # Be permissive: default to a single constant edge feature instead of crashing
        norm["edge_dim"] = 1
        if "logger" in globals():
            logger.warning(
                "edge_dim missing for edge_mpnn; defaulting to 1 (no edge features found)"
            )

    filtered = {k: v for k, v in norm.items() if (v is not None and k in sig_params)}
    extra = [k for k in norm if k not in sig_params]
    if extra and "logger" in globals():
        logger.debug("build_encoder: ignoring unsupported keys: %s", extra)
    enc = build_encoder(**filtered)

    # If edge features are missing/empty, pad them so forward() won’t blow up
    def _ensure_edge_attr(g, need_dim: int, device=None):
        """
        Ensure g.edge_attr exists and has shape (E, need_dim).
        Works whether g.x / g.edge_attr are numpy arrays or torch tensors.
        """
        import numpy as np

        try:
            import torch as _t

            _HAS_TORCH = True
        except Exception:
            _HAS_TORCH = False

        # 1) How many edges?
        E = 0
        ei = getattr(g, "edge_index", None)
        if ei is not None:
            try:
                E = int(ei.shape[1])
            except Exception:
                E = int(np.array(ei).shape[1])
        else:
            adj = getattr(g, "adj", None)
            if adj is not None:
                if _HAS_TORCH and isinstance(adj, _t.Tensor):
                    E = int((adj > 0).sum().item())
                else:
                    A = np.asarray(adj)
                    E = int((A > 0).sum())

        # 2) Build a zeros matrix of the *same type family* as g.x (numpy or torch)
        def _zeros_like_x(n_rows, n_cols):
            x = getattr(g, "x", None)
            if _HAS_TORCH and isinstance(x, _t.Tensor):
                dt = x.dtype
                dev = x.device if hasattr(x, "device") else device
                if dev is not None:
                    return _t.zeros((n_rows, n_cols), dtype=dt, device=dev)
                return _t.zeros((n_rows, n_cols), dtype=dt)
            # fallback: numpy
            dt = getattr(x, "dtype", np.float32)
            return np.zeros((n_rows, n_cols), dtype=dt)

        # 3) Create or fix edge_attr
        e = getattr(g, "edge_attr", None)
        e_w = getattr(e, "shape", (0, 0))[1] if e is not None else 0

        if e is None or e_w == 0:
            g.edge_attr = _zeros_like_x(E, need_dim)
            return g

        # 4) Pad / truncate to need_dim (handle both numpy and torch)
        if _HAS_TORCH and isinstance(e, _t.Tensor):
            if e.shape[1] < need_dim:
                pad = _zeros_like_x(e.shape[0], need_dim - e.shape[1])
                g.edge_attr = _t.cat([e, pad], dim=1)
            elif e.shape[1] > need_dim:
                g.edge_attr = e[:, :need_dim]
        else:
            e_np = np.asarray(e)
            if e_np.shape[1] < need_dim:
                pad = _zeros_like_x(e_np.shape[0], need_dim - e_np.shape[1])
                g.edge_attr = np.concatenate([e_np, pad], axis=1)
            elif e_np.shape[1] > need_dim:
                g.edge_attr = e_np[:, :need_dim]

        return g

    # load weights (prefer finetune's encoder substate; else sidecar)
    enc_sub = (state or {}).get("encoder", {})
    if not enc_sub and isinstance(side_state, dict):
        enc_sub = side_state
    _load_state_dict_forgiving(enc, enc_sub)

    # ---- build & load HEAD from checkpoint (infer shape from saved weights) ----
    import torch.nn as nn

    head_state = (state or {}).get("head", {})
    in_dim, out_dim = None, 1

    if isinstance(head_state, dict):
        for k, v in head_state.items():
            if k.endswith("weight") and getattr(v, "ndim", 0) == 2:
                out_dim, in_dim = v.shape  # [out, in]
                break

    # best-effort fallback for in_dim if weight shape wasn’t found
    if in_dim is None:
        in_dim = (
            enc_cfg.get("hidden_dim")
            or getattr(enc, "hidden_dim", None)
            or getattr(enc, "out_dim", None)
        )
    if in_dim is None:
        logger.error("Cannot infer head input dim; encoder_cfg=%s", enc_cfg)
        raise RuntimeError("Cannot infer head input dim")

    head = nn.Linear(int(in_dim), int(out_dim))
    if isinstance(head_state, dict) and head_state:
        _load_state_dict_forgiving(head, head_state)

    # move to device & eval
    enc = _to_dev(enc, device)
    head = _to_dev(head, device)
    enc.eval()
    head.eval()

    # use probabilities for classification; raw scores for regression
    task_is_cls = getattr(args, "task_type", "regression") == "classification"
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for start in range(0, len(dataset), args.batch_size):
        batch_indices = list(range(start, min(start + args.batch_size, len(dataset))))
        batch_x, batch_adj, batch_ptr, batch_labels = dataset.get_batch(batch_indices)

        batch_pos_np = None
        if hasattr(dataset, "graphs"):
            pos_blocks = []
            all_have_pos = True
            for idx in batch_indices:
                g_single = dataset.graphs[idx]
                pos_arr = getattr(g_single, "pos", None)
                if pos_arr is None:
                    all_have_pos = False
                    break
                pos_blocks.append(np.asarray(pos_arr, dtype=np.float32))
            if all_have_pos and pos_blocks:
                batch_pos_np = np.concatenate(pos_blocks, axis=0)

        batch_x = batch_x.to(device, non_blocking=True)
        batch_adj = batch_adj.to(device, non_blocking=True)
        # batch_ptr may be None; when present, pooling indices should be long
        batch_ptr = (
            batch_ptr.to(device, non_blocking=True).long()
            if batch_ptr is not None
            else None
        )

        with torch.no_grad():
            edge_idx = batch_adj.nonzero().T.detach().cpu().numpy()
            if edge_idx.size == 0:
                edge_idx = np.zeros((2, 0), dtype=np.int64)
            g = GraphData(
                x=batch_x.detach().cpu().numpy(),
                edge_index=edge_idx,
                pos=batch_pos_np,
            )

            if batch_ptr is not None:
                g.graph_ptr = batch_ptr.detach().cpu()

            g = _ensure_edge_attr(g, int(enc_cfg["edge_dim"]), device=device)
            node_emb = _encode_graph(enc, g)  # [N, D]
            # Guard against NaNs/Infs before pooling or batching
            node_emb = torch.nan_to_num(node_emb, nan=0.0, posinf=0.0, neginf=0.0)
            graph_emb = (
                global_mean_pool(node_emb, batch_ptr)
                if batch_ptr is not None
                else node_emb.mean(dim=0, keepdim=True)
            )
            logits = head(graph_emb).squeeze(1)
            # Clamp any stray non-finite values in the head output
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            preds_t = torch.sigmoid(logits) if task_is_cls else logits
            # And sanitize the post-activation tensor too (esp. regression path)
            preds_t = torch.nan_to_num(preds_t, nan=0.0, posinf=1e6, neginf=-1e6)
            all_preds.append(preds_t.detach().cpu().numpy())
            # targets: one value per graph in the batch (already aligned with graph_emb)
            all_targets.append(batch_labels.detach().cpu().numpy())

    # concat & filter non-finite rows (both y_true and y_pred)
    y_pred = np.concatenate(all_preds).astype(np.float32)
    y_true = np.concatenate(all_targets).astype(np.float32)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if task_is_cls:
        y_true = y_true.astype(np.int64, copy=False)
        # y_pred are probabilities in [0,1] for classification
        return compute_classification_metrics(y_true, y_pred)
    return compute_regression_metrics(y_true, y_pred)
