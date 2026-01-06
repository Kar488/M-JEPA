"""Supervised training routines for downstream tasks.

This module defines a function to train a simple linear head on top of a
frozen encoder for classification or regression tasks. The data is
split into train/validation/test sets using a stratified approach for
classification to ensure that each class appears in all splits when
possible. Performance metrics are computed using utilities from
`utils.metrics`.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import random
import re
import time as _time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_curve

from data.mdataset import GraphDataset
from data.scaffold_split import scaffold_split_indices
from explain.integrated_gradients import (
    aggregate_undirected_edge_scores,
    build_zero_baseline_graph,
    compute_integrated_gradients,
    describe_atom_types,
    describe_bond_types,
    normalise_attributions,
    render_molecule_heatmap,
)
from explain.motif_ig import (
    aggregate_motif_ig,
    compute_motif_deltas,
    find_motifs,
    save_motif_artifacts,
)
from models.encoder import GNNEncoder
from utils.early_stopping import EarlyStopping
from utils.dataset import SupportsTeardown
from utils.metrics import compute_classification_metrics, compute_regression_metrics
from utils.graph_ops import _encode_graph
from utils.ddp import should_retry_with_gloo
from utils.dataloader import (
    autotune_worker_pool,
    check_fd_budget,
    ensure_file_system_sharing_strategy,
    ensure_open_file_limit,
)

logger = logging.getLogger(__name__)

__all__ = ["stratified_split", "train_linear_head", "set_stage_config", "get_stage_config"]


_STAGE_CONFIG: Dict[str, Any] = {}


def set_stage_config(config: Optional[Dict[str, Any]]) -> None:
    """Register orchestrator-provided stage configuration for wall-clock guards."""

    global _STAGE_CONFIG

    if not config:
        _STAGE_CONFIG = {}
        return

    sanitized: Dict[str, Any] = {}
    for key, value in config.items():
        if value is None or isinstance(value, (int, float, str, bool)):
            sanitized[key] = value
            continue
        try:
            sanitized[key] = float(value)
        except Exception:
            try:
                sanitized[key] = str(value)
            except Exception:
                continue
    _STAGE_CONFIG = sanitized


def get_stage_config() -> Dict[str, Any]:
    """Return a shallow copy of the active stage configuration."""

    return dict(_STAGE_CONFIG)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float64))


def _prob_to_logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def _compute_pos_weight_from_labels(labels: np.ndarray) -> Optional[np.ndarray]:
    if labels.size == 0:
        return None
    arr = np.asarray(labels, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    valid = np.isfinite(arr)
    if not valid.any():
        return None
    pos_mask = (arr > 0) & valid
    neg_mask = (~pos_mask) & valid
    pos_counts = pos_mask.sum(axis=0).astype(np.float64)
    neg_counts = neg_mask.sum(axis=0).astype(np.float64)
    weights: List[float] = []
    for pos, neg in zip(pos_counts, neg_counts):
        if pos <= 0:
            weights.append(float("nan"))
            continue
        if neg <= 0:
            weights.append(1.0)
            continue
        weights.append(neg / pos)
    cleaned = [w for w in weights if math.isfinite(w) and w > 0]
    if not cleaned:
        return None
    filled = [w if (math.isfinite(w) and w > 0) else cleaned[0] for w in weights]
    return np.asarray(filled, dtype=np.float32)


def _build_layerwise_param_groups(module: nn.Module, base_lr: float, decay: float) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    stack: List[Tuple[nn.Module, int]] = [(module, 0)]
    while stack:
        mod, depth = stack.pop()
        params = list(mod.named_parameters(recurse=False))
        if params:
            lr = base_lr * (decay ** depth)
            lr = max(lr, 0.0)
            group_params = []
            for _, param in params:
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)
                group_params.append(param)
            if group_params:
                groups.append({"params": group_params, "lr": lr})
        for child in mod.children():
            stack.append((child, depth + 1))
    return groups


def _apply_threshold_offset(logits: np.ndarray, threshold: Optional[float]) -> np.ndarray:
    if threshold is None:
        return logits
    if threshold <= 0 or threshold >= 1:
        return logits
    odds = threshold / (1.0 - threshold)
    return logits - math.log(odds)


def _temperature_scale_logits(logits: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, float]:
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    temperature = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def _closure():
        optimizer.zero_grad()
        loss = loss_fn(logits_t / temperature, targets_t)
        loss.backward()
        return loss

    optimizer.step(_closure)
    temp_value = float(temperature.detach().clamp(min=1e-3).item())
    return logits / temp_value, temp_value


def _isotonic_calibration(logits: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, IsotonicRegression]:
    probs = _sigmoid_np(logits.reshape(-1))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, targets.reshape(-1))
    calibrated = iso.transform(probs)
    return calibrated.reshape(logits.shape), iso


def _tune_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f1") -> Tuple[Optional[float], Optional[float]]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() <= 0:
        return None, None
    y = y[mask]
    p = p[mask]
    if y.size == 0 or p.size == 0:
        return None, None

    best_threshold: Optional[float] = None
    best_score: Optional[float] = None
    metric = (metric or "f1").lower()

    if metric == "roc_auc":
        fpr, tpr, thresholds = roc_curve(y, p)
        if thresholds.size == 0:
            return None, None
        scores = tpr - fpr
        idx = int(np.argmax(scores))
        best_threshold = float(thresholds[idx])
        best_score = float(scores[idx])
    else:
        candidates = np.linspace(0.05, 0.95, 19, dtype=np.float64)
        for candidate in candidates:
            preds = (p >= candidate).astype(np.int64)
            try:
                score = f1_score(y.astype(np.int64), preds)
            except Exception:
                continue
            if not math.isfinite(score):
                continue
            if best_score is None or score > best_score:
                best_score = float(score)
                best_threshold = float(candidate)

    return best_threshold, best_score


def _resolve_cuda_spawn_context(device_type: str):
    """Return a torch multiprocessing context compatible with CUDA workers.

    When CUDA is initialised in the parent process, ``fork``-based DataLoader
    workers can raise ``Cannot re-initialize CUDA in forked subprocess``. To
    avoid that, prefer the ``spawn`` start method when building worker pools.

    Args:
        device_type: ``str`` device type (``"cuda"``/``"cpu"``/etc.).

    Returns:
        A ``torch.multiprocessing`` context configured for ``spawn`` when
        available, otherwise ``None``.
    """

    if device_type != "cuda":
        return None

    mp = getattr(torch, "multiprocessing", None)
    if mp is None:
        return None

    get_ctx = getattr(mp, "get_context", None)
    if get_ctx is None:
        return None

    try:
        return get_ctx("spawn")
    except RuntimeError:
        return None

# NOTE: keep in sync with experiments.case_study._to_list behaviour for tests.
def _as_index_list(indices: Iterable[int]) -> List[int]:
    """Convert an arbitrary index container into a plain ``list`` of ``int``."""

    if isinstance(indices, list):
        return [int(i) for i in indices]
    if isinstance(indices, np.ndarray):
        return indices.astype(int).reshape(-1).tolist()
    tolist = getattr(indices, "tolist", None)
    if callable(tolist):
        try:
            values = tolist()
            if isinstance(values, list):
                return [int(i) for i in values]
            if isinstance(values, tuple):
                return [int(i) for i in values]
        except Exception:
            pass
    return [int(i) for i in list(indices)]

# Test harness supprt
def _simple_pack_batch(dataset, batch_indices, task_type: str):
    """
    Builds a mini-batch from dataset.graphs where each graph g has:
      g.x: (n_i, F)
      g.edge_index: (2, E_i) optional
    Returns TORCH tensors:
      batch_x: (N, F) float32
      batch_adj: (N, N) float32 dense adjacency
      batch_ptr: (B+1,) int64
      batch_labels: (B,) float32 (for BCEWithLogits) or None
    """
    xs, edges_np, ptr = [], [], [0]
    node_offset = 0
    edge_blocks: List[torch.Tensor] = []  # type: ignore[var-annotated]
    pos_blocks: List[torch.Tensor] = []  # type: ignore[var-annotated]
    edge_attr_blocks: List[Optional[torch.Tensor]] = []  # type: ignore[var-annotated]
    edge_counts: List[int] = []
    all_have_pos = True

    for idx in batch_indices:
        g = dataset.graphs[idx]
        # node features -> np
        x = g.x
        if isinstance(x, np.ndarray):
            x_np = x
        elif hasattr(x, "detach"):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        xs.append(x_np)
        n = x_np.shape[0]

        # edges -> np
        ei = getattr(g, "edge_index", None)
        if ei is None:
            ei_np = np.zeros((2, 0), dtype=np.int64)
        elif isinstance(ei, np.ndarray):
            ei_np = ei
        elif hasattr(ei, "detach"):
            ei_np = ei.detach().cpu().numpy()
        else:
            ei_np = np.asarray(ei)
        if ei_np.size > 0:
            shifted = ei_np + node_offset
            edges_np.append(shifted)
            edge_blocks.append(torch.from_numpy(shifted.astype(np.int64, copy=False)))
        else:
            edge_blocks.append(torch.zeros((2, 0), dtype=torch.long))
        edge_counts.append(int(ei_np.shape[1]))

        edge_attr = getattr(g, "edge_attr", None)
        if edge_attr is None:
            edge_attr_blocks.append(None)
        else:
            if torch.is_tensor(edge_attr):
                edge_attr_t = edge_attr.detach().cpu()
            else:
                edge_attr_t = torch.as_tensor(edge_attr)
            if edge_attr_t.shape[0] != edge_counts[-1]:
                try:
                    edge_attr_t = edge_attr_t.reshape(edge_counts[-1], *edge_attr_t.shape[1:])
                except Exception:
                    edge_attr_t = None
            if edge_attr_t is not None and edge_attr_t.shape[0] == edge_counts[-1]:
                edge_attr_blocks.append(edge_attr_t)
            else:
                edge_attr_blocks.append(None)

        pos = getattr(g, "pos", None)
        if pos is None:
            all_have_pos = False
        else:
            pos_blocks.append(torch.as_tensor(pos, dtype=torch.float32))

        node_offset += n
        ptr.append(node_offset)

    X = np.concatenate(xs, axis=0).astype(np.float32)
    N = X.shape[0]

    # dense (N x N) adjacency expected by encoder (uses adj @ h)
    if edges_np:
        E = np.concatenate(edges_np, axis=1).astype(np.int64)
        adj = np.zeros((N, N), dtype=np.float32)
        adj[E[0], E[1]] = 1.0
    else:
        adj = np.zeros((N, N), dtype=np.float32)

    ptr_arr = np.asarray(ptr, dtype=np.int64)

    labels = getattr(dataset, "labels", None)
    if labels is None:
        batch_labels_t = None
    else:
        lab = np.asarray(labels)
        sel = np.asarray(batch_indices, dtype=np.int64)
        lbl = lab[sel].astype(np.float32, copy=False)  # BCE expects float
        batch_labels_t = torch.from_numpy(lbl)

    batch_edge_index = (
        torch.cat(edge_blocks, dim=1)
        if edge_blocks and any(e.numel() for e in edge_blocks)
        else torch.zeros((2, 0), dtype=torch.long)
    )

    batch_pos = None
    if all_have_pos and pos_blocks:
        batch_pos = torch.cat(pos_blocks, dim=0)

    batch_edge_attr = None
    edge_attr_template = next((ea for ea in edge_attr_blocks if ea is not None), None)
    if edge_attr_template is not None:
        attr_dtype = edge_attr_template.dtype
        attr_suffix = tuple(edge_attr_template.shape[1:])
        filled_attrs: List[torch.Tensor] = []
        for edge_count, edge_attr in zip(edge_counts, edge_attr_blocks):
            if edge_attr is None:
                zero_shape = (edge_count,) + attr_suffix
                filled_attrs.append(torch.zeros(zero_shape, dtype=attr_dtype))
            else:
                filled_attrs.append(edge_attr.to(dtype=attr_dtype))
        if filled_attrs:
            batch_edge_attr = torch.cat(filled_attrs, dim=0)
        else:
            batch_edge_attr = torch.zeros((0,) + attr_suffix, dtype=attr_dtype)

    extras = {
        "edge_index": batch_edge_index,
        "pos": batch_pos,
        "edge_attr": batch_edge_attr,
        "batch_indices": torch.as_tensor(batch_indices, dtype=torch.long),
    }

    return (
        torch.from_numpy(X),               # (N,F) float32
        torch.from_numpy(adj),             # (N,N) float32
        torch.from_numpy(ptr_arr),         # (B+1,) int64
        batch_labels_t,                    # (B,) float32 or None
        extras,
    )


class _GraphBatchCollator:
    """Callable that packs graph indices into batched tensors."""

    def __init__(self, dataset, task_type: str):
        self.dataset = dataset
        self.task_type = task_type

    def __call__(self, batch_indices):
        indices = [int(i) for i in batch_indices]
        if not indices:
            raise ValueError("GraphBatchCollator received an empty batch")
        if hasattr(self.dataset, "get_batch"):
            batch = self.dataset.get_batch(indices)
        else:
            batch = _simple_pack_batch(self.dataset, indices, self.task_type)
        if not isinstance(batch, (tuple, list)):
            raise ValueError("get_batch must return a tuple or list")

        if len(batch) == 5:
            batch_x, batch_adj, batch_ptr, batch_labels, extras = batch
        elif len(batch) == 4:
            batch_x, batch_adj, batch_ptr, batch_labels = batch
            extras = self._build_extras(indices)
        else:
            raise ValueError(
                "get_batch must return (x, adj, ptr, labels) optionally followed by extras"
            )

        extras_dict = dict(extras) if isinstance(extras, dict) else {}
        if "batch_indices" not in extras_dict:
            extras_dict["batch_indices"] = torch.as_tensor(indices, dtype=torch.long)

        return batch_x, batch_adj, batch_ptr, batch_labels, extras_dict

    def _build_extras(self, indices):
        edge_blocks: List[torch.Tensor] = []  # type: ignore[var-annotated]
        pos_blocks: List[torch.Tensor] = []  # type: ignore[var-annotated]
        edge_attr_blocks: List[Optional[torch.Tensor]] = []  # type: ignore[var-annotated]
        edge_counts: List[int] = []
        all_have_pos = True
        node_offset = 0

        for idx in indices:
            g = self.dataset.graphs[idx]

            n_i = None
            if hasattr(g, "num_nodes"):
                try:
                    n_i = int(g.num_nodes())
                except Exception:
                    n_i = None
            if n_i is None:
                x_attr = getattr(g, "x", None)
                if x_attr is not None:
                    n_i = int(np.asarray(x_attr).shape[0])
            if n_i is None:
                try:
                    x_i, _ = g.to_tensors()
                    n_i = int(x_i.shape[0])
                except Exception:
                    n_i = 0

            edge_index = getattr(g, "edge_index", None)
            current_edge_count = 0
            if edge_index is None:
                try:
                    _, adj_i = g.to_tensors()
                except Exception:
                    adj_i = None
                if adj_i is not None:
                    adj_t = torch.as_tensor(adj_i)
                    idxs = (adj_t > 0).nonzero(as_tuple=False).T
                    idxs = idxs.to(dtype=torch.long)
                    current_edge_count = int(idxs.shape[1])
                    edge_blocks.append(idxs + node_offset)
                else:
                    edge_blocks.append(torch.zeros((2, 0), dtype=torch.long))
                    current_edge_count = 0
            else:
                ei_t = torch.as_tensor(edge_index, dtype=torch.long)
                current_edge_count = int(ei_t.shape[1])
                edge_blocks.append(ei_t + node_offset)
            edge_counts.append(current_edge_count)

            edge_attr = getattr(g, "edge_attr", None)
            if edge_attr is None:
                edge_attr_blocks.append(None)
            else:
                if torch.is_tensor(edge_attr):
                    edge_attr_t = edge_attr.detach().cpu()
                else:
                    edge_attr_t = torch.as_tensor(edge_attr)
                if edge_attr_t.shape[0] != current_edge_count:
                    try:
                        edge_attr_t = edge_attr_t.reshape(current_edge_count, *edge_attr_t.shape[1:])
                    except Exception:
                        edge_attr_t = None
                if edge_attr_t is not None and edge_attr_t.shape[0] == current_edge_count:
                    edge_attr_blocks.append(edge_attr_t)
                else:
                    edge_attr_blocks.append(None)

            pos = getattr(g, "pos", None)
            if pos is None:
                all_have_pos = False
            else:
                pos_blocks.append(torch.as_tensor(pos, dtype=torch.float32))

            node_offset += n_i if n_i is not None else 0

        batch_edge_index = (
            torch.cat(edge_blocks, dim=1)
            if edge_blocks and any(e.numel() for e in edge_blocks)
            else torch.zeros((2, 0), dtype=torch.long)
        )

        batch_pos = None
        if all_have_pos and pos_blocks:
            batch_pos = torch.cat(pos_blocks, dim=0)

        batch_edge_attr = None
        edge_attr_template = next((ea for ea in edge_attr_blocks if ea is not None), None)
        if edge_attr_template is not None:
            attr_dtype = edge_attr_template.dtype
            attr_suffix = tuple(edge_attr_template.shape[1:])
            filled_attrs: List[torch.Tensor] = []
            for edge_count, edge_attr in zip(edge_counts, edge_attr_blocks):
                if edge_attr is None:
                    zero_shape = (edge_count,) + attr_suffix
                    filled_attrs.append(torch.zeros(zero_shape, dtype=attr_dtype))
                else:
                    filled_attrs.append(edge_attr.to(dtype=attr_dtype))
            if filled_attrs:
                batch_edge_attr = torch.cat(filled_attrs, dim=0)
            else:
                batch_edge_attr = torch.zeros((0,) + attr_suffix, dtype=attr_dtype)

        extras = {
            "edge_index": batch_edge_index,
            "pos": batch_pos,
            "edge_attr": batch_edge_attr,
            "batch_indices": torch.as_tensor(indices, dtype=torch.long),
        }
        return extras


def _extract_batch_indices(batch_meta) -> Optional[List[int]]:
    """Return dataset indices stored in ``batch_meta`` if present."""

    if not isinstance(batch_meta, dict):
        return None
    raw = batch_meta.get("batch_indices")
    if raw is None:
        return None
    if torch.is_tensor(raw):
        return [int(x) for x in raw.detach().cpu().view(-1).tolist()]
    if isinstance(raw, np.ndarray):
        return [int(x) for x in raw.reshape(-1).tolist()]
    return [int(x) for x in raw]


def _normalise_explain_mode(mode: Optional[str]) -> str:
    """Canonicalise explanation modes and accept common aliases."""

    mode_norm = (mode or "").strip().lower()
    if mode_norm == "motif_ig":
        return "ig_motif"
    return mode_norm


def _normalise_explain_modes(mode: Optional[Union[str, Iterable[str]]]) -> List[str]:
    """Return a de-duplicated list of canonical explanation modes.

    Accepts comma/whitespace-delimited strings or any iterable of modes and
    normalises motif aliases to ``ig_motif``.
    """

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
        token_norm = _normalise_explain_mode(token)
        if token_norm and token_norm not in normalised:
            normalised.append(token_norm)
    return normalised


def _move_batch_to_device(batch, device: torch.device, non_blocking: bool):
    """Move a batched graph tuple returned by ``get_batch`` to ``device``."""

    if len(batch) == 5:
        batch_x, batch_adj, batch_ptr, batch_labels, extras = batch
    elif len(batch) == 4:
        batch_x, batch_adj, batch_ptr, batch_labels = batch
        extras = {}
    else:
        raise ValueError("Expected batch of length 4 or 5")

    batch_x = batch_x.to(device=device, non_blocking=non_blocking)
    batch_adj = batch_adj.to(device=device, non_blocking=non_blocking)
    batch_ptr = batch_ptr.to(device=device, non_blocking=non_blocking)
    batch_labels = (
        batch_labels.to(device=device, non_blocking=non_blocking)
        if batch_labels is not None
        else None
    )

    edge_index = None
    pos = None
    edge_attr = None
    if isinstance(extras, dict):
        edge_index = extras.get("edge_index")
        pos = extras.get("pos")
        edge_attr = extras.get("edge_attr")
    if edge_index is not None:
        edge_index = edge_index.to(device=device, non_blocking=non_blocking)
    if pos is not None:
        pos = pos.to(device=device, non_blocking=non_blocking)

    if edge_attr is not None:
        if torch.is_tensor(edge_attr):
            edge_attr = edge_attr.to(device=device, non_blocking=non_blocking)
        else:
            edge_attr = torch.as_tensor(edge_attr, device=device)

    batch_indices = extras.get("batch_indices") if isinstance(extras, dict) else None

    moved_extras = {
        "edge_index": edge_index,
        "pos": pos,
        "edge_attr": edge_attr,
        "batch_indices": batch_indices,
    }
    return batch_x, batch_adj, batch_ptr, batch_labels, moved_extras


def _ensure_binary_classification_logits(logits: torch.Tensor) -> torch.Tensor:
    """Reduce head outputs to a single logit per sample for binary tasks.

    ``train_linear_head`` historically expected classification heads to emit a
    single logit.  Some call-sites now provide two logits (negative/positive)
    which breaks the ``BCEWithLogitsLoss`` contract.  This helper squeezes
    singleton channels and, when multiple logits are provided, converts them to
    an equivalent positive-class logit by comparing the positive logit against
    the log-sum-exp of the remaining classes.
    """

    if logits.ndim == 0:
        return logits.reshape(1)
    if logits.ndim == 1:
        return logits

    # ``logits`` has at least two dimensions. Interpret the last dimension as
    # the class axis and collapse to a single logit.
    if logits.size(-1) == 1:
        return logits.squeeze(-1)

    if logits.size(-1) >= 2:
        pos_logit = logits.select(-1, logits.size(-1) - 1)
        neg_logsumexp = torch.logsumexp(logits[..., :-1], dim=-1)
        return pos_logit - neg_logsumexp

    raise ValueError(
        "Classification head produced an unexpected shape: "
        f"{tuple(logits.shape)}"
    )

def _build_graph_view(
    batch_x: torch.Tensor,
    batch_adj: torch.Tensor,
    batch_ptr: Optional[torch.Tensor],
    batch_meta: object,
) -> SimpleNamespace:
    """Assemble a lightweight graph object for ``_encode_graph``.

    The encoder APIs in :mod:`models.gnn_variants` expect ``g.edge_index`` to
    exist even when the downstream dataset only materialises dense adjacencies.
    This helper promotes the optional metadata returned by
    :func:`_move_batch_to_device` into attributes on a ``SimpleNamespace`` and
    synthesises a zero‑edge ``edge_index`` tensor when the loader cannot
    provide one.  3D coordinates (``pos``) are normalised to ``float32`` on the
    same device as ``batch_x`` so encoders such as :class:`SchNet3D` receive
    consistent geometry information.  ``batch_ptr`` is carried through when
    present so pooled readouts continue to function for batched graphs.
    """

    extras = batch_meta if isinstance(batch_meta, dict) else {}

    edge_index = extras.get("edge_index") if isinstance(extras, dict) else None
    device = batch_x.device
    if edge_index is not None:
        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device=device, dtype=torch.long)
    else:
        if isinstance(batch_adj, torch.Tensor) and batch_adj.numel() > 0:
            idx = (batch_adj > 0).nonzero(as_tuple=False).T
            edge_index = idx.to(device=device, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    pos = extras.get("pos") if isinstance(extras, dict) else None
    if pos is not None:
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos, dtype=torch.float32, device=device)
        else:
            pos = pos.to(device=device, dtype=torch.float32)

    graph_obj = SimpleNamespace(
        x=batch_x,
        adj=batch_adj,
        edge_index=edge_index,
        pos=pos,
        graph_ptr=batch_ptr,
    )

    edge_attr = extras.get("edge_attr") if isinstance(extras, dict) else None
    if edge_attr is not None:
        if not torch.is_tensor(edge_attr):
            edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32, device=device)
        else:
            edge_attr = edge_attr.to(device=device)
        graph_obj.edge_attr = edge_attr

    return graph_obj

def _pool_batch_embeddings(node_embeddings: torch.Tensor, batch_ptr: torch.Tensor) -> torch.Tensor:
    """Average node embeddings per graph using pointer offsets."""

    if node_embeddings.numel() == 0:
        return node_embeddings.new_zeros((0, node_embeddings.shape[-1]))

    if batch_ptr.numel() <= 1:
        return node_embeddings.mean(dim=0, keepdim=True)

    device = node_embeddings.device
    ptr = batch_ptr.to(device=device)
    lengths = ptr[1:] - ptr[:-1]

    total_nodes = node_embeddings.shape[0]
    if int(lengths.sum().item()) != total_nodes:
        raise ValueError(
            "batch_ptr does not describe the provided node embeddings: "
            f"expected {int(lengths.sum().item())} nodes but received {total_nodes}"
        )

    num_graphs = lengths.numel()
    graph_emb = node_embeddings.new_zeros((num_graphs, node_embeddings.shape[-1]))

    # ``torch.bucketize`` works for both integer and floating point pointers.
    node_positions = torch.arange(total_nodes, device=device, dtype=ptr.dtype)
    boundaries = ptr[1:]
    graph_ids = torch.bucketize(node_positions, boundaries, right=True).to(torch.long)
    graph_emb.index_add_(0, graph_ids, node_embeddings)

    denom = lengths.to(node_embeddings.dtype).unsqueeze(-1).clamp_min(1)
    graph_emb = graph_emb / denom
    return graph_emb


class _IGArtifactLogger:
    """Manage Integrated Gradients computation and artifact export."""

    def __init__(
        self,
        *,
        dataset: GraphDataset,
        encoder: nn.Module,
        head_module: nn.Module,
        task_type: str,
        device: torch.device,
        explain_mode: Optional[str],
        explain_config: Optional[Dict[str, Any]],
        stage_config: Optional[Dict[str, Any]],
    ) -> None:
        mode = _normalise_explain_mode(explain_mode)
        self.enabled = mode == "ig"
        if not self.enabled:
            return

        self.dataset = dataset
        self.encoder = encoder
        self.head = head_module
        self.task_type = task_type
        self.device = device
        self._seen: Set[int] = set()
        self._warned_missing_idx = False
        self.records: List[Dict[str, Any]] = []
        self.smiles = getattr(dataset, "smiles", None)
        config = dict(explain_config or {})
        stage_cfg = dict(stage_config or {})

        try:
            steps = int(config.get("steps", 50))
        except Exception:
            steps = 50
        self.steps = max(1, steps)
        self.normalise_mode = str(config.get("normalise", "signed")).strip().lower()

        self.task_name = str(
            config.get("task_name")
            or stage_cfg.get("task_name")
            or stage_cfg.get("task")
            or "task"
        )

        base_dir = (
            config.get("output_dir")
            or stage_cfg.get("report_dir")
            or stage_cfg.get("reports_dir")
            or stage_cfg.get("stage_dir")
            or os.path.join(os.getcwd(), "reports")
        )
        base_dir = os.path.abspath(str(base_dir))
        self.output_dir = os.path.join(base_dir, "ig_explanations", self.task_name)
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("[ig] exporting explanations to %s", self.output_dir)

    def _format_identifier(self, idx: int, smiles: Optional[str]) -> str:
        base = f"graph_{idx:05d}"
        if smiles:
            slug = re.sub(r"[^0-9A-Za-z]+", "_", smiles).strip("_")
            if slug:
                slug = slug[:48]
                return f"{base}_{slug}"
        return base

    def _direction(self, value: float) -> str:
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return "neutral"

    def _model_callable(self) -> Callable[[SimpleNamespace], torch.Tensor]:
        encoder = self.encoder
        head = self.head
        task_type = self.task_type

        def _forward(graph_obj: SimpleNamespace) -> torch.Tensor:
            node_embeddings = _encode_graph(encoder, graph_obj)
            ptr = getattr(graph_obj, "graph_ptr", None)
            if ptr is None:
                num_nodes = int(graph_obj.x.shape[0])
                ptr = torch.tensor([0, num_nodes], dtype=torch.long, device=self.device)
            graph_emb = _pool_batch_embeddings(node_embeddings, ptr)
            output = head(graph_emb)
            if task_type == "classification":
                output = _ensure_binary_classification_logits(output)
            else:
                output = output.squeeze(-1)
            return output.reshape(-1)[0]

        return _forward

    def _write_csv(self, path: str, rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "type", "ig_score", "direction"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _compute_for_index(self, idx: int) -> None:
        if idx in self._seen:
            return
        try:
            graph = self.dataset.graphs[idx]
        except Exception:
            logger.warning("[ig] failed to load graph %d", idx, exc_info=True)
            return

        baseline = build_zero_baseline_graph(graph)
        model_fn = self._model_callable()

        try:
            with torch.enable_grad():
                node_scores, edge_scores = compute_integrated_gradients(
                    model_fn,
                    graph,
                    baseline_graph=baseline,
                    m_steps=self.steps,
                    device=self.device,
                )
        except Exception:
            logger.warning("[ig] attribution failed for graph %d", idx, exc_info=True)
            return

        self._seen.add(idx)
        smiles = self.smiles[idx] if self.smiles and idx < len(self.smiles) else None
        molecule_id = self._format_identifier(idx, smiles)

        node_rows = [
            {
                "id": f"{idx}_{node_idx}",
                "type": atom_type,
                "ig_score": float(score),
                "direction": self._direction(float(score)),
            }
            for node_idx, (atom_type, score) in enumerate(
                zip(describe_atom_types(smiles, len(node_scores)), node_scores)
            )
        ]

        undirected = aggregate_undirected_edge_scores(getattr(graph, "edge_index", None), edge_scores)
        bond_pairs = list(undirected.keys())
        bond_types = describe_bond_types(smiles, bond_pairs)
        bond_rows = [
            {
                "id": f"{i}-{j}",
                "type": bond_types.get((i, j), f"bond_{i}_{j}"),
                "ig_score": float(score),
                "direction": self._direction(float(score)),
            }
            for (i, j), score in undirected.items()
        ]

        atom_csv = os.path.join(self.output_dir, f"{molecule_id}_atoms.csv")
        bond_csv = os.path.join(self.output_dir, f"{molecule_id}_bonds.csv")
        heatmap_png = os.path.join(self.output_dir, f"{molecule_id}_heatmap.png")

        self._write_csv(atom_csv, node_rows)
        self._write_csv(bond_csv, bond_rows)

        atom_norm = normalise_attributions(node_scores, mode=self.normalise_mode)
        bond_norm: Dict[Tuple[int, int], float] = {}
        if bond_pairs:
            bond_norm_values = normalise_attributions(
                [undirected[pair] for pair in bond_pairs], mode=self.normalise_mode
            )
            for pair, value in zip(bond_pairs, bond_norm_values):
                bond_norm[pair] = float(value)

        render_molecule_heatmap(smiles, atom_norm, bond_norm, heatmap_png)

        self.records.append(
            {
                "graph_index": idx,
                "molecule_id": molecule_id,
                "atom_csv": atom_csv,
                "bond_csv": bond_csv,
                "heatmap_png": heatmap_png,
            }
        )

    def process_batch(self, batch_meta: object) -> None:
        if not self.enabled:
            return
        indices = _extract_batch_indices(batch_meta)
        if not indices:
            if not self._warned_missing_idx:
                logger.warning(
                    "[ig] batch metadata missing dataset indices; skipping explanation export."
                )
                self._warned_missing_idx = True
            return
        for idx in indices:
            self._compute_for_index(idx)

    def finalize(self, metrics: Dict[str, Any]) -> None:
        if not self.enabled or not self.records:
            return
        metrics["ig_artifact_root"] = self.output_dir
        metrics["ig_artifacts"] = [dict(entry) for entry in self.records]


class _MotifIGArtifactLogger:
    """Manage motif-level IG aggregation and artifact export."""

    def __init__(
        self,
        *,
        dataset: GraphDataset,
        encoder: nn.Module,
        head_module: nn.Module,
        task_type: str,
        device: torch.device,
        explain_mode: Optional[str],
        explain_config: Optional[Dict[str, Any]],
        stage_config: Optional[Dict[str, Any]],
    ) -> None:
        mode = _normalise_explain_mode(explain_mode)
        self.enabled = mode == "ig_motif"
        if not self.enabled:
            return

        self.dataset = dataset
        self.encoder = encoder
        self.head = head_module
        self.task_type = task_type
        self.device = device
        self._seen: Set[int] = set()
        self._warned_missing_idx = False
        self.records: List[Dict[str, Any]] = []
        self.smiles = getattr(dataset, "smiles", None)
        config = dict(explain_config or {})
        stage_cfg = dict(stage_config or {})

        try:
            steps = int(config.get("steps", 50))
        except Exception:
            steps = 50
        self.steps = max(1, steps)
        self.normalise_mode = str(config.get("normalise", "signed")).strip().lower()
        raw_task_names = config.get("task_names") or stage_cfg.get("task_names")
        if isinstance(raw_task_names, str):
            raw_task_names = [raw_task_names]
        self.task_names = [str(name) for name in raw_task_names] if raw_task_names else []

        self.task_name = str(
            config.get("task_name")
            or stage_cfg.get("task_name")
            or stage_cfg.get("task")
            or "task"
        )

        base_dir = (
            config.get("output_dir")
            or stage_cfg.get("report_dir")
            or stage_cfg.get("reports_dir")
            or stage_cfg.get("stage_dir")
            or os.path.join(os.getcwd(), "reports")
        )
        base_dir = os.path.abspath(str(base_dir))
        self.output_dir = os.path.join(base_dir, "ig_motif_explanations", self.task_name)
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("[ig_motif] exporting explanations to %s", self.output_dir)

    def _format_identifier(self, idx: int, smiles: Optional[str]) -> str:
        base = f"graph_{idx:05d}"
        if smiles:
            slug = re.sub(r"[^0-9A-Za-z]+", "_", smiles).strip("_")
            if slug:
                slug = slug[:48]
                return f"{base}_{slug}"
        return base

    def _graph_to_namespace(self, graph: GraphData) -> SimpleNamespace:
        ptr = torch.tensor([0, graph.num_nodes()], dtype=torch.long, device=self.device)
        ns = SimpleNamespace(
            x=torch.as_tensor(graph.x, dtype=torch.float32, device=self.device),
            edge_index=torch.as_tensor(
                getattr(graph, "edge_index", np.zeros((2, 0), dtype=np.int64)),
                dtype=torch.long,
                device=self.device,
            ),
            graph_ptr=ptr,
        )
        if getattr(graph, "edge_attr", None) is not None:
            ns.edge_attr = torch.as_tensor(graph.edge_attr, dtype=torch.float32, device=self.device)
        if getattr(graph, "pos", None) is not None:
            ns.pos = torch.as_tensor(graph.pos, dtype=torch.float32, device=self.device)
        return ns

    def _model_callable_for_ig(self) -> Callable[[SimpleNamespace], torch.Tensor]:
        encoder = self.encoder
        head = self.head
        task_type = self.task_type

        def _forward(graph_obj: SimpleNamespace) -> torch.Tensor:
            node_embeddings = _encode_graph(encoder, graph_obj)
            ptr = getattr(graph_obj, "graph_ptr", None)
            if ptr is None:
                num_nodes = int(graph_obj.x.shape[0])
                ptr = torch.tensor([0, num_nodes], dtype=torch.long, device=self.device)
            graph_emb = _pool_batch_embeddings(node_embeddings, ptr)
            output = head(graph_emb)
            if task_type == "classification":
                output = _ensure_binary_classification_logits(output)
            else:
                output = output.squeeze(-1)
            return output.reshape(-1)[0]

        return _forward

    def _model_logits_callable(self) -> Callable[[SimpleNamespace], torch.Tensor]:
        encoder = self.encoder
        head = self.head
        task_type = self.task_type

        def _forward(graph_obj: SimpleNamespace) -> torch.Tensor:
            node_embeddings = _encode_graph(encoder, graph_obj)
            ptr = getattr(graph_obj, "graph_ptr", None)
            if ptr is None:
                num_nodes = int(graph_obj.x.shape[0])
                ptr = torch.tensor([0, num_nodes], dtype=torch.long, device=self.device)
            graph_emb = _pool_batch_embeddings(node_embeddings, ptr)
            output = head(graph_emb)
            if task_type == "classification":
                output = _ensure_binary_classification_logits(output)
            else:
                output = output.squeeze(-1)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            if output.dim() == 1:
                return output
            return output.reshape(-1)

        return _forward

    def _resolve_task_names(self, baseline_logits: np.ndarray) -> List[str]:
        if self.task_names:
            return self.task_names
        length = int(baseline_logits.reshape(-1).shape[0])
        return [f"task_{i}" for i in range(length)]

    def _compute_for_index(self, idx: int) -> None:
        if idx in self._seen:
            return
        try:
            graph = self.dataset.graphs[idx]
        except Exception:
            logger.warning("[ig_motif] failed to load graph %d", idx, exc_info=True)
            return

        baseline = build_zero_baseline_graph(graph)
        ig_model = self._model_callable_for_ig()
        logits_model = self._model_logits_callable()

        try:
            with torch.enable_grad():
                node_scores, edge_scores = compute_integrated_gradients(
                    ig_model,
                    graph,
                    baseline_graph=baseline,
                    m_steps=self.steps,
                    device=self.device,
                )
        except Exception:
            logger.warning("[ig_motif] attribution failed for graph %d", idx, exc_info=True)
            return

        self._seen.add(idx)
        smiles = self.smiles[idx] if self.smiles and idx < len(self.smiles) else None
        molecule_id = self._format_identifier(idx, smiles)

        # FIX A: Prefer explicit SMILES for motif extraction
        motif_map = find_motifs(smiles) if smiles else find_motifs(graph)

        # Optional: attach SMILES to graph for downstream explainability
        if smiles and not getattr(graph, "smiles", None):
            try:
                graph.smiles = smiles
            except Exception:
                pass
        motif_scores = aggregate_motif_ig(
            node_scores,
            edge_scores,
            motif_map,
            getattr(graph, "edge_index", None),
        )

        baseline_ns = self._graph_to_namespace(graph)
        with torch.no_grad():
            baseline_logits = torch.as_tensor(logits_model(baseline_ns), dtype=torch.float32, device=self.device)
        baseline_np = baseline_logits.detach().cpu().numpy().astype(np.float32, copy=False)

        motif_deltas = compute_motif_deltas(
            logits_model,
            graph,
            motif_map,
            device=self.device,
            baseline_logits=baseline_np,
        )
        task_names = self._resolve_task_names(baseline_np)

        graph_dir = os.path.join(self.output_dir, molecule_id)
        artifacts = save_motif_artifacts(
            smiles=smiles,
            motif_map=motif_map,
            motif_scores=motif_scores,
            motif_deltas=motif_deltas,
            task_names=task_names,
            output_dir=graph_dir,
            normalise_mode=self.normalise_mode,
        )

        self.records.append(
            {
                "graph_index": idx,
                "molecule_id": molecule_id,
                "artifact_dir": graph_dir,
                **{k: v for k, v in artifacts.items() if k != "motif_rows"},
            }
        )

    def process_batch(self, batch_meta: object) -> None:
        if not self.enabled:
            return
        indices = _extract_batch_indices(batch_meta)
        if not indices:
            if not self._warned_missing_idx:
                logger.warning(
                    "[ig_motif] batch metadata missing dataset indices; skipping explanation export."
                )
                self._warned_missing_idx = True
            return
        for idx in indices:
            self._compute_for_index(idx)

    def finalize(self, metrics: Dict[str, Any]) -> None:
        if not self.enabled or not self.records:
            return
        metrics["ig_motif_artifact_root"] = self.output_dir
        metrics["ig_motif_artifacts"] = [dict(entry) for entry in self.records]


def _build_explain_loggers(
    *,
    modes: Iterable[str],
    dataset: GraphDataset,
    encoder: nn.Module,
    head_module: nn.Module,
    task_type: str,
    device: torch.device,
    explain_config: Optional[Dict[str, Any]],
    stage_config: Optional[Dict[str, Any]],
) -> List[Union[_IGArtifactLogger, _MotifIGArtifactLogger]]:
    """Instantiate the requested explanation loggers."""

    loggers: List[Union[_IGArtifactLogger, _MotifIGArtifactLogger]] = []
    for mode in modes:
        logger_cls: Optional[Union[type[_IGArtifactLogger], type[_MotifIGArtifactLogger]]] = None
        if mode == "ig":
            logger_cls = _IGArtifactLogger
        elif mode == "ig_motif":
            logger_cls = _MotifIGArtifactLogger
        if logger_cls is None:
            continue
        logger = logger_cls(
            dataset=dataset,
            encoder=encoder,
            head_module=head_module,
            task_type=task_type,
            device=device,
            explain_mode=mode,
            explain_config=explain_config,
            stage_config=stage_config,
        )
        if getattr(logger, "enabled", False):
            loggers.append(logger)
    return loggers

def stratified_split(
    indices: List[int], labels: np.ndarray, train_frac: float, val_frac: float
) -> Tuple[List[int], List[int], List[int]]:
    """Perform a stratified split for binary classification.

    If labels contain only one class, fall back to a random split.

    Args:
        indices: List of indices to split.
        labels: Array of labels corresponding to the indices.
        train_frac: Fraction of samples to allocate to the training set.
        val_frac: Fraction of samples to allocate to the validation set.

    Returns:
        train_idx, val_idx, test_idx: Three lists of indices.
    """
    unique = np.unique(labels)
    logger.debug(
        "stratified_split with %d indices and classes %s",
        len(indices),
        unique,
    )
    if len(unique) < 2:
        # Only one class present; perform a random split
        random.shuffle(indices)
        n = len(indices)
        train_end = int(train_frac * n)
        val_end = int((train_frac + val_frac) * n)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]
    # Separate indices by class
    class0 = [idx for idx in indices if labels[idx] == 0]
    class1 = [idx for idx in indices if labels[idx] == 1]
    random.shuffle(class0)
    random.shuffle(class1)

    def split_class(class_indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
        n = len(class_indices)
        train_end = int(train_frac * n)
        val_end = int((train_frac + val_frac) * n)
        return (
            class_indices[:train_end],
            class_indices[train_end:val_end],
            class_indices[val_end:],
        )

    c0_train, c0_val, c0_test = split_class(class0)
    c1_train, c1_val, c1_test = split_class(class1)
    train_idx = c0_train + c1_train
    val_idx = c0_val + c1_val
    test_idx = c0_test + c1_test
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)
    return train_idx, val_idx, test_idx

def _dataset_size(dataset) -> int:
    """Return the number of graphs in ``dataset`` with sensible fallbacks."""

    try:
        return len(dataset)  # type: ignore[arg-type]
    except TypeError:
        pass

    for attr_name in ("graphs", "data", "items"):
        sized_attr = getattr(dataset, attr_name, None)
        if sized_attr is None:
            continue
        try:
            return len(sized_attr)
        except TypeError:
            continue

    raise TypeError(
        "Dataset of type "
        f"{type(dataset).__name__} does not provide a length and lacks a sized "
        "attribute among: graphs, data, items"
    )


def _train_linear_head_impl(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    task_type: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda",
    patience: int = 10,
    num_workers: int = -1,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    bf16=False,
    use_scaffold: bool = False,
    devices: int = 1,
    *,
    max_batches: int = 0,
    time_budget_mins: int = 0,
    head: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    encoder_lr: Optional[float] = None,
    head_lr: Optional[float] = None,
    freeze_encoder: bool = True,
    early_stop_metric: str = "val_loss",
    early_stop_mode: Optional[str] = None,
    cache_graph_embeddings: bool = True,
    train_indices: Optional[Iterable[int]] = None,
    val_indices: Optional[Iterable[int]] = None,
    test_indices: Optional[Iterable[int]] = None,
    enable_batch_autoscale: bool = False,
    batch_autoscale_min_steps: int = 10,
    batch_autoscale_floor: int = 64,
    unfreeze_top_layers: int = 0,
    stage_config: Optional[Dict[str, Any]] = None,
    pos_weight: Optional[Any] = None,
    class_weight: Optional[Any] = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    dynamic_pos_weight: bool = False,
    oversample_minority: bool = False,
    layerwise_decay: Optional[float] = None,
    calibrate_probabilities: bool = False,
    calibration_method: str = "temperature",
    threshold_metric: str = "f1",
    explain_mode: Optional[Union[str, Iterable[str]]] = None,
    explain_config: Optional[Dict[str, Any]] = None,
    **unused,
) -> Dict[str, Any]:
    """Train a linear head on a frozen encoder for classification or regression.

    When ``devices > 1`` the encoder and head are wrapped with
    :class:`~torch.nn.parallel.DistributedDataParallel` and gradients are
    synchronised across ranks. Validation loss for early stopping is
    averaged across all processes to ensure a consistent stopping epoch.

    Imbalance-handling controls (``pos_weight``, ``class_weight``,
    ``dynamic_pos_weight``, ``oversample_minority``, ``use_focal_loss`` and
    ``focal_gamma``) apply only to BCE classification heads. For heavily
    imbalanced tasks such as Tox21, enabling focal loss with ``focal_gamma=2.0``
    alongside ``oversample_minority=True`` generally stabilises ROC-AUC without
    overfitting the majority class.

    Args:
        dataset: A ``GraphDataset`` with labels.
        encoder: A pre‑trained GNN encoder.
        task_type: Either ``"classification"`` or ``"regression"``.
        epochs: Maximum number of epochs.
        lr: Learning rate for the optimiser.
        batch_size: Batch size for training.
        device: Computation device.
        num_workers: Number of subprocesses to use for data loading.  Pass ``-1``
            to auto-tune based on dataset size and available CPU cores.
        pin_memory: if true data transfer from CPU → GPU faster.
        persistent_workers: if true avoids the overhead of respawning worker processes every epoch.
        prefetch_factor: number of batches loaded in advance by each worker.
        bf16: if true mixed precision training on newer GPUs/TPUs
        patience: Number of epochs with no improvement before stopping.
        use_scaffold: Whether to use scaffold split if SMILES are provided.
        devices: Number of GPUs for DDP.
        batch_indices: Optional list of indices for a single batch; if provided, overrides internal splitting.
        max batches: Maximum number of batches to train on; if set, overrides epochs
        time_budget_mins: Time budget in minutes for the training; if set, overrides epochs.
        head: Optional linear head module to optimise.  When ``None`` a
            single-layer head is constructed internally.
        optimizer: Optional optimiser to use for the head (and any
            unfrozen encoder parameters).  A new optimiser is created if
            one is not supplied.
        scheduler: Optional learning-rate scheduler that steps after
            each epoch, provided at least one optimisation step ran.
        cache_graph_embeddings: When ``True`` (default) encoder outputs
            are cached per-graph so subsequent epochs reuse precomputed
            embeddings instead of re-encoding the frozen backbone.
    Returns:
        A dictionary of metrics on the test set (only populated on rank 0).
    """

    assert dataset.labels is not None, "Dataset must have labels."
    assert task_type in {"classification", "regression"}

    from utils.ddp import (
        cleanup,
        get_rank,
        get_world_size,
        init_distributed,
        is_main_process,
    )

    distributed = False
    if devices > 1:
        try:
            distributed = init_distributed()
        except RuntimeError as exc:
            if should_retry_with_gloo(exc):
                cleanup()
                raise

            logger.warning(
                "Distributed initialisation failed (requested devices=%s); "
                "falling back to single-process execution.",
                devices,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            cleanup()
            distributed = False
            devices = 1
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
        else:
            if distributed:
                dist_mod = getattr(torch, "distributed", None)
                is_initialised = bool(
                    dist_mod is not None
                    and getattr(dist_mod, "is_available", lambda: False)()
                    and getattr(dist_mod, "is_initialized", lambda: False)()
                )
                if not is_initialised:
                    logger.warning(
                        "Distributed initialisation reported success but no process group is active; "
                        "falling back to single-process execution.",
                    )
                    cleanup()
                    distributed = False
                    devices = 1
                    os.environ["WORLD_SIZE"] = "1"
                    os.environ["LOCAL_WORLD_SIZE"] = "1"
                    os.environ["RANK"] = "0"
                    os.environ["LOCAL_RANK"] = "0"

    device_t = torch.device(device)
    ddp_device_index: Optional[int] = None
    if device_t.type == "cuda":
        cuda_mod = getattr(torch, "cuda", None)
        is_available = getattr(cuda_mod, "is_available", None) if cuda_mod is not None else None
        if callable(is_available) and is_available():
            candidate_index = getattr(device_t, "index", None)
            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is not None:
                try:
                    local_rank_val = int(local_rank_env)
                except (TypeError, ValueError):
                    local_rank_val = None
                else:
                    try:
                        cuda_mod.set_device(local_rank_val)  # type: ignore[attr-defined]
                        candidate_index = local_rank_val
                    except Exception:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "Failed to set CUDA device from LOCAL_RANK=%s", local_rank_env, exc_info=True
                            )
            if candidate_index is None or int(candidate_index) < 0:
                try:
                    candidate_index = int(cuda_mod.current_device())  # type: ignore[attr-defined]
                except Exception:
                    candidate_index = None
            if candidate_index is not None and int(candidate_index) >= 0:
                candidate_index = int(candidate_index)
                device_t = torch.device("cuda", candidate_index)
                ddp_device_index = candidate_index

    stage_config_local = stage_config
    if stage_config_local is None and "stage_config" in unused:
        stage_config_local = unused.pop("stage_config")
    if stage_config_local is None:
        stage_config_local = get_stage_config()
    else:
        try:
            merged = get_stage_config()
            merged.update(dict(stage_config_local))
            stage_config_local = merged
        except Exception:
            stage_config_local = get_stage_config()
    # unify to encoder's device in case 'device' arg and model diverge
    enc_param = next(encoder.parameters(), None)
    if enc_param is not None:
        device_t = enc_param.device
        if isinstance(device_t, torch.device) and device_t.type == "cuda":
            index_attr = getattr(device_t, "index", None)
            if index_attr is not None and int(index_attr) >= 0:
                ddp_device_index = int(index_attr)

    def _apply_encoder_trainability(module: nn.Module) -> None:
        params_fn = getattr(module, "parameters", None)
        if not callable(params_fn):
            return
        params = list(params_fn())
        if freeze_encoder:
            for param in params:
                param.requires_grad = False
            return
        if unfreeze_top_layers is None or unfreeze_top_layers <= 0:
            for param in params:
                param.requires_grad = True
            return
        for param in params:
            param.requires_grad = False
        modules = list(module.children())
        if not modules:
            for param in params:
                param.requires_grad = True
            return
        top_count = max(1, int(unfreeze_top_layers))
        selected = modules[-top_count:]
        for selected_module in selected:
            sub_params_fn = getattr(selected_module, "parameters", None)
            if not callable(sub_params_fn):
                continue
            try:
                for param in sub_params_fn():
                    param.requires_grad = True
            except Exception:
                continue

    encoder = encoder.to(device_t)
    _apply_encoder_trainability(encoder)
    encoder_has_trainable = any(p.requires_grad for p in encoder.parameters())
    if encoder_has_trainable and cache_graph_embeddings:
        logger.info(
            "Disabling graph embedding cache because encoder parameters are trainable."
        )
        cache_graph_embeddings = False
    num_graphs = _dataset_size(dataset)

    indices = list(range(num_graphs))

    explicit_indices = any(
        value is not None for value in (train_indices, val_indices, test_indices)
    )
    if explicit_indices:
        train_idx = _as_index_list(train_indices or [])
        val_idx = _as_index_list(val_indices or [])
        test_idx = _as_index_list(test_indices or [])
    elif use_scaffold and getattr(dataset, "smiles", None) is not None:
        train_idx, val_idx, test_idx = scaffold_split_indices(dataset.smiles)
        train_idx = _as_index_list(train_idx)
        val_idx = _as_index_list(val_idx)
        test_idx = _as_index_list(test_idx)
    elif task_type == "classification":
        train_idx, val_idx, test_idx = stratified_split(
            indices, dataset.labels, train_frac=0.8, val_frac=0.1
        )
    else:
        random.shuffle(indices)
        train_end = int(0.8 * num_graphs)
        val_end = int(0.9 * num_graphs)
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

    try:
        requested_batch_size = int(batch_size)
    except Exception:
        requested_batch_size = 1
    if requested_batch_size <= 0:
        requested_batch_size = 1
    batch_size = requested_batch_size

    if enable_batch_autoscale:
        min_steps = max(1, int(batch_autoscale_min_steps))
        floor = max(1, int(batch_autoscale_floor))
        total_train = max(0, len(train_idx))
        scaled = batch_size
        if total_train > 0 and min_steps > 0:
            while scaled > floor and (total_train // scaled) < min_steps:
                next_scaled = max(floor, scaled // 2)
                if next_scaled == scaled:
                    break
                scaled = next_scaled
        batch_size = max(1, scaled)
        logger.info(
            "[batch_autoscale] requested=%d actual=%d (N_train=%d)",
            requested_batch_size,
            batch_size,
            total_train,
        )

    requested_workers = num_workers
    num_workers, persistent_workers, prefetch_factor = autotune_worker_pool(
        requested_workers=requested_workers,
        dataset_size=num_graphs,
        batch_size=batch_size,
        device_type=device_t.type,
        persistent_workers=bool(persistent_workers),
        prefetch_factor=prefetch_factor,
        logger=logger,
        stage="finetune",
    )
    if num_workers > 0:
        ensure_file_system_sharing_strategy()
        worker_count = int(num_workers)
        prefetch_budget = (
            int(prefetch_factor) if isinstance(prefetch_factor, (int, float)) else 0
        )
        if worker_count > 0:
            prefetch_budget = max(prefetch_budget, 2)
        min_fd_budget = max(
            4096, 1024 + 128 * max(worker_count, 1) * max(prefetch_budget, 1)
        )
        ensure_open_file_limit(min_fd_budget)

    head_module = head
    if head_module is None:
        in_dim = getattr(encoder, "hidden_dim", None) or getattr(encoder, "out_dim", None)
        if in_dim is None:
            # Fallback: infer embedding size from one sample
            with torch.no_grad():
                emb = _encode_graph(encoder, dataset.graphs[0])
                in_dim = int(emb.shape[-1])
        head_module = nn.Linear(in_dim, 1)
    head_module = head_module.to(device_t)

    head_has_trainable = any(p.requires_grad for p in head_module.parameters())

    def _unwrap_if_ddp(module: Optional[nn.Module]) -> Optional[nn.Module]:
        if module is None:
            return None
        return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module

    encoder = _unwrap_if_ddp(encoder) or encoder
    head_module = _unwrap_if_ddp(head_module) or head_module

    if distributed:
        ddp_device_ids = None
        ddp_output_device = None
        if device_t.type == "cuda":
            candidate = ddp_device_index
            if candidate is None or candidate < 0:
                cuda_mod = getattr(torch, "cuda", None)
                if cuda_mod is not None:
                    try:
                        candidate = int(cuda_mod.current_device())  # type: ignore[attr-defined]
                    except Exception:
                        candidate = None
            if candidate is not None and candidate >= 0:
                ddp_device_ids = [candidate]
                ddp_output_device = candidate
        if encoder_has_trainable and not isinstance(encoder, nn.parallel.DistributedDataParallel):
            encoder = nn.parallel.DistributedDataParallel(
                encoder,
                device_ids=ddp_device_ids,
                output_device=ddp_output_device,
            )
        else:
            logger.debug(
                "Skipping DDP wrapper for encoder because it has no trainable parameters."
            )
        if head_has_trainable and not isinstance(
            head_module, nn.parallel.DistributedDataParallel
        ):
            head_module = nn.parallel.DistributedDataParallel(
                head_module,
                device_ids=ddp_device_ids,
                output_device=ddp_output_device,
            )
        else:
            logger.debug(
                "Skipping DDP wrapper for head because it has no trainable parameters."
            )
    head_param_source = (
        head_module.module
        if isinstance(head_module, nn.parallel.DistributedDataParallel)
        else head_module
    )

    encoder_initial_mode = encoder.training
    head_initial_mode = head_module.training
    if task_type == "classification":
        # BCE-specific imbalance controls (pos_weight/class_weight, focal loss, sampling)
        # are confined to this branch so regression heads remain unaffected.
        pos_weight_tensor: Optional[torch.Tensor] = None

        def _coerce_pos_weight(value: Any) -> Optional[torch.Tensor]:
            if value is None:
                return None
            try:
                tensor = torch.as_tensor(value, dtype=torch.float32)
            except Exception:
                logger.warning("Failed to coerce pos_weight to tensor; ignoring", exc_info=True)
                return None
            return tensor

        pos_weight_tensor = _coerce_pos_weight(pos_weight)
        if pos_weight_tensor is None and class_weight is not None:
            neg_weight: Optional[float] = None
            pos_weight_value: Optional[float] = None
            if isinstance(class_weight, dict):
                neg_weight = class_weight.get(0)
                pos_weight_value = class_weight.get(1)
            else:
                try:
                    weight_tensor = torch.as_tensor(class_weight, dtype=torch.float32)
                    flat = weight_tensor.flatten()
                    if flat.numel() >= 2:
                        neg_weight = float(flat[0].item())
                        pos_weight_value = float(flat[1].item())
                except Exception:
                    logger.warning(
                        "Failed to coerce class_weight to tensor when deriving pos_weight; ignoring",
                        exc_info=True,
                    )
            if neg_weight not in (None, 0) and pos_weight_value is not None:
                try:
                    ratio = float(pos_weight_value) / float(neg_weight)
                    pos_weight_tensor = torch.tensor([ratio], dtype=torch.float32)
                except Exception:
                    logger.warning(
                        "Failed to derive pos_weight from class_weight; ignoring", exc_info=True
                    )

        if pos_weight_tensor is not None:
            pos_weight_tensor = pos_weight_tensor.to(device_t).reshape(-1)

        def _make_loss_fn(weight: Optional[torch.Tensor]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            kwargs = {"pos_weight": weight} if weight is not None else {}
            if not use_focal_loss:
                return nn.BCEWithLogitsLoss(**kwargs)

            focal_base = nn.BCEWithLogitsLoss(reduction="none", **kwargs)
            gamma = float(focal_gamma) if focal_gamma is not None else 2.0

            def _focal(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                bce = focal_base(preds, targets)
                probs = torch.sigmoid(preds)
                pt = torch.where(targets > 0, probs, 1 - probs)
                modulating = torch.pow(1 - pt, gamma)
                loss = modulating * bce
                return loss.mean()

            return _focal

        loss_fn = _make_loss_fn(pos_weight_tensor)
    else:
        loss_fn = nn.MSELoss()
    optimiser: Optimizer
    if optimizer is not None:
        optimiser = optimizer
    else:
        enc_for_opt = (
            encoder.module
            if isinstance(encoder, nn.parallel.DistributedDataParallel)
            else encoder
        )
        enc_params = [p for p in enc_for_opt.parameters() if p.requires_grad]
        head_params = [p for p in head_param_source.parameters() if p.requires_grad]
        param_groups = []
        enc_lr_resolved = encoder_lr if encoder_lr is not None else lr
        if enc_params:
            if layerwise_decay is not None and layerwise_decay > 0:
                param_groups.extend(
                    _build_layerwise_param_groups(enc_for_opt, float(enc_lr_resolved), float(layerwise_decay))
                )
            else:
                param_groups.append({"params": enc_params, "lr": enc_lr_resolved})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr or lr})

        if param_groups:
            base_lr = param_groups[0].get("lr", head_lr or encoder_lr or lr)
            optimiser = torch.optim.Adam(param_groups, lr=base_lr)
        elif head_params:
            optimiser = torch.optim.Adam(head_params, lr=head_lr or lr)
        else:
            raise ValueError("No trainable parameters found for optimiser")
    rank = get_rank() if distributed else 0
    world = get_world_size() if distributed else 1
    train_idx_rank = train_idx[rank::world]

    train_sampler_weights: Optional[List[float]] = None
    if task_type == "classification" and oversample_minority and len(train_idx_rank) > 0:
        labels_attr = getattr(dataset, "labels", None)
        if labels_attr is not None:
            labels_arr = np.asarray(labels_attr)
            try:
                subset = labels_arr[train_idx_rank]
            except Exception:
                subset = labels_arr
            computed_weights = _compute_pos_weight_from_labels(np.asarray(subset))
            if computed_weights is not None:
                scale = float(np.nanmean(computed_weights))
                if math.isfinite(scale) and scale > 0:
                    weights = np.ones(len(train_idx_rank), dtype=np.float32)
                    pos_mask = subset > 0 if subset.ndim == 1 else (subset > 0).any(axis=1)
                    weights[np.asarray(pos_mask, dtype=bool)] = scale
                    weights = np.clip(weights, 1e-6, None)
                    train_sampler_weights = weights.tolist()

    if task_type == "classification" and dynamic_pos_weight and pos_weight_tensor is None:
        labels_attr = getattr(dataset, "labels", None)
        if labels_attr is not None and len(train_idx_rank) > 0:
            dynamic_weight_np = _compute_pos_weight_from_labels(np.asarray(labels_attr)[train_idx_rank])
            if dynamic_weight_np is not None:
                pos_weight_tensor = torch.as_tensor(dynamic_weight_np, dtype=torch.float32, device=device_t).reshape(-1)
                loss_fn = _make_loss_fn(pos_weight_tensor)

    collate_fn = _GraphBatchCollator(dataset, task_type)
    pin_memory_enabled = bool(pin_memory and device_t.type == "cuda")
    metric_aliases = {
        "val_loss": "val_loss",
        "loss": "val_loss",
        "val_auc": "roc_auc",
        "auc": "roc_auc",
        "auroc": "roc_auc",
        "roc_auc": "roc_auc",
        "val_pr_auc": "pr_auc",
        "pr_auc": "pr_auc",
        "val_rmse": "rmse",
        "rmse": "rmse",
        "val_mae": "mae",
        "mae": "mae",
        "val_r2": "r2",
        "r2": "r2",
    }
    metric_key = metric_aliases.get(str(early_stop_metric).lower(), "val_loss")
    monitor_mode = early_stop_mode or ("max" if metric_key in {"roc_auc", "pr_auc", "r2"} else "min")
    early_stopper = (
        EarlyStopping(patience=patience, mode=monitor_mode) if patience > 0 else None
    )
    best_val_snapshot: Dict[str, float] = {}

    cache_state = {"enabled": bool(cache_graph_embeddings)}

    def _estimate_fd_handles(workers: int, prefetch: int, loader_count: int) -> int:
        base = 256 + 48 * loader_count
        if cache_state["enabled"]:
            base += 96 * loader_count
        if loader_count <= 0 or workers <= 0:
            return base
        slots = loader_count * max(workers, 1)
        prefetch = max(prefetch, 1)
        return base + slots * (32 + 48 * prefetch)

    active_loader_count = sum(
        1
        for split_size in (len(train_idx_rank), len(val_idx), len(test_idx))
        if split_size > 0
    )
    current_workers = max(0, int(num_workers))
    if current_workers > 0:
        effective_prefetch = (
            max(1, int(prefetch_factor))
            if isinstance(prefetch_factor, (int, float))
            else 2
        )
    else:
        effective_prefetch = 0

    if active_loader_count > 0:
        required_handles = _estimate_fd_handles(
            current_workers, effective_prefetch, active_loader_count
        )
        fd_budget = check_fd_budget(required_handles)
        available = fd_budget.available

        if not fd_budget.ok and available is not None:
            best_workers = current_workers
            best_prefetch = effective_prefetch

            def _handles_for(workers: int, prefetch: int) -> int:
                return _estimate_fd_handles(workers, prefetch, active_loader_count)

            found = False
            slack = 64
            for workers_candidate in range(current_workers, -1, -1):
                if workers_candidate <= 0:
                    prefetch_candidates = [0]
                else:
                    prefetch_candidates = list(range(effective_prefetch, 0, -1))
                for prefetch_candidate in prefetch_candidates:
                    handles_needed = _handles_for(workers_candidate, prefetch_candidate)
                    if handles_needed <= available + slack:
                        best_workers = workers_candidate
                        best_prefetch = prefetch_candidate
                        found = True
                        break
                if found:
                    break

            original_prefetch = prefetch_factor
            if best_workers <= 0:
                new_prefetch: Optional[int] = None
            else:
                new_prefetch = max(1, best_prefetch)

            def _prefetch_changed() -> bool:
                if best_workers <= 0:
                    return original_prefetch is not None
                if isinstance(original_prefetch, (int, float)):
                    return int(original_prefetch) != new_prefetch
                return new_prefetch != effective_prefetch

            if best_workers != current_workers or _prefetch_changed():
                logger.warning(
                    "[finetune] Limited file-descriptor budget (required=%d, available=%s, soft_limit=%s, open_files=%s); "
                    "reducing num_workers from %d to %d and prefetch_factor from %s to %s.",
                    required_handles,
                    available,
                    fd_budget.soft_limit,
                    fd_budget.open_files,
                    current_workers,
                    best_workers,
                    "None" if original_prefetch is None else int(original_prefetch),
                    "None" if best_workers <= 0 else new_prefetch,
                )

                num_workers = best_workers
                if num_workers <= 0:
                    persistent_workers = False
                    prefetch_factor = None
                else:
                    prefetch_factor = new_prefetch

    def _effective_worker_count(split_size: int) -> int:
        if num_workers <= 0:
            return 0
        if split_size <= 0:
            return 0
        approx_batches = max(1, math.ceil(split_size / max(1, batch_size)))
        effective = min(num_workers, approx_batches)
        if effective < num_workers:
            logger.debug(
                "Reducing DataLoader workers from %d to %d for split of %d samples",
                num_workers,
                effective,
                split_size,
            )
        return effective

    def _build_loader(indices: List[int], shuffle: bool, split_name: str) -> Optional[DataLoader]:
        if not indices:
            return None
        worker_count = _effective_worker_count(len(indices))
        loader_prefetch = prefetch_factor
        spawn_ctx = None
        if worker_count > 0:
            spawn_ctx = _resolve_cuda_spawn_context(device_t.type)
            if device_t.type == "cuda" and spawn_ctx is None:
                logger.warning(
                    "CUDA detected but spawn multiprocessing context unavailable; "
                    "falling back to single-process DataLoader workers."
                )
                worker_count = 0
        if worker_count <= 0:
            loader_prefetch = None
        elif loader_prefetch is not None:
            batches_for_split = max(1, math.ceil(len(indices) / max(1, batch_size)))
            loader_prefetch = max(1, min(loader_prefetch, max(2, batches_for_split)))
        sampler = None
        if (
            split_name == "train"
            and oversample_minority
            and train_sampler_weights is not None
            and len(train_sampler_weights) == len(indices)
        ):
            sampler = WeightedRandomSampler(
                weights=train_sampler_weights,
                num_samples=len(train_sampler_weights),
                replacement=True,
            )
            shuffle = False

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": worker_count,
            "pin_memory": pin_memory_enabled,
            "collate_fn": collate_fn,
            "drop_last": False,
        }
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        if worker_count > 0:
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
            if loader_prefetch is not None:
                loader_kwargs["prefetch_factor"] = loader_prefetch
            if spawn_ctx is not None:
                loader_kwargs["multiprocessing_context"] = spawn_ctx
        return DataLoader(list(indices), **loader_kwargs)

    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None

    def _remove_loader_iterator(loader: Optional[DataLoader]) -> None:
        if loader is None:
            return
        if hasattr(loader, "_iterator"):
            try:
                delattr(loader, "_iterator")
            except Exception:
                logger.debug("Unable to delete DataLoader iterator state", exc_info=True)

    def _reset_mp_loader_iters(loaders: Iterable[Optional[DataLoader]]) -> None:
        dataloader_mod = getattr(torch.utils.data, "dataloader", None)
        mp_iter_cls = (
            getattr(dataloader_mod, "_MultiProcessingDataLoaderIter", None)
            if dataloader_mod is not None
            else None
        )
        reset_fn = getattr(mp_iter_cls, "_reset", None) if mp_iter_cls is not None else None
        if not callable(reset_fn):
            return
        for loader in loaders:
            if loader is None:
                continue
            try:
                reset_fn(loader)
            except TypeError:
                try:
                    reset_fn(loader, False)
                except Exception:
                    logger.debug("Failed to reset DataLoader workers", exc_info=True)
            except Exception:
                logger.debug("Failed to reset DataLoader workers", exc_info=True)

    def _shutdown_loader(loader: Optional[DataLoader]) -> None:
        """Close worker pools associated with a DataLoader, if any."""
        if loader is None:
            return

        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown_workers):
                try:
                    shutdown_workers()
                except Exception:
                    logger.debug("Failed to shutdown DataLoader workers cleanly", exc_info=True)

        dataset = getattr(loader, "dataset", None)
        close_fn = None
        if isinstance(dataset, SupportsTeardown):
            close_fn = getattr(dataset, "close", None)
        elif dataset is not None:
            close_fn = getattr(dataset, "shutdown", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                logger.warning("Failed to close dataset during DataLoader shutdown", exc_info=True)

    def _rebuild_loaders(
        existing: Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        for loader in existing:
            _shutdown_loader(loader)
        for loader in existing:
            _remove_loader_iterator(loader)
        _reset_mp_loader_iters(existing)
        return (
            _build_loader(train_idx_rank, shuffle=True, split_name="train"),
            _build_loader(val_idx, shuffle=False, split_name="val"),
            _build_loader(test_idx, shuffle=False, split_name="test"),
        )

    def _refresh_loaders() -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        return _rebuild_loaders((train_loader, val_loader, test_loader))

    train_loader, val_loader, test_loader = _refresh_loaders()
    planned_train_batches = 0
    train_loader_workers = 0
    if train_loader is not None:
        try:
            planned_train_batches = len(train_loader)  # type: ignore[arg-type]
        except TypeError:
            planned_train_batches = 0
        train_loader_workers = getattr(train_loader, "num_workers", 0)
        logger.info(
            "[finetune] train loader built with %d batches (split size=%d, batch_size=%d, workers=%d)",
            planned_train_batches,
            len(train_idx_rank),
            batch_size,
            train_loader_workers,
        )

    def _get_loader(name: str) -> Optional[DataLoader]:
        if name == "train":
            return train_loader
        if name == "val":
            return val_loader
        if name == "test":
            return test_loader
        raise ValueError(f"Unknown loader name: {name}")

    _start_wall = _time.perf_counter()

    def _coerce_positive_seconds(raw: Any) -> Optional[float]:
        if raw is None:
            return None
        try:
            if isinstance(raw, (int, float)):
                value = float(raw)
            else:
                text = str(raw).strip()
                if not text:
                    return None
                multiplier = 1.0
                lowered = text.lower()
                if lowered.endswith("m") and lowered[:-1].strip().replace(".", "", 1).lstrip("-+").isdigit():
                    multiplier = 60.0
                    text = lowered[:-1]
                elif lowered.endswith("s"):
                    text = lowered[:-1]
                value = float(text)
                value *= multiplier
        except Exception:
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value

    def _cfg_seconds(*names: str) -> Optional[float]:
        for name in names:
            if name not in stage_config_local:
                continue
            raw_value = stage_config_local.get(name)
            value = _coerce_positive_seconds(raw_value)
            if value is None:
                continue
            lowered = name.lower()
            if lowered.endswith("mins") or lowered.endswith("minutes") or lowered.endswith("_min"):
                value *= 60.0
            return value
        return None

    _explicit_budget_secs = float(time_budget_mins) * 60.0 if time_budget_mins and time_budget_mins > 0 else None
    _stage_budget_override = _cfg_seconds("time_budget_secs", "budget_secs", "stage_budget_secs")
    _orchestrator_timeout = _cfg_seconds("timeout_secs", "soft_timeout_secs", "wall_clock_secs", "orchestrator_timeout_secs")

    _effective_budget_secs = _explicit_budget_secs
    for candidate in (_stage_budget_override, _orchestrator_timeout):
        if candidate is None:
            continue
        if _effective_budget_secs is None or candidate < _effective_budget_secs:
            _effective_budget_secs = candidate

    _heartbeat_secs = _cfg_seconds("heartbeat_secs", "heartbeat_interval_secs", "orchestrator_heartbeat_secs")
    _grace_secs = _cfg_seconds("grace_secs", "grace_period_secs", "kill_after_secs", "grace_seconds")

    _log_interval_secs = _cfg_seconds("headroom_log_interval_secs")
    if _log_interval_secs is None:
        base_interval = _heartbeat_secs if _heartbeat_secs is not None else 0.0
        _log_interval_secs = max(60.0, base_interval if base_interval > 0 else 60.0)
    else:
        _log_interval_secs = max(1.0, _log_interval_secs)

    _safety_margin_secs = _cfg_seconds("safety_margin_secs", "budget_safety_margin_secs")
    if _safety_margin_secs is None:
        margin_candidates = [
            candidate
            for candidate in (
                (_heartbeat_secs * 2.0) if _heartbeat_secs is not None else None,
                _grace_secs,
                (_effective_budget_secs * 0.05) if _effective_budget_secs is not None else None,
                120.0,
            )
            if candidate is not None and candidate > 0
        ]
        _safety_margin_secs = max(margin_candidates) if margin_candidates else 120.0

    if _effective_budget_secs is not None and _safety_margin_secs >= _effective_budget_secs:
        _safety_margin_secs = max(min(_effective_budget_secs * 0.5, _effective_budget_secs - 1.0), 0.0)

    _budget_remaining_secs: Optional[float] = None
    _headroom_triggered = False
    _headroom_log_bucket: Optional[int] = None

    if _effective_budget_secs is not None:
        def _fmt(value: Optional[float]) -> str:
            return f"{value:.1f}s" if value is not None else "unset"

        logger.info(
            "[finetune] wall-clock budget configured: effective=%s (time_budget=%s, stage_override=%s, orchestrator=%s)"
            " safety_margin=%s heartbeat=%s log_interval=%s",
            _fmt(_effective_budget_secs),
            _fmt(_explicit_budget_secs),
            _fmt(_stage_budget_override),
            _fmt(_orchestrator_timeout),
            _fmt(_safety_margin_secs),
            _fmt(_heartbeat_secs),
            _fmt(_log_interval_secs),
        )
        if _log_interval_secs > 0:
            _headroom_log_bucket = int(math.ceil(_effective_budget_secs / _log_interval_secs))

    def _log_headroom(remaining: float) -> None:
        nonlocal _headroom_log_bucket
        if _effective_budget_secs is None or _log_interval_secs <= 0:
            return
        remaining = max(0.0, remaining)
        bucket = int(max(0, math.floor(remaining / _log_interval_secs)))
        if _headroom_log_bucket is None or bucket < _headroom_log_bucket:
            logger.info(
                "[finetune] remaining wall-clock headroom %.1fs (effective %.1fs, safety margin %.1fs)",
                remaining,
                _effective_budget_secs,
                _safety_margin_secs,
            )
            _headroom_log_bucket = bucket

    def _time_left() -> bool:
        nonlocal _budget_remaining_secs, _headroom_triggered
        if _effective_budget_secs is None:
            return True
        elapsed = _time.perf_counter() - _start_wall
        remaining = _effective_budget_secs - elapsed
        _budget_remaining_secs = remaining
        if remaining <= 0:
            logger.info(
                "[finetune] time budget exhausted (elapsed=%.1fs, budget=%.1fs); stopping linear-head training.",
                elapsed,
                _effective_budget_secs,
            )
            _headroom_triggered = True
            return False
        _log_headroom(remaining)
        if remaining <= _safety_margin_secs:
            logger.warning(
                "[finetune] remaining headroom %.1fs below safety margin %.1fs; stopping early to flush outputs.",
                remaining,
                _safety_margin_secs,
            )
            _headroom_triggered = True
            return False
        return True

    import contextlib
    use_amp = bf16 and device_t.type == "cuda"

    def _amp_context():
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else contextlib.nullcontext()
        )

    embedding_cache: Dict[int, torch.Tensor] = {}

    def _rebuild_loaders_after_failure(
        *,
        disable_pin_memory: bool = False,
        drop_workers: bool = False,
        disable_cache: bool = True,
    ) -> bool:
        """Tear down existing loaders and rebuild them with safer settings."""

        nonlocal pin_memory_enabled, train_loader, val_loader, test_loader
        nonlocal num_workers, persistent_workers, prefetch_factor

        previous_loaders = (train_loader, val_loader, test_loader)

        if disable_pin_memory:
            pin_memory_enabled = False

        if disable_cache:
            cache_state["enabled"] = False
            embedding_cache.clear()

        if drop_workers:
            num_workers = 0
            persistent_workers = False
            prefetch_factor = None

        train_loader, val_loader, test_loader = _rebuild_loaders(previous_loaders)
        return True

    def _handle_pin_memory_failure(err: BaseException) -> bool:
        if not pin_memory_enabled and num_workers == 0:
            return False

        logger.warning(
            "Pin-memory DataLoader pipeline failed (%s). "
            "Rebuilding loaders without pinned memory, disabling the cache, and reducing workers.",
            err,
        )

        return _rebuild_loaders_after_failure(
            disable_pin_memory=True,
            drop_workers=True,
        )

    def _handle_fd_exhaustion_failure(err: BaseException) -> bool:
        logger.warning(
            "DataLoader hit file-descriptor limits (%s). "
            "Rebuilding loaders with a single-process, non-pinned configuration.",
            err,
        )

        return _rebuild_loaders_after_failure(
            disable_pin_memory=True,
            drop_workers=True,
        )

    def _should_retry_loader_error(err: BaseException) -> bool:
        msg = str(err).lower()
        pin_triggers = (
            "pin memory",
            "pin_memory",
            "pin memory thread",
            "pin_memory thread",
        )
        if any(trigger in msg for trigger in pin_triggers):
            return _handle_pin_memory_failure(err)

        fd_triggers = ("too many open files", "errno 24", "ancdata")
        if any(trigger in msg for trigger in fd_triggers):
            return _handle_fd_exhaustion_failure(err)
        return False

    def _grad_context():
        return contextlib.nullcontext() if encoder_has_trainable else torch.no_grad()

    def _get_graph_embeddings(
        batch_x: torch.Tensor,
        batch_adj: torch.Tensor,
        batch_ptr: Optional[torch.Tensor],
        batch_meta: object,
    ) -> torch.Tensor:
        idx_list = _extract_batch_indices(batch_meta)
        use_cache = cache_state["enabled"] and idx_list is not None
        if use_cache and all(idx in embedding_cache for idx in idx_list):
            stacked = torch.stack([embedding_cache[idx] for idx in idx_list], dim=0)
            return stacked.to(device_t)

        base_encoder = encoder.module if isinstance(
            encoder, nn.parallel.DistributedDataParallel
        ) else encoder

        if idx_list is not None and hasattr(base_encoder, "encode_graph"):
            graphs = [dataset.graphs[idx] for idx in idx_list]
            with _grad_context():
                with _amp_context():
                    emb_list = []
                    for graph in graphs:
                        emb = base_encoder.encode_graph(graph, device_t)
                        if not torch.is_tensor(emb):
                            emb = torch.as_tensor(emb, device=device_t)
                        else:
                            emb = emb.to(device_t)
                        if emb.dim() == 2 and emb.size(0) == 1:
                            emb = emb.squeeze(0)
                        emb_list.append(emb)
                    graph_emb = torch.stack(emb_list, dim=0)

            if use_cache:
                for graph_idx, emb in zip(idx_list, graph_emb):
                    embedding_cache[graph_idx] = emb.detach().cpu()

            return graph_emb

        graph_obj = _build_graph_view(batch_x, batch_adj, batch_ptr, batch_meta)
        with _grad_context():
            with _amp_context():
                node_embeddings = _encode_graph(encoder, graph_obj)

        graph_emb = _pool_batch_embeddings(node_embeddings, batch_ptr)

        if use_cache:
            for graph_idx, emb in zip(idx_list, graph_emb):
                embedding_cache[graph_idx] = emb.detach().cpu()

        return graph_emb

    def _precompute_embeddings(loader_name: str) -> None:
        if not cache_state["enabled"]:
            return
        while True:
            loader = _get_loader(loader_name)
            if loader is None:
                return
            try:
                for batch in loader:
                    batch_x, batch_adj, batch_ptr, _, batch_meta = _move_batch_to_device(
                        batch, device_t, pin_memory_enabled
                    )
                    _get_graph_embeddings(batch_x, batch_adj, batch_ptr, batch_meta)
                return
            except (RuntimeError, OSError) as err:
                if not _should_retry_loader_error(err):
                    raise

    if cache_state["enabled"]:
        encoder.eval()
        try:
            _precompute_embeddings("train")
            _precompute_embeddings("val")
            _precompute_embeddings("test")
        except (RuntimeError, OSError) as err:
            if not _should_retry_loader_error(err):
                raise

    def _yield_batches(name: str):
        while True:
            loader = _get_loader(name)
            if loader is None:
                return
            try:
                for batch in loader:
                    yield batch
                return
            except (RuntimeError, OSError) as err:
                if not _should_retry_loader_error(err):
                    raise

    total_batches_done = 0
    last_epoch_batches = 0
    if train_loader is None:
        logger.warning(
            "No training samples available; skipping linear-head optimisation (split size=%d).",
            len(train_idx_rank),
        )
    else:
        for epoch in range(epochs):
            if max_batches > 0 and total_batches_done >= max_batches:
                logger.info(
                    "Max linear-head batches reached (%d); stopping training.",
                    max_batches,
                )
                break
            if not _time_left():
                logger.info("Time budget hit before epoch %d; stopping training.", epoch)
                break

            if task_type == "classification" and dynamic_pos_weight:
                labels_attr = getattr(dataset, "labels", None)
                if labels_attr is not None and len(train_idx_rank) > 0:
                    updated = _compute_pos_weight_from_labels(np.asarray(labels_attr)[train_idx_rank])
                    if updated is not None:
                        pos_weight_tensor = torch.as_tensor(updated, dtype=torch.float32, device=device_t).reshape(-1)
                        loss_fn = _make_loss_fn(pos_weight_tensor)

            encoder.eval()
            head_module.train()
            batch_losses = []
            epoch_batches = 0
            hit_batch_cap = False

            for batch in _yield_batches("train"):
                if max_batches > 0 and total_batches_done >= max_batches:
                    hit_batch_cap = True
                    break
                if not _time_left():
                    logger.info(
                        "Time budget hit during linear-head train epoch=%d; breaking.",
                        epoch,
                    )
                    break

                batch_x, batch_adj, batch_ptr, batch_labels, batch_meta = _move_batch_to_device(
                    batch, device_t, pin_memory_enabled
                )
                if batch_labels is None:
                    raise ValueError("Dataset must have labels for supervised training.")

                graph_emb = _get_graph_embeddings(batch_x, batch_adj, batch_ptr, batch_meta)
                param = next(head_param_source.parameters(), None)
                if param is not None and graph_emb.dtype != param.dtype:
                    graph_emb = graph_emb.to(param.dtype)

                with _amp_context():
                    raw_preds = head_module(graph_emb)

                if task_type == "classification":
                    preds = _ensure_binary_classification_logits(raw_preds)
                else:
                    preds = raw_preds.squeeze(-1)

                if not torch.isfinite(preds).all():
                    preds = torch.nan_to_num(preds)

                targets = batch_labels
                try:
                    targets = targets.reshape(preds.shape)
                except RuntimeError as exc:  # pragma: no cover - sanity guard
                    raise ValueError(
                        f"Target size {tuple(batch_labels.shape)} must match predictions {tuple(preds.shape)}"
                    ) from exc
                targets = targets.to(dtype=preds.dtype)

                with _amp_context():
                    loss = loss_fn(preds.float(), targets.float())
                batch_losses.append(loss.item())
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_batches += 1
                total_batches_done += 1
            last_epoch_batches = epoch_batches

            if batch_losses:
                logger.debug("Epoch %d training loss %.4f", epoch, float(np.mean(batch_losses)))
            else:
                logger.debug("Epoch %d produced no training batches", epoch)

            if early_stopper is not None and val_loader is not None:
                encoder.eval()
                head_module.eval()
                val_losses = []
                val_preds_store: List[np.ndarray] = []
                val_targets_store: List[np.ndarray] = []

                for batch in _yield_batches("val"):
                    batch_x, batch_adj, batch_ptr, batch_labels, batch_meta = _move_batch_to_device(
                        batch, device_t, pin_memory_enabled
                    )
                    if batch_labels is None:
                        raise ValueError("Validation loader returned samples without labels.")

                    graph_emb = _get_graph_embeddings(batch_x, batch_adj, batch_ptr, batch_meta)
                    param = next(head_param_source.parameters(), None)
                    if param is not None and graph_emb.dtype != param.dtype:
                        graph_emb = graph_emb.to(param.dtype)

                    with _amp_context():
                        raw_preds = head_module(graph_emb)

                    if task_type == "classification":
                        preds = _ensure_binary_classification_logits(raw_preds)
                    else:
                        preds = raw_preds.squeeze(-1)

                    preds = torch.nan_to_num(preds)

                    targets = batch_labels
                    try:
                        targets = targets.reshape(preds.shape)
                    except RuntimeError as exc:  # pragma: no cover - sanity guard
                        raise ValueError(
                            f"Target size {tuple(batch_labels.shape)} must match predictions {tuple(preds.shape)}"
                        ) from exc

                    val_preds_store.append(
                        preds.detach().to(torch.float32).cpu().numpy()
                    )
                    val_targets_store.append(
                        targets.detach().to(torch.float32).cpu().numpy()
                    )
                    targets = targets.to(dtype=preds.dtype)
                    vloss = loss_fn(preds, targets).item()
                    val_losses.append(vloss)

                avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
                avg_t = torch.tensor([avg_val_loss], device=device_t)
                if distributed:
                    dist_mod = getattr(torch, "distributed", None)
                    if (
                        dist_mod is not None
                        and getattr(dist_mod, "is_available", lambda: False)()
                        and getattr(dist_mod, "is_initialized", lambda: False)()
                    ):
                        dist_mod.all_reduce(avg_t, op=dist_mod.ReduceOp.AVG)
                avg_val_loss = avg_t.item()
                val_metric_values: Dict[str, float] = {}
                if val_preds_store and val_targets_store:
                    y_true_val = np.concatenate(val_targets_store)
                    y_pred_val = np.concatenate(val_preds_store)
                    y_true_val = np.nan_to_num(y_true_val, nan=0.0, posinf=0.0, neginf=0.0)
                    y_pred_val = np.nan_to_num(y_pred_val, nan=0.0, posinf=0.0, neginf=0.0)
                    if task_type == "classification":
                        val_metric_values = compute_classification_metrics(
                            y_true_val, y_pred_val
                        )
                    else:
                        val_metric_values = compute_regression_metrics(y_true_val, y_pred_val)

                monitor_value = avg_val_loss
                current_mode = monitor_mode
                if metric_key != "val_loss":
                    metric_val = val_metric_values.get(metric_key)
                    if metric_val is not None and not np.isnan(metric_val):
                        monitor_value = float(metric_val)
                    else:
                        logger.debug(
                            "Validation metric '%s' unavailable; falling back to val_loss.",
                            early_stop_metric,
                        )
                        monitor_value = avg_val_loss
                        current_mode = "min"

                if early_stopper.mode != current_mode:
                    logger.debug(
                        "Switching early stopping mode from %s to %s due to metric fallback.",
                        early_stopper.mode,
                        current_mode,
                    )
                    early_stopper.mode = current_mode
                    early_stopper.best = None
                    early_stopper.counter = 0
                    best_val_snapshot = {}

                val_snapshot: Dict[str, float] = {"val_loss": float(avg_val_loss)}
                for name, value in val_metric_values.items():
                    if value is None:
                        continue
                    try:
                        cast_value = float(value)
                    except Exception:
                        continue
                    if math.isnan(cast_value):
                        continue
                    val_snapshot[f"val_{name}"] = cast_value

                prev_best = early_stopper.best
                should_stop = early_stopper.step(monitor_value)
                if early_stopper.best != prev_best:
                    best_val_snapshot = dict(val_snapshot)
                elif not best_val_snapshot:
                    best_val_snapshot = dict(val_snapshot)

                if should_stop:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            if scheduler is not None and epoch_batches > 0:
                scheduler.step()

            if hit_batch_cap and max_batches > 0 and total_batches_done >= max_batches:
                logger.info(
                    "Max linear-head batches reached (%d); stopping training.",
                    max_batches,
                )
                break

    base_encoder_for_ig = (
        encoder.module if isinstance(encoder, nn.parallel.DistributedDataParallel) else encoder
    )
    base_head_for_ig = (
        head_module.module
        if isinstance(head_module, nn.parallel.DistributedDataParallel)
        else head_module
    )
    ig_loggers: List[Union[_IGArtifactLogger, _MotifIGArtifactLogger]] = []
    explain_modes_normalised = _normalise_explain_modes(explain_mode)
    if explain_modes_normalised and (is_main_process() or not distributed):
        ig_loggers = _build_explain_loggers(
            modes=explain_modes_normalised,
            dataset=dataset,
            encoder=base_encoder_for_ig,
            head_module=base_head_for_ig,
            task_type=task_type,
            device=device_t,
            explain_config=explain_config,
            stage_config=stage_config_local,
        )

    metrics: Dict[str, Any] = {"head": head_param_source}
    tuned_threshold: Optional[float] = None
    tuned_score: Optional[float] = None
    calibration_payload: Dict[str, Any] = {}
    calibrator: Optional[Tuple[str, Any]] = None

    if is_main_process() or not distributed:
        encoder.eval()
        head_module.eval()

        def _collect_outputs(split: str, *, log_meta: bool = False) -> Tuple[np.ndarray, np.ndarray]:
            preds_list: List[np.ndarray] = []
            targets_list: List[np.ndarray] = []
            loader = _get_loader(split)
            if loader is None:
                return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            for batch in _yield_batches(split):
                batch_x, batch_adj, batch_ptr, batch_labels, batch_meta = _move_batch_to_device(
                    batch, device_t, pin_memory_enabled
                )
                if batch_labels is None:
                    raise ValueError(f"{split} loader returned samples without labels.")

                graph_emb = _get_graph_embeddings(batch_x, batch_adj, batch_ptr, batch_meta)
                param = next(head_param_source.parameters(), None)
                if param is not None and graph_emb.dtype != param.dtype:
                    graph_emb = graph_emb.to(param.dtype)

                with _amp_context():
                    raw_preds = head_module(graph_emb)

                if task_type == "classification":
                    preds = _ensure_binary_classification_logits(raw_preds)
                else:
                    preds = raw_preds.squeeze(-1)

                preds = torch.nan_to_num(preds)
                targets = batch_labels
                try:
                    targets = targets.reshape(preds.shape)
                except RuntimeError as exc:  # pragma: no cover - sanity guard
                    raise ValueError(
                        f"Target size {tuple(batch_labels.shape)} must match predictions {tuple(preds.shape)}"
                    ) from exc

                preds_list.append(preds.detach().to(torch.float32).cpu().numpy())
                targets_list.append(targets.detach().to(torch.float32).cpu().numpy())
                if log_meta and ig_loggers:
                    for ig_logger in ig_loggers:
                        ig_logger.process_batch(batch_meta)

            if preds_list and targets_list:
                return (
                    np.concatenate(preds_list),
                    np.concatenate(targets_list),
                )
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        val_logits, val_targets = _collect_outputs("val")
        test_logits, test_targets = _collect_outputs("test", log_meta=True)

        eval_val_logits = np.nan_to_num(val_logits, nan=0.0, posinf=0.0, neginf=0.0)
        eval_test_logits = np.nan_to_num(test_logits, nan=0.0, posinf=0.0, neginf=0.0)

        if task_type == "classification":
            if calibrate_probabilities and val_targets.size > 0:
                method_norm = (calibration_method or "temperature").lower()
                if method_norm == "isotonic":
                    calibrated_probs, iso_model = _isotonic_calibration(eval_val_logits, val_targets)
                    eval_val_logits = _prob_to_logit(calibrated_probs)
                    calibrator = ("isotonic", iso_model)
                else:
                    eval_val_logits, temp_value = _temperature_scale_logits(eval_val_logits, val_targets)
                    calibrator = ("temperature", temp_value)
                    calibration_payload["calibration/temperature"] = float(temp_value)
                calibration_payload["calibration/method"] = calibrator[0]

            def _apply_calibrator_to_logits(logits: np.ndarray) -> np.ndarray:
                if calibrator is None:
                    return logits
                if calibrator[0] == "temperature":
                    temp = float(calibrator[1]) if calibrator[1] is not None else 1.0
                    return logits / max(temp, 1e-3)
                calibrated = calibrator[1].transform(_sigmoid_np(logits.reshape(-1))).reshape(logits.shape)
                return _prob_to_logit(calibrated)

            eval_test_logits = _apply_calibrator_to_logits(eval_test_logits)
            if calibrator is not None:
                eval_val_logits = _apply_calibrator_to_logits(eval_val_logits)

            val_probs = _sigmoid_np(eval_val_logits)
            tuned_threshold, tuned_score = _tune_threshold(val_targets, val_probs, threshold_metric)
            if tuned_threshold is not None:
                eval_test_logits = _apply_threshold_offset(eval_test_logits, tuned_threshold)
                eval_val_logits = _apply_threshold_offset(eval_val_logits, tuned_threshold)
                calibration_payload["threshold/value"] = float(tuned_threshold)
            if tuned_score is not None:
                calibration_payload["threshold/score"] = float(tuned_score)

        # Final safety net
        y_true = np.nan_to_num(test_targets, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(eval_test_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if task_type == "classification":
            metrics.update(compute_classification_metrics(y_true, y_pred))
        else:
            metrics.update(compute_regression_metrics(y_true, y_pred))
        if best_val_snapshot:
            metrics.update(best_val_snapshot)
        elif val_loader is not None:
            metrics.setdefault("val_loss", float("nan"))
        metrics.update(calibration_payload)
        metrics["train/batches"] = float(total_batches_done)
        metrics["train/loader_batches"] = float(planned_train_batches)
        metrics["train/epoch_batches"] = float(last_epoch_batches)
        for ig_logger in ig_loggers:
            ig_logger.finalize(metrics)

    if _effective_budget_secs is not None:
        metrics.setdefault("time/budget_secs", float(_effective_budget_secs))
        if _budget_remaining_secs is not None:
            metrics.setdefault("time/headroom_secs", max(float(_budget_remaining_secs), 0.0))
        metrics.setdefault("time/budget_exhausted", float(1.0 if _headroom_triggered else 0.0))

    try:
        encoder.train(encoder_initial_mode)
    except Exception:  # pragma: no cover - best effort restoration
        logger.debug("Failed to restore encoder training mode", exc_info=True)

    try:
        head_module.train(head_initial_mode)
    except Exception:  # pragma: no cover - best effort restoration
        logger.debug("Failed to restore head training mode", exc_info=True)

    if distributed:
        dist_mod = getattr(torch, "distributed", None)
        try:
            if dist_mod is not None and dist_mod.is_available() and dist_mod.is_initialized():
                dist_mod.barrier()
        except Exception:  # pragma: no cover - best effort synchronisation
            logger.debug("Distributed barrier failed during linear-head shutdown", exc_info=True)
        cleanup()
    metrics["head"] = head_param_source
    return metrics


def train_linear_head(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    task_type: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda",
    patience: int = 10,
    num_workers: int = -1,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    bf16=False,
    use_scaffold: bool = False,
    devices: int = 1,
    *,
    max_batches: int = 0,
    time_budget_mins: int = 0,
    head: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    encoder_lr: Optional[float] = None,
    head_lr: Optional[float] = None,
    freeze_encoder: bool = True,
    early_stop_metric: str = "val_loss",
    early_stop_mode: Optional[str] = None,
    cache_graph_embeddings: bool = True,
    train_indices: Optional[Iterable[int]] = None,
    val_indices: Optional[Iterable[int]] = None,
    test_indices: Optional[Iterable[int]] = None,
    enable_batch_autoscale: bool = False,
    batch_autoscale_min_steps: int = 10,
    batch_autoscale_floor: int = 64,
    unfreeze_top_layers: int = 0,
    stage_config: Optional[Dict[str, Any]] = None,
    pos_weight: Optional[Any] = None,
    class_weight: Optional[Any] = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    dynamic_pos_weight: bool = False,
    oversample_minority: bool = False,
    layerwise_decay: Optional[float] = None,
    calibrate_probabilities: bool = False,
    calibration_method: str = "temperature",
    threshold_metric: str = "f1",
    explain_mode: Optional[Union[str, Iterable[str]]] = None,
    explain_config: Optional[Dict[str, Any]] = None,
    **unused,
) -> Dict[str, Any]:
    """Train a linear head and retry with gloo when NCCL detects duplicates."""

    backend_override = os.environ.get("DDP_FORCE_BACKEND", "").strip().lower()

    try:
        return _train_linear_head_impl(
            dataset=dataset,
            encoder=encoder,
            task_type=task_type,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            patience=patience,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            bf16=bf16,
            use_scaffold=use_scaffold,
            devices=devices,
            max_batches=max_batches,
            time_budget_mins=time_budget_mins,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            encoder_lr=encoder_lr,
            head_lr=head_lr,
            freeze_encoder=freeze_encoder,
            early_stop_metric=early_stop_metric,
            early_stop_mode=early_stop_mode,
            cache_graph_embeddings=cache_graph_embeddings,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            enable_batch_autoscale=enable_batch_autoscale,
            batch_autoscale_min_steps=batch_autoscale_min_steps,
            batch_autoscale_floor=batch_autoscale_floor,
            unfreeze_top_layers=unfreeze_top_layers,
            stage_config=stage_config,
            pos_weight=pos_weight,
            class_weight=class_weight,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            dynamic_pos_weight=dynamic_pos_weight,
            oversample_minority=oversample_minority,
            layerwise_decay=layerwise_decay,
            calibrate_probabilities=calibrate_probabilities,
            calibration_method=calibration_method,
            threshold_metric=threshold_metric,
            explain_mode=explain_mode,
            explain_config=explain_config,
            **unused,
        )
    except Exception as exc:
        if backend_override == "gloo" or not should_retry_with_gloo(exc):
            raise

        logger.warning(
            "Distributed linear head training failed with NCCL backend (%s); retrying with gloo.",
            exc,
        )

        cleanup_fn = None
        try:
            from utils.ddp import cleanup as cleanup_fn  # type: ignore[assignment]
        except Exception:
            cleanup_fn = None  # type: ignore[assignment]

        if callable(cleanup_fn):
            try:
                cleanup_fn()
            except Exception:
                logger.debug("DDP cleanup prior to gloo retry failed", exc_info=True)

        previous_backend = os.environ.get("DDP_FORCE_BACKEND")
        try:
            os.environ["DDP_FORCE_BACKEND"] = "gloo"
            return _train_linear_head_impl(
                dataset=dataset,
                encoder=encoder,
                task_type=task_type,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
                patience=patience,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                bf16=bf16,
                use_scaffold=use_scaffold,
                devices=devices,
                max_batches=max_batches,
                time_budget_mins=time_budget_mins,
                head=head,
                optimizer=optimizer,
                scheduler=scheduler,
                encoder_lr=encoder_lr,
                head_lr=head_lr,
                freeze_encoder=freeze_encoder,
                early_stop_metric=early_stop_metric,
                early_stop_mode=early_stop_mode,
                cache_graph_embeddings=cache_graph_embeddings,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                enable_batch_autoscale=enable_batch_autoscale,
                batch_autoscale_min_steps=batch_autoscale_min_steps,
                batch_autoscale_floor=batch_autoscale_floor,
                unfreeze_top_layers=unfreeze_top_layers,
                stage_config=stage_config,
                pos_weight=pos_weight,
                class_weight=class_weight,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                dynamic_pos_weight=dynamic_pos_weight,
                oversample_minority=oversample_minority,
                layerwise_decay=layerwise_decay,
                calibrate_probabilities=calibrate_probabilities,
                calibration_method=calibration_method,
                threshold_metric=threshold_metric,
                **unused,
            )
        finally:
            if previous_backend is None:
                os.environ.pop("DDP_FORCE_BACKEND", None)
            else:
                os.environ["DDP_FORCE_BACKEND"] = previous_backend


train_linear_head.__doc__ = _train_linear_head_impl.__doc__
