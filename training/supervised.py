"""Supervised training routines for downstream tasks.

This module defines a function to train a simple linear head on top of a
frozen encoder for classification or regression tasks. The data is
split into train/validation/test sets using a stratified approach for
classification to ensure that each class appears in all splits when
possible. Performance metrics are computed using utilities from
`utils.metrics`.
"""

from __future__ import annotations

import logging
import math
import random
import time as _time
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from data.mdataset import GraphDataset
from data.scaffold_split import scaffold_split_indices
from models.encoder import GNNEncoder
from utils.early_stopping import EarlyStopping
from utils.metrics import compute_classification_metrics, compute_regression_metrics
from utils.graph_ops import _encode_graph
from utils.dataloader import (
    autotune_worker_pool,
    ensure_file_system_sharing_strategy,
    ensure_open_file_limit,
)

logger = logging.getLogger(__name__)

__all__ = ["stratified_split", "train_linear_head"]

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
    **unused,
) -> Dict[str, float]:
    """Train a linear head on a frozen encoder for classification or regression.

    When ``devices > 1`` the encoder and head are wrapped with
    :class:`~torch.nn.parallel.DistributedDataParallel` and gradients are
    synchronised across ranks. Validation loss for early stopping is
    averaged across all processes to ensure a consistent stopping epoch.

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

    distributed = devices > 1 and init_distributed()
    device_t = torch.device(device)
    # unify to encoder's device in case 'device' arg and model diverge
    enc_param = next(encoder.parameters(), None)
    if enc_param is not None:
        device_t = enc_param.device

    encoder = encoder.to(device_t)
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
    encoder_has_trainable = any(p.requires_grad for p in encoder.parameters())
    if encoder_has_trainable and cache_graph_embeddings:
        logger.info(
            "Disabling graph embedding cache because encoder parameters are trainable."
        )
        cache_graph_embeddings = False
    num_graphs = _dataset_size(dataset)

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

    indices = list(range(num_graphs))

    # Use provided batch_indices for single-batch training, else split dataset
    if use_scaffold and getattr(dataset, "smiles", None) is not None:
        train_idx, val_idx, test_idx = scaffold_split_indices(dataset.smiles)
        train_idx, val_idx, test_idx = (
            train_idx.tolist(),
            val_idx.tolist(),
            test_idx.tolist(),
        )
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

    if distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder, device_ids=[torch.cuda.current_device()] if device_t.type == "cuda" else None
        )
        head_module = nn.parallel.DistributedDataParallel(
            head_module, device_ids=[torch.cuda.current_device()] if device_t.type == "cuda" else None
        )
    head_param_source = (
        head_module.module
        if isinstance(head_module, nn.parallel.DistributedDataParallel)
        else head_module
    )
    loss_fn = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()
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
        if enc_params:
            param_groups.append({"params": enc_params, "lr": encoder_lr or lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr or lr})

        if param_groups:
            base_lr = head_lr or encoder_lr or lr
            optimiser = torch.optim.Adam(param_groups, lr=base_lr)
        elif head_params:
            optimiser = torch.optim.Adam(head_params, lr=head_lr or lr)
        else:
            raise ValueError("No trainable parameters found for optimiser")
    rank = get_rank() if distributed else 0
    world = get_world_size() if distributed else 1
    train_idx_rank = train_idx[rank::world]

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

    def _build_loader(indices: List[int], shuffle: bool) -> Optional[DataLoader]:
        if not indices:
            return None
        worker_count = _effective_worker_count(len(indices))
        loader_prefetch = prefetch_factor
        if worker_count <= 0:
            loader_prefetch = None
        elif loader_prefetch is not None:
            batches_for_split = max(1, math.ceil(len(indices) / max(1, batch_size)))
            loader_prefetch = max(1, min(loader_prefetch, max(2, batches_for_split)))
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": worker_count,
            "pin_memory": pin_memory_enabled,
            "collate_fn": collate_fn,
            "drop_last": False,
        }
        if worker_count > 0:
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
            if loader_prefetch is not None:
                loader_kwargs["prefetch_factor"] = loader_prefetch
        return DataLoader(list(indices), **loader_kwargs)

    def _refresh_loaders() -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        return (
            _build_loader(train_idx_rank, shuffle=True),
            _build_loader(val_idx, shuffle=False),
            _build_loader(test_idx, shuffle=False),
        )

    train_loader, val_loader, test_loader = _refresh_loaders()

    def _get_loader(name: str) -> Optional[DataLoader]:
        if name == "train":
            return train_loader
        if name == "val":
            return val_loader
        if name == "test":
            return test_loader
        raise ValueError(f"Unknown loader name: {name}")

    _start_wall = _time.perf_counter()

    def _time_left() -> bool:
        return (time_budget_mins <= 0) or ((_time.perf_counter() - _start_wall) < time_budget_mins * 60)

    import contextlib
    use_amp = bf16 and device_t.type == "cuda"

    def _amp_context():
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else contextlib.nullcontext()
        )

    embedding_cache: Dict[int, torch.Tensor] = {}

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

        try:
            setattr(loader, "_iterator", None)
        except Exception:
            logger.debug("Unable to reset DataLoader iterator state", exc_info=True)

    def _handle_pin_memory_failure(err: BaseException) -> bool:
        nonlocal pin_memory_enabled, train_loader, val_loader, test_loader, num_workers, persistent_workers

        if not pin_memory_enabled and num_workers == 0:
            return False

        logger.warning(
            "Pin-memory DataLoader pipeline failed (%s). "
            "Rebuilding loaders without pinned memory, disabling the cache, and reducing workers.",
            err,
        )

        _shutdown_loader(train_loader)
        _shutdown_loader(val_loader)
        _shutdown_loader(test_loader)

        train_loader = None
        val_loader = None
        test_loader = None

        pin_memory_enabled = False
        cache_state["enabled"] = False
        num_workers = 0
        persistent_workers = False
        embedding_cache.clear()
        train_loader, val_loader, test_loader = _refresh_loaders()
        return True

    def _should_retry_loader_error(err: BaseException) -> bool:
        msg = str(err).lower()
        triggers = (
            "pin memory",
            "pin_memory",
            "pin memory thread",
            "pin_memory thread",
            "too many open files",
            "errno 24",
            "ancdata",
        )
        if any(trigger in msg for trigger in triggers):
            return _handle_pin_memory_failure(err)
        return False

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
            with torch.no_grad():
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
        with torch.no_grad():
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

    if train_loader is None:
        logger.warning("No training samples available; skipping linear-head optimisation.")
    else:
        total_batches_done = 0
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
                    preds = head_module(graph_emb).squeeze(1)

                if not torch.isfinite(preds).all():
                    preds = torch.nan_to_num(preds)

                targets = batch_labels.to(dtype=preds.dtype)

                with _amp_context():
                    loss = loss_fn(preds.float(), targets.float())
                batch_losses.append(loss.item())
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_batches += 1
                total_batches_done += 1

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
                        preds = head_module(graph_emb).squeeze(1)
                    preds = torch.nan_to_num(preds)
                    val_preds_store.append(
                        preds.detach().to(torch.float32).cpu().numpy()
                    )
                    val_targets_store.append(
                        batch_labels.detach().to(torch.float32).cpu().numpy()
                    )
                    targets = batch_labels.to(dtype=preds.dtype)
                    vloss = loss_fn(preds, targets).item()
                    val_losses.append(vloss)

                avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
                avg_t = torch.tensor([avg_val_loss], device=device_t)
                if distributed:
                    torch.distributed.all_reduce(avg_t, op=torch.distributed.ReduceOp.AVG)
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

    metrics: Dict[str, float] = {}
    if is_main_process() or not distributed:
        encoder.eval()
        head_module.eval()
        all_targets = []
        all_preds = []

        if test_loader is not None:
            for batch in _yield_batches("test"):
                batch_x, batch_adj, batch_ptr, batch_labels, batch_meta = _move_batch_to_device(
                    batch, device_t, pin_memory_enabled
                )
                if batch_labels is None:
                    raise ValueError("Test loader returned samples without labels.")

                graph_emb = _get_graph_embeddings(batch_x, batch_adj, batch_ptr, batch_meta)
                param = next(head_param_source.parameters(), None)
                if param is not None and graph_emb.dtype != param.dtype:
                    graph_emb = graph_emb.to(param.dtype)

                with _amp_context():
                    preds = head_module(graph_emb).squeeze(1)
                preds = torch.nan_to_num(preds)
                all_preds.append(preds.detach().to(torch.float32).cpu().numpy())
                all_targets.append(batch_labels.detach().to(torch.float32).cpu().numpy())

        if all_targets and all_preds:
            y_true = np.concatenate(all_targets)
            y_pred = np.concatenate(all_preds)
        else:
            y_true = np.array([], dtype=np.float32)
            y_pred = np.array([], dtype=np.float32)
        # Final safety net
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        if task_type == "classification":
            metrics = compute_classification_metrics(y_true, y_pred)
        else:
            metrics = compute_regression_metrics(y_true, y_pred)
        if best_val_snapshot:
            metrics.update(best_val_snapshot)
        elif val_loader is not None:
            metrics.setdefault("val_loss", float("nan"))
        metrics["head"] = head_param_source

    if distributed:
        cleanup()
    return metrics
