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
import random
import time as _time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.mdataset import GraphDataset
from data.scaffold_split import scaffold_split_indices
from models.encoder import GNNEncoder
from utils.early_stopping import EarlyStopping
from utils.metrics import compute_classification_metrics, compute_regression_metrics
from utils.graph_ops import _encode_graph, _pool_graph_emb

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
            edges_np.append(ei_np + node_offset)

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

    return (
        torch.from_numpy(X),               # (N,F) float32
        torch.from_numpy(adj),             # (N,N) float32
        torch.from_numpy(ptr_arr),         # (B+1,) int64
        batch_labels_t,                    # (B,) float32 or None
    )

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

def train_linear_head(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    task_type: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda",
    patience: int = 10,
    num_workers=0, pin_memory=True, persistent_workers=True, prefetch_factor=4, bf16=False, 
    use_scaffold: bool = False,
    devices: int = 1,
    *,
    max_batches: int = 0,
    time_budget_mins: int = 0,
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
        num_workers: Number of subprocesses to use for data loading.
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
    for p in encoder.parameters():
        p.requires_grad = False
    num_graphs = len(dataset)
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

    in_dim = getattr(encoder, "hidden_dim", None) or getattr(encoder, "out_dim", None)
    if in_dim is None:
        # Fallback: infer embedding size from one sample
        with torch.no_grad():
            emb = _encode_graph(encoder, dataset.graphs[0])
            in_dim = int(emb.shape[-1])
    head = nn.Linear(in_dim, 1).to(device_t)

    if distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder, device_ids=[torch.cuda.current_device()] if device_t.type == "cuda" else None
        )
        head = nn.parallel.DistributedDataParallel(
            head, device_ids=[torch.cuda.current_device()] if device_t.type == "cuda" else None
        )
    loss_fn = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=patience) if patience > 0 else None

    rank = get_rank() if distributed else 0
    world = get_world_size() if distributed else 1
    train_idx_rank = train_idx[rank::world]

    _start_wall = _time.perf_counter()

    def _time_left() -> bool:
        return (time_budget_mins <= 0) or ((_time.perf_counter() - _start_wall) < time_budget_mins * 60)

    import contextlib
    # default autocast context so evaluation code has a valid handle even if
    # no training batches are processed (e.g. extremely small datasets)
    _amp_ctx = contextlib.nullcontext()
    
    for epoch in range(epochs):
        encoder.eval()
        head.train()
        batch_losses = []

        batches_done = 0
        for start in range(0, len(train_idx_rank), batch_size):
            if max_batches > 0 and batches_done >= max_batches:
                break
            if not _time_left():
                logger.info("Time budget hit during linear-head train epoch=%d; breaking.", epoch)
                break
       
            batch_indices = train_idx_rank[start : start + batch_size]
            assert getattr(dataset, "labels", None) is not None, "Dataset must have labels."            
            # autocast for the forward path (bf16 on 4090; else full precision)
            _amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (bf16 and device_t.type == "cuda")
                else contextlib.nullcontext()
            )

            with torch.no_grad():
                with _amp_ctx:
                    # Encode each graph in the batch. Prefer the faithful GraphData path
                    # (keeps edge_attr); fall back to (x, adj) for stub graphs in tests.
                    _embs = []
                    for i in batch_indices:
                        g_i = dataset.graphs[i]
                        # Prefer the GraphData path (preserves edge_attr) first
                        try:
                            h_nodes = _encode_graph(encoder, g_i)                      # [Ni, D]
                            g_vec   = _pool_graph_emb(h_nodes, g_i).reshape(1, -1)     # [1, D]
                        except Exception:
                            # Fallback for test stubs that expose to_tensors() and encoders that accept (x, adj)
                            if hasattr(g_i, "to_tensors"):
                                x_i, adj_i = g_i.to_tensors()
                                x_i   = x_i.to(device_t, non_blocking=True)
                                adj_i = adj_i.to(device_t, non_blocking=True)
                                try:
                                    h_nodes = encoder(x_i, adj_i)
                                except TypeError:
                                    h_nodes = encoder(x_i)
                                g_vec = h_nodes.mean(dim=0, keepdim=True)
                            else:
                                # Re-raise if no tensor fallback exists
                                raise
                        _embs.append(g_vec.to(device_t))
                    graph_emb = torch.cat(_embs, dim=0)                                 # [B, D]

            
            param = next(head.parameters(), None)
            if param is not None and graph_emb.dtype != param.dtype:
                # Option 1: cast the input tensor to the head’s dtype
                graph_emb = graph_emb.to(param.dtype)
                
            preds = head(graph_emb).squeeze(1)
            # Guard against any numerical issues in the head
            if not torch.isfinite(preds).all():
                preds = torch.nan_to_num(preds)

            # Targets on the same device/dtype as preds
            targets = torch.tensor(
                dataset.labels[batch_indices],
                dtype=preds.dtype,
                device=device_t,
            )
           
            # Guard: ensure preds and targets match in length
            assert preds.shape[0] == targets.shape[0], (
                f"Preds length {preds.shape[0]} != targets length {targets.shape[0]}; "
                f"batch_indices={batch_indices}"
            )
            with _amp_ctx:
                loss = loss_fn(preds.float(), targets.float())
            batch_losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            batches_done += 1

        logger.debug("Epoch %d training loss %.4f", epoch, float(np.mean(batch_losses)))
        if early_stopper is not None:
            encoder.eval()
            head.eval()
            val_losses = []

            for start in range(0, len(val_idx), batch_size):
                batch_indices = val_idx[start : start + batch_size]
                with torch.no_grad():
                    with _amp_ctx:
                        _embs = []
                        for i in batch_indices:
                            g_i = dataset.graphs[i]
                            try:
                                h_nodes = _encode_graph(encoder, g_i)
                                g_vec   = _pool_graph_emb(h_nodes, g_i).reshape(1, -1)
                            except Exception:
                                if hasattr(g_i, "to_tensors"):
                                    x_i, adj_i = g_i.to_tensors()
                                    x_i   = x_i.to(device_t, non_blocking=True)
                                    adj_i = adj_i.to(device_t, non_blocking=True)
                                    try:
                                        h_nodes = encoder(x_i, adj_i)
                                    except TypeError:
                                        h_nodes = encoder(x_i)
                                    g_vec = h_nodes.mean(dim=0, keepdim=True)
                                else:
                                    raise
                            _embs.append(g_vec.to(device_t))
                        graph_emb = torch.cat(_embs, dim=0)

                with _amp_ctx:
                    preds = head(graph_emb).squeeze(1)
                # Guard against any numerical issues in the head
                preds = torch.nan_to_num(preds)
                # Match target dtype/device to preds for MSE
                targets = torch.tensor(
                    dataset.labels[batch_indices],
                    dtype=preds.dtype,
                    device=device_t,
                )   
                vloss = loss_fn(preds, targets).item()
                val_losses.append(vloss)
            avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            avg_t = torch.tensor([avg_val_loss], device=device_t)
            if distributed:
                torch.distributed.all_reduce(avg_t, op=torch.distributed.ReduceOp.AVG)
            avg_val_loss = avg_t.item()
            if early_stopper.step(avg_val_loss):
                logger.info("Early stopping at epoch %d", epoch)
                break

    metrics: Dict[str, float] = {}
    if is_main_process() or not distributed:
        encoder.eval()
        head.eval()
        all_targets = []
        all_preds = []
       
        for start in range(0, len(test_idx), batch_size):
            batch_indices = test_idx[start : start + batch_size]
            with torch.no_grad():
                with _amp_ctx:
                    _embs = []
                    for i in batch_indices:
                        g_i = dataset.graphs[i]
                        try:
                            h_nodes = _encode_graph(encoder, g_i)
                            g_vec   = _pool_graph_emb(h_nodes, g_i).reshape(1, -1)
                        except Exception:
                            if hasattr(g_i, "to_tensors"):
                                x_i, adj_i = g_i.to_tensors()
                                x_i   = x_i.to(device_t, non_blocking=True)
                                adj_i = adj_i.to(device_t, non_blocking=True)
                                try:
                                    h_nodes = encoder(x_i, adj_i)
                                except TypeError:
                                    h_nodes = encoder(x_i)
                                g_vec = h_nodes.mean(dim=0, keepdim=True)
                            else:
                                raise
                        _embs.append(g_vec.to(device_t))
                    graph_emb = torch.cat(_embs, dim=0)

            with _amp_ctx:
                preds = head(graph_emb).squeeze(1)
            # Cast BF16/FP16 → FP32 before numpy conversion to avoid TypeError
            preds = preds.detach().to(torch.float32).cpu().numpy()
            # Sanitize before aggregation to keep sklearn happy
            preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
            targets = dataset.labels[batch_indices]
            all_targets.append(targets)
            all_preds.append(preds)
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        # Final safety net
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        if task_type == "classification":
            metrics = compute_classification_metrics(y_true, y_pred)
        else:
            metrics = compute_regression_metrics(y_true, y_pred)
        metrics["head"] = head.module if isinstance(head, nn.parallel.DistributedDataParallel) else head

    if distributed:
        cleanup()
    return metrics
