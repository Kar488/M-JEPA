"""Supervised training routines for downstream tasks.

This module defines a function to train a simple linear head on top of a
frozen encoder for classification or regression tasks. The data is
split into train/validation/test sets using a stratified approach for
classification to ensure that each class appears in all splits when
possible. Performance metrics are computed using utilities from
`utils.metrics`.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple
import time as _time

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.mdataset import GraphDataset
from data.scaffold_split import scaffold_split_indices
from models.encoder import GNNEncoder
from utils.metrics import compute_classification_metrics, compute_regression_metrics
from utils.pooling import global_mean_pool
from utils.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


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
    use_scaffold: bool = False,
    devices: int = 1,
    *,
    max_batches: int = 0,
    time_budget_mins: int = 0,
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
        patience: Number of epochs with no improvement before stopping.
        use_scaffold: Whether to use scaffold split if SMILES are provided.
        devices: Number of GPUs for DDP.
        batch_indices: Optional list of indices for a single batch; if provided, overrides internal splitting.
        max batches: Maximum number of batches to train on; if set, overrides epochs
        time_budget_mins: Time budget in minutes for the training; if set, overrides epochs.
    Returns:
        A dictionary of metrics on the test set (only populated on rank 0).
    """
    logger.info(
        "Training linear head on %d graphs for %s task", len(dataset), task_type
    )
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
        
    head = nn.Linear(encoder.hidden_dim, 1).to(device_t)
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

    _start_wall = _time.time()

    def _time_left() -> bool:
        return (time_budget_mins <= 0) or ((_time.time() - _start_wall) < time_budget_mins * 60)
    
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
            batch_x, batch_adj, batch_ptr, batch_labels = dataset.get_batch(batch_indices)
            batch_x = batch_x.to(device_t)
            batch_adj = batch_adj.to(device_t)
            
            node_emb = encoder(batch_x, batch_adj)
            
            graph_emb = global_mean_pool(node_emb, batch_ptr.to(device_t))

            num_graphs = batch_ptr.numel() - 1
            if graph_emb.shape[0] != num_graphs:
                graph_emb = graph_emb[:num_graphs]
               
            preds = head(graph_emb).squeeze(1)
            # Guard against any numerical issues in the head
            if not torch.isfinite(preds).all():
                preds = torch.nan_to_num(preds)

            # Ensure targets are a Tensor on the same device/dtype as preds
            if batch_labels is not None:
                targets = batch_labels
                if not isinstance(targets, torch.Tensor):
                    targets = torch.tensor(
                        targets = torch.as_tensor(targets)
                    )
                targets = targets.to(device=device_t, dtype=preds.dtype, non_blocking=True)
            else:
                targets = torch.tensor(
                    dataset.labels[batch_indices],
                    dtype=preds.dtype,
                    device=device_t,
                )

            
            # Guard: ensure preds and targets match in length
            assert preds.shape[0] == targets.shape[0], (
                f"Preds length {preds.shape[0]} != targets length {targets.shape[0]}; "
                f"batch_ptr={batch_ptr.tolist()}, batch_indices={batch_indices}"
            )

            loss = loss_fn(preds, targets)
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
                batch_x, batch_adj, batch_ptr, _ = dataset.get_batch(batch_indices)
                batch_x = batch_x.to(device_t)
                batch_adj = batch_adj.to(device_t)
                node_emb = encoder(batch_x, batch_adj)
                graph_emb = global_mean_pool(node_emb, batch_ptr.to(device_t))
                
                num_graphs = batch_ptr.numel() - 1
                if graph_emb.shape[0] != num_graphs:
                    graph_emb = graph_emb[:num_graphs]

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
            batch_x, batch_adj, batch_ptr, _ = dataset.get_batch(batch_indices)
            batch_x = batch_x.to(device_t)
            batch_adj = batch_adj.to(device_t)
            node_emb = encoder(batch_x, batch_adj)
            graph_emb = global_mean_pool(node_emb, batch_ptr.to(device_t))

            num_graphs = batch_ptr.numel() - 1
            if graph_emb.shape[0] != num_graphs:
                graph_emb = graph_emb[:num_graphs]
            
            preds = head(graph_emb).squeeze(1).detach().cpu().numpy()
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
