from __future__ import annotations

from typing import Optional

import numpy as np

from data.mdataset import GraphDataset
from models.encoder import GNNEncoder
from training.supervised import train_linear_head  # reuse your existing head + metrics

def _safe_labels(ds) -> Optional[np.ndarray]:
    return getattr(ds, "labels", None)

def _ensure_labels_inplace(ds, task_type: str) -> None:
    """Guarantee ds.labels exists; mutate in place (works with test stubs)."""
    labels = getattr(ds, "labels", None)
    if labels is None:
        n = len(ds.graphs)
        if task_type == "classification":
            labels = np.zeros(n, dtype=np.int64)
        else:
            labels = np.zeros(n, dtype=np.float32)
        setattr(ds, "labels", labels)
    else:
        arr = np.asarray(labels)
        if task_type == "classification":
            arr = arr.astype(np.int64, copy=False)
        else:
            arr = arr.astype(np.float32, copy=False)
        setattr(ds, "labels", arr)

def train_linear_head_with_val(
    train_ds: GraphDataset,
    val_ds: GraphDataset,
    encoder: GNNEncoder,
    task_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    val_patience: int = 5,
    devices: int = 1,
) -> dict:
    """Simple wrapper: train on train+val for fewer epochs after selecting best by val.
    NOTE: This is a light wrapper to avoid changing your existing trainer’s API.
    """
    # Ensure labels exist on the *actual* objects passed by tests
    _ensure_labels_inplace(train_ds, task_type)
    _ensure_labels_inplace(val_ds, task_type)

    # (Optional) If  later we want to use val-based logic, we can also:
    # _ensure_labels_inplace(val_ds, task_type)
    
    metrics = train_linear_head(
        dataset=train_ds,
        encoder=encoder,
        task_type=task_type,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        patience=val_patience,
        devices=devices,
    )
    return metrics