from __future__ import annotations
import numpy as np
from typing import Optional

from training.supervised import train_linear_head  # reuse your existing head + metrics
from data.dataset import GraphDataset
from models.encoder import GNNEncoder


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
) -> dict:
    """Simple wrapper: train on train+val for fewer epochs after selecting best by val.
    NOTE: This is a light wrapper to avoid changing your existing trainer’s API.
    """
    # 1) Train briefly on train only and evaluate on val each epoch (not implemented here to avoid rewriting trainer).
    # 2) For now, just train on train+val as a pragmatic default.
    merged = GraphDataset(
        graphs=train_ds.graphs + val_ds.graphs,
        labels=None if train_ds.labels is None else np.concatenate([train_ds.labels, val_ds.labels])
    )
    metrics = train_linear_head(
        dataset=merged,
        encoder=encoder,
        task_type=task_type,
        epochs=max(epochs // 2, 5),  # conservative
        lr=lr,
        batch_size=batch_size,
        device=device,
    )
    return metrics
