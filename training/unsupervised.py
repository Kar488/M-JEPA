"""Unsupervised training routines for JEPA and contrastive baselines.

This module contains functions to train the Joint‑Embedding Predictive
Architecture (JEPA) and a simple contrastive baseline on unlabelled
molecular graphs. The routines are designed to operate on batches of
graphs using the block‑diagonal batching strategy provided by the
`data` package.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import GraphDataset, sample_subgraphs
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from models.ema import EMA
from utils.pooling import global_mean_pool
from utils.metrics import compute_classification_metrics, compute_regression_metrics  # noqa: F401 (imported for side effect)


def regularisation_loss(model: nn.Module, lam: float = 1e-4) -> torch.Tensor:
    """L2 regularisation on model parameters."""
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        reg += torch.sum(p ** 2)
    return lam * reg


def train_jepa(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    ema_encoder: GNNEncoder,
    predictor: MLPPredictor,
    ema: EMA,
    epochs: int = 10,
    batch_size: int = 32,
    mask_ratio: float = 0.15,
    contiguous: bool = False,
    lr: float = 1e-4,
    device: str = "cpu",
    reg_lambda: float = 1e-4,
) -> List[float]:
    """Train JEPA on unlabelled graphs.

    Returns a list of mean losses per epoch.
    """
    encoder = encoder.to(device)
    ema_encoder = ema_encoder.to(device)
    predictor = predictor.to(device)
    optimiser = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    losses: List[float] = []
    indices = list(range(len(dataset)))
    for epoch in range(epochs):
        random.shuffle(indices)
        batch_losses = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            contexts = []
            targets = []
            for idx in batch_indices:
                g = dataset.graphs[idx]
                ctx_g, tgt_g = sample_subgraphs(g, mask_ratio=mask_ratio, contiguous=contiguous)
                contexts.append(ctx_g)
                targets.append(tgt_g)
            ctx_data = GraphDataset(contexts)
            tgt_data = GraphDataset(targets)
            ctx_x, ctx_adj, ctx_ptr, _ = ctx_data.get_batch(list(range(len(contexts))))
            tgt_x, tgt_adj, tgt_ptr, _ = tgt_data.get_batch(list(range(len(targets))))
            ctx_x = ctx_x.to(device)
            ctx_adj = ctx_adj.to(device)
            tgt_x = tgt_x.to(device)
            tgt_adj = tgt_adj.to(device)
            ctx_node_emb = encoder(ctx_x, ctx_adj)
            ctx_graph_emb = global_mean_pool(ctx_node_emb, ctx_ptr.to(device))
            ema.copy_to(ema_encoder)
            with torch.no_grad():
                tgt_node_emb = ema_encoder(tgt_x, tgt_adj)
                tgt_graph_emb = global_mean_pool(tgt_node_emb, tgt_ptr.to(device))
            pred_tgt_emb = predictor(ctx_graph_emb)
            mse_loss = F.mse_loss(pred_tgt_emb, tgt_graph_emb.detach())
            reg_loss = regularisation_loss(predictor, lam=reg_lambda)
            loss = mse_loss + reg_loss
            batch_losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            ema.update(encoder)
        losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
    return losses


def train_contrastive(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    projection_dim: int = 128,
    epochs: int = 10,
    batch_size: int = 32,
    mask_ratio: float = 0.15,
    lr: float = 1e-4,
    device: str = "cpu",
    temperature: float = 0.1,
) -> List[float]:
    """Train a simple contrastive baseline.

    Returns a list of mean losses per epoch.
    """
    encoder = encoder.to(device)
    proj_head = nn.Sequential(
        nn.Linear(encoder.hidden_dim, projection_dim),
        nn.ReLU(),
        nn.Linear(projection_dim, projection_dim),
    ).to(device)
    optimiser = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=lr)
    losses: List[float] = []
    indices = list(range(len(dataset)))
    for epoch in range(epochs):
        random.shuffle(indices)
        batch_losses = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            view1_graphs = []
            view2_graphs = []
            for idx in batch_indices:
                g = dataset.graphs[idx]
                v1, _ = sample_subgraphs(g, mask_ratio=mask_ratio, contiguous=False)
                # Randomly choose contiguous or random masking for second view
                if random.random() < 0.5:
                    v2, _ = sample_subgraphs(g, mask_ratio=mask_ratio, contiguous=True)
                else:
                    v2, _ = sample_subgraphs(g, mask_ratio=mask_ratio, contiguous=False)
                view1_graphs.append(v1)
                view2_graphs.append(v2)
            d1 = GraphDataset(view1_graphs)
            v1_x, v1_adj, v1_ptr, _ = d1.get_batch(list(range(len(view1_graphs))))
            d2 = GraphDataset(view2_graphs)
            v2_x, v2_adj, v2_ptr, _ = d2.get_batch(list(range(len(view2_graphs))))
            v1_x, v1_adj, v2_x, v2_adj = (
                v1_x.to(device), v1_adj.to(device), v2_x.to(device), v2_adj.to(device)
            )
            h1_node = encoder(v1_x, v1_adj)
            h2_node = encoder(v2_x, v2_adj)
            h1_graph = global_mean_pool(h1_node, v1_ptr.to(device))
            h2_graph = global_mean_pool(h2_node, v2_ptr.to(device))
            z1 = proj_head(h1_graph)
            z2 = proj_head(h2_graph)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            # Similarity matrix: z1 against both z2 and z1 (for negatives)
            sim_matrix = torch.mm(z1, torch.cat([z2, z1], dim=0).t()) / temperature
            labels = torch.arange(0, len(batch_indices), device=device)
            mask = torch.eye(len(batch_indices), device=device)
            logits = torch.cat([
                sim_matrix[:, : len(batch_indices)],
                sim_matrix[:, len(batch_indices):] * (1.0 - mask),
            ], dim=1)
            targets = labels
            loss = F.cross_entropy(logits, targets)
            batch_losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
    return losses
