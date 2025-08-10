from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score

from data.mdataset import GraphData, GraphDataset


def _batch_iter(graphs: List[GraphData], labels: np.ndarray, batch_size: int):
    for i in range(0, len(graphs), batch_size):
        g = graphs[i : i + batch_size]
        y = labels[i : i + batch_size]
        yield g, y


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 2 else x.unsqueeze(0)


def train_multilabel_head(
    graphs: List[GraphData],
    labels: np.ndarray,  # shape [N, T], float32 with NaN for missing
    encoder: nn.Module,
    epochs: int = 30,
    lr: float = 5e-3,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict:
    device_t = torch.device(device)
    encoder.eval().to(device_t)  # frozen encoder
    with torch.no_grad():
        D = int(_ensure_2d(encoder(graphs[0])).shape[1])

    head = nn.Linear(D, labels.shape[1]).to(device_t)
    opt = optim.Adam(head.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def _forward(graph_batch: List[GraphData]):
        with torch.no_grad():
            H = encoder(graph_batch).to(device_t)  # [B, D]
        return head(H)

    # simple split (80/20) if needed
    n = len(graphs)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    def _gather(sub):
        return [graphs[i] for i in sub], labels[sub]

    g_tr, y_tr = _gather(train_idx)
    g_va, y_va = _gather(val_idx)

    for ep in range(1, epochs + 1):
        head.train()
        ep_loss = 0.0
        for g, y in _batch_iter(g_tr, y_tr, batch_size):
            y_t = torch.as_tensor(y, dtype=torch.float32, device=device_t)  # [B, T]
            logits = _forward(g)
            loss_mat = bce(logits, y_t)
            # mask missing labels (NaN)
            mask = ~torch.isnan(y_t)
            loss = (loss_mat[mask]).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu().item())

    # metrics on val
    head.eval()
    y_true_all, y_pred_all = [], []
    for g, y in _batch_iter(g_va, y_va, batch_size):
        logits = _forward(g)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device_t)
        y_true_all.append(y_t.cpu().numpy())
        y_pred_all.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    # Compute macro ROC-AUC/PR-AUC across tasks (skip tasks with 1 class or NaNs only)
    roc_list, pr_list = [], []
    for t in range(y_true.shape[1]):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        mask = ~np.isnan(yt)
        if mask.sum() < 2 or len(np.unique(yt[mask])) < 2:
            continue
        try:
            roc_list.append(roc_auc_score(yt[mask], yp[mask]))
            pr_list.append(average_precision_score(yt[mask], yp[mask]))
        except Exception:
            pass
    return {
        "val_roc_auc_macro": float(np.mean(roc_list)) if roc_list else 0.0,
        "val_pr_auc_macro": float(np.mean(pr_list)) if pr_list else 0.0,
        "tasks_evaluated": int(len(roc_list)),
    }
