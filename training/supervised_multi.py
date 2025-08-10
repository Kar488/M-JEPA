from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from data.mdataset import GraphDataset


@torch.no_grad()
def _embed_shard(encoder: nn.Module, graphs, batch: int, device: str) -> np.ndarray:
    encoder = encoder.to(device).eval()
    chunks: List[np.ndarray] = []
    for i in range(0, len(graphs), batch):
        H = encoder(graphs[i : i + batch]).detach().cpu().numpy()
        if H.ndim == 1:
            H = H[None, :]
        chunks.append(H)
    return (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.zeros((0, getattr(encoder, "hidden_dim", 1)), dtype=np.float32)
    )


def _worker_embed(
    rank: int,
    world: int,
    encoder_state: dict,
    graphs,
    batch: int,
    device_ids: List[int],
    out_path: str,
):
    device = f"cuda:{device_ids[rank]}" if torch.cuda.is_available() else "cpu"
    # Recreate encoder from state dict (encoder class import path must match your project)
    from models.factory import build_encoder  # uses same config saved by caller

    cfg = encoder_state.pop("_cfg")
    enc = build_encoder(**cfg)
    enc.load_state_dict(encoder_state)
    # shard indices
    n = len(graphs)
    start = (n * rank) // world
    end = (n * (rank + 1)) // world
    X = _embed_shard(enc, graphs[start:end], batch, device)
    np.save(f"{out_path}.rank{rank}.npy", X)


def compute_embeddings_multi_gpu(
    encoder: nn.Module,
    dataset: GraphDataset,
    batch: int,
    device_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """Shard dataset across GPUs, compute embeddings in parallel, then concatenate."""
    if device_ids is None or len(device_ids) <= 1:
        return _embed_shard(
            encoder,
            dataset.graphs,
            batch,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    world = len(device_ids)
    tmp = Path("outputs/emb_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    out_prefix = str(tmp / "emb")
    # package minimal cfg so workers can rebuild the encoder
    cfg = dict(
        gnn_type=getattr(encoder, "gnn_type", "mpnn"),
        input_dim=getattr(encoder, "input_dim", dataset.graphs[0].x.shape[1]),
        hidden_dim=getattr(encoder, "hidden_dim", 256),
        num_layers=getattr(encoder, "num_layers", 3),
        edge_dim=getattr(encoder, "edge_dim", None),
    )
    state = copy.deepcopy(encoder.state_dict())
    state["_cfg"] = cfg
    ctx = mp.get_context("spawn")
    procs = []
    for r in range(world):
        p = ctx.Process(
            target=_worker_embed,
            args=(r, world, state, dataset.graphs, batch, device_ids, out_prefix),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    chunks = [np.load(f"{out_prefix}.rank{r}.npy") for r in range(world)]
    return np.concatenate(chunks, axis=0)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, task_type: str):
        super().__init__()
        self.out = nn.Linear(in_dim, 1)
        self.task_type = task_type

    def forward(self, x):
        return self.out(x).squeeze(-1)


def train_linear_head_earlystop(
    encoder: nn.Module,
    train_ds: GraphDataset,
    val_ds: GraphDataset,
    task_type: str = "classification",
    epochs: int = 50,
    lr: float = 5e-3,
    batch_size: int = 128,
    device: str = "cuda",
    patience: int = 7,
    multi_gpu_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # 1) Precompute embeddings (parallel if multi_gpu_ids provided)
    X_tr = compute_embeddings_multi_gpu(
        encoder, train_ds, batch=batch_size, device_ids=multi_gpu_ids
    )
    X_va = compute_embeddings_multi_gpu(
        encoder, val_ds, batch=batch_size, device_ids=multi_gpu_ids
    )
    y_tr = torch.tensor(train_ds.labels, dtype=torch.float32, device=device)
    y_va = torch.tensor(val_ds.labels, dtype=torch.float32, device=device)

    in_dim = X_tr.shape[1]
    head = LinearHead(in_dim, task_type).to(device)
    opt = optim.AdamW(head.parameters(), lr=lr)
    if task_type == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    # 2) Train with early stopping
    best = dict(loss=float("inf"), state=None, epoch=0)
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_va, dtype=torch.float32, device=device)

    for ep in range(1, epochs + 1):
        head.train()
        opt.zero_grad(set_to_none=True)
        pred = head(Xtr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            val_loss = loss_fn(head(Xva), y_va).item()

        if val_loss < best["loss"]:
            best.update(loss=val_loss, state=copy.deepcopy(head.state_dict()), epoch=ep)
        elif ep - best["epoch"] >= patience:
            break

    # restore best
    if best["state"] is not None:
        head.load_state_dict(best["state"])

    # 3) Report metrics on val (you can extend to test similarly)
    out = {"val_loss": float(best["loss"])}
    if task_type == "classification":
        from sklearn.metrics import average_precision_score, roc_auc_score

        with torch.no_grad():
            pv = torch.sigmoid(head(Xva)).cpu().numpy()
        yv = y_va.detach().cpu().numpy()
        out["val_roc_auc"] = float(roc_auc_score(yv, pv))
        out["val_pr_auc"] = float(average_precision_score(yv, pv))
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        with torch.no_grad():
            pv = head(Xva).cpu().numpy()
        yv = y_va.detach().cpu().numpy()
        out["val_rmse"] = float(np.sqrt(mean_squared_error(yv, pv)))
        out["val_mae"] = float(mean_absolute_error(yv, pv))
        out["val_r2"] = float(r2_score(yv, pv))
    return out
