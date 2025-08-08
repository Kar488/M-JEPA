from __future__ import annotations
from typing import List, Tuple, Optional
import math
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data.dataset import GraphDataset, GraphData
from utils.pooling import global_mean_pool

# ---- Optional utilities (graceful fallbacks) ---------------------------------
try:
    from utils.schedule import cosine_with_warmup  # cosine decay with warm-up
except Exception:  # pragma: no cover
    cosine_with_warmup = None  # type: ignore

try:
    from utils.checkpoint import save_checkpoint
except Exception:  # pragma: no cover
    def save_checkpoint(*_, **__):  # type: ignore
        pass

try:
    from utils.logging import maybe_init_wandb
except Exception:  # pragma: no cover
    class _DummyWB:
        def log(self, *a, **k): pass
        def finish(self): pass
    def maybe_init_wandb(enable: bool, project: str = "m-jepa", config=None):  # type: ignore
        return _DummyWB()


# =============================================================================
# Helper functions
# =============================================================================

def _batch_iter(graphs: List[GraphData], batch_size: int):
    """Yield small lists of GraphData."""
    for i in range(0, len(graphs), batch_size):
        yield graphs[i:i + batch_size]


def _subgraph(g: GraphData, node_idx: List[int]) -> GraphData:
    """Extract an induced subgraph specified by node indices."""
    if len(node_idx) == 0 or g.x.shape[0] == 0:
        x = np.zeros((0, g.x.shape[1]), dtype=np.float32)
        e = np.zeros((2, 0), dtype=np.int64)
        ea = None if g.edge_attr is None else np.zeros((0, g.edge_attr.shape[1]), dtype=np.float32)
        return GraphData(x=x, edge_index=e, edge_attr=ea)

    remap = {old: new for new, old in enumerate(node_idx)}
    mask = np.isin(g.edge_index[0], node_idx) & np.isin(g.edge_index[1], node_idx)
    e = g.edge_index[:, mask].copy()
    for t in range(e.shape[1]):
        e[0, t] = remap[int(e[0, t])]
        e[1, t] = remap[int(e[1, t])]
    x = g.x[node_idx]
    ea = g.edge_attr[mask] if g.edge_attr is not None else None
    return GraphData(x=x, edge_index=e, edge_attr=ea)


def _mask_subgraph(g: GraphData, mask_ratio: float, contiguous: bool = False) -> Tuple[GraphData, GraphData]:
    """Split nodes into context/target sets and return two induced subgraphs."""
    n = int(g.x.shape[0])
    if n == 0:
        return g, g
    k = max(1, int(math.ceil(mask_ratio * n)))
    if contiguous:
        start = np.random.randint(0, n)
        tgt_idx = [(start + j) % n for j in range(k)]
    else:
        tgt_idx = np.random.choice(n, size=k, replace=False).tolist()
    ctx_idx = [i for i in range(n) if i not in set(tgt_idx)]
    return _subgraph(g, ctx_idx), _subgraph(g, tgt_idx)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 2 else x.unsqueeze(0)


# =============================================================================
# JEPA pretraining
# =============================================================================

def train_jepa(
    *,
    dataset: GraphDataset,
    encoder: nn.Module,
    ema_encoder: nn.Module,
    predictor: nn.Module,
    ema,  # EMA helper that has .update(encoder)
    epochs: int = 100,
    batch_size: int = 256,
    mask_ratio: float = 0.15,
    contiguous: bool = False,
    lr: float = 1e-4,
    device: str = "cuda",
    reg_lambda: float = 1e-4,
    # logging / ckpt / sched
    use_wandb: bool = False,
    wandb_project: str = "m-jepa",
    wandb_run: Optional[str] = None,   # not used but kept for API symmetry
    ckpt_path: Optional[str] = None,
    ckpt_every: int = 10,
    use_scheduler: bool = True,
    warmup_steps: int = 1000,
) -> List[float]:
    """
    JEPA training loop:
      - Sample (context, target) subgraphs
      - Encode: h_c = f_θ(V_c), h_t = f_ξ(V_t)
      - Predict: \hat h_t = g_ψ(h_c)
      - Minimise: ||\hat h_t - h_t||^2 + λ ||ψ||^2
      - Update EMA: ξ ← τ ξ + (1-τ) θ
    """
    device_t = torch.device(device)
    encoder.to(device_t).train()
    ema_encoder.to(device_t).eval()
    predictor.to(device_t).train()

    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    steps_per_epoch = max(1, math.ceil(len(dataset.graphs) / batch_size))
    total_steps = epochs * steps_per_epoch

    scheduler = None
    if use_scheduler and cosine_with_warmup is not None:
        scheduler = cosine_with_warmup(opt, warmup_steps=warmup_steps, total_steps=total_steps)

    wb = maybe_init_wandb(use_wandb, project=wandb_project, config=dict(
        method="jepa", lr=lr, batch_size=batch_size, mask_ratio=mask_ratio, contiguous=contiguous
    ))

    losses: List[float] = []
    mse = nn.MSELoss()
    step = 0

    if ckpt_path:
        os.makedirs(ckpt_path, exist_ok=True)

    for ep in range(1, epochs + 1):
        ep_loss = 0.0

        for batch in _batch_iter(dataset.graphs, batch_size):
            ctx_list, tgt_list = [], []

            # Generate pairs
            for g in batch:
                g_ctx, g_tgt = _mask_subgraph(g, mask_ratio, contiguous)

                # Encode
                hc = encoder(g_ctx)              # expect [1, D] or [D]
                with torch.no_grad():
                    ht = ema_encoder(g_tgt)

                ctx_list.append(_ensure_2d(hc))
                tgt_list.append(_ensure_2d(ht))

            h_c = torch.cat(ctx_list, dim=0).to(device_t)  # [B, D]
            h_t = torch.cat(tgt_list, dim=0).to(device_t)  # [B, D]

            # Predictor + loss
            pred = predictor(h_c)
            l2_reg = torch.zeros((), device=device_t)
            for p in predictor.parameters():
                l2_reg = l2_reg + (p ** 2).sum()
            loss = mse(pred, h_t) + reg_lambda * l2_reg

            # Step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()

            # EMA update
            ema.update(encoder)

            # Logging
            lossv = float(loss.detach().cpu().item())
            ep_loss += lossv
            step += 1
            if use_wandb:
                wb.log({
                    "train/jepa_loss": lossv,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "step": step,
                    "epoch": ep
                })

        ep_loss /= steps_per_epoch
        losses.append(ep_loss)
        if use_wandb:
            wb.log({"epoch/jepa_loss": ep_loss, "epoch": ep})

        # Checkpoint
        if ckpt_path and (ep % ckpt_every == 0 or ep == epochs):
            save_checkpoint(
                os.path.join(ckpt_path, f"jepa_ep{ep:04d}.pt"),
                encoder=encoder.state_dict(),
                ema_encoder=ema_encoder.state_dict(),
                predictor=predictor.state_dict(),
                epoch=ep,
            )

    try:
        wb.finish()
    except Exception:
        pass

    return losses


# =============================================================================
# Contrastive baseline (SimCLR‑style)
# =============================================================================

def train_contrastive(
    *,
    dataset: GraphDataset,
    encoder: nn.Module,
    projection_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 256,
    mask_ratio: float = 0.15,
    lr: float = 1e-4,
    device: str = "cuda",
    temperature: float = 0.1,
    # logging / ckpt / sched
    use_wandb: bool = False,
    wandb_project: str = "m-jepa",
    wandb_run: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    ckpt_every: int = 10,
    use_scheduler: bool = True,
    warmup_steps: int = 1000,
) -> List[float]:
    """
    Very small SimCLR‑style contrastive trainer using node masking as augmentation.
    Two masked views v1 and v2 are encoded and projected, and NT‑Xent loss is applied.
    """
    device_t = torch.device(device)
    encoder.to(device_t).train()

    # Projector head
    try:
        d = encoder.proj.out_features  # some encoders expose this
    except Exception:
        # fall back to a reasonable guess; will be overwritten on first forward
        d = 256
    proj = nn.Sequential(
        nn.Linear(d, projection_dim),
        nn.ReLU(inplace=True),
        nn.Linear(projection_dim, projection_dim),
    ).to(device_t)

    opt = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=lr)
    steps_per_epoch = max(1, math.ceil(len(dataset.graphs) / batch_size))
    total_steps = epochs * steps_per_epoch
    scheduler = None
    if use_scheduler and cosine_with_warmup is not None:
        scheduler = cosine_with_warmup(opt, warmup_steps=warmup_steps, total_steps=total_steps)

    wb = maybe_init_wandb(use_wandb, project=wandb_project, config=dict(
        method="contrastive", lr=lr, batch_size=batch_size, mask_ratio=mask_ratio, temperature=temperature
    ))

    if ckpt_path:
        os.makedirs(ckpt_path, exist_ok=True)

    losses: List[float] = []
    step = 0

    for ep in range(1, epochs + 1):
        ep_loss = 0.0

        for batch in _batch_iter(dataset.graphs, batch_size):
            z1_list, z2_list = [], []

            for g in batch:
                v1, _ = _mask_subgraph(g, mask_ratio, contiguous=False)
                v2, _ = _mask_subgraph(g, mask_ratio, contiguous=False)

                h1 = _ensure_2d(encoder(v1))
                h2 = _ensure_2d(encoder(v2))

                # lazily infer encoder output dim for projector first time
                if isinstance(proj[0], nn.Linear) and proj[0].in_features != h1.size(1):
                    in_dim = h1.size(1)
                    proj[0] = nn.Linear(in_dim, projection_dim).to(device_t)

                z1_list.append(proj(h1))
                z2_list.append(proj(h2))

            z1 = torch.cat(z1_list, dim=0)  # [B, P]
            z2 = torch.cat(z2_list, dim=0)  # [B, P]

            # NT-Xent
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            logits = z1 @ z2.t() / temperature
            target = torch.arange(z1.size(0), device=device_t)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()

            lossv = float(loss.detach().cpu().item())
            ep_loss += lossv
            step += 1
            if use_wandb:
                wb.log({
                    "train/contrastive_loss": lossv,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "step": step,
                    "epoch": ep
                })

        ep_loss /= steps_per_epoch
        losses.append(ep_loss)
        if use_wandb:
            wb.log({"epoch/contrastive_loss": ep_loss, "epoch": ep})

        if ckpt_path and (ep % ckpt_every == 0 or ep == epochs):
            save_checkpoint(
                os.path.join(ckpt_path, f"contrastive_ep{ep:04d}.pt"),
                encoder=encoder.state_dict(),
                projector=proj.state_dict(),
                epoch=ep,
            )

    try:
        wb.finish()
    except Exception:
        pass

    return losses
