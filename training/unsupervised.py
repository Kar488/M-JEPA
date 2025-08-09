from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.dataset import GraphData, GraphDataset
from data.augment import apply_graph_augmentations
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logging import maybe_init_wandb
from utils.schedule import cosine_with_warmup
from utils.ddp import (
    DistributedSamplerList,
    init_distributed,
    is_main_process,
    cleanup,
)


def _batch_iter(graphs: List[GraphData], batch_size: int):
    for i in range(0, len(graphs), batch_size):
        yield graphs[i : i + batch_size]


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 2 else x.unsqueeze(0)


def _subgraph(g: GraphData, idx: List[int]) -> GraphData:
    if len(idx) == 0 or g.x.shape[0] == 0:
        import numpy as np

        x = np.zeros((0, g.x.shape[1]), dtype=np.float32)
        e = np.zeros((2, 0), dtype=np.int64)
        ea = (
            None
            if g.edge_attr is None
            else np.zeros((0, g.edge_attr.shape[1]), dtype=np.float32)
        )
        return GraphData(x=x, edge_index=e, edge_attr=ea)
    remap = {old: new for new, old in enumerate(idx)}
    import numpy as np

    mask = np.isin(g.edge_index[0], idx) & np.isin(g.edge_index[1], idx)
    e = g.edge_index[:, mask].copy()
    for t in range(e.shape[1]):
        e[0, t] = remap[int(e[0, t])]
        e[1, t] = remap[int(e[1, t])]
    x = g.x[idx]
    ea = g.edge_attr[mask] if g.edge_attr is not None else None
    return GraphData(x=x, edge_index=e, edge_attr=ea)


def _mask_subgraph(
    g: GraphData, mask_ratio: float, contiguous: bool
) -> Tuple[GraphData, GraphData]:
    n = int(g.x.shape[0])
    if n == 0:
        return g, g
    k = max(1, int(math.ceil(mask_ratio * n)))
    if contiguous:
        start = np.random.randint(0, n)
        tgt = [(start + j) % n for j in range(k)]
    else:
        tgt = np.random.choice(n, size=k, replace=False).tolist()
    ctx = [i for i in range(n) if i not in set(tgt)]
    return _subgraph(g, ctx), _subgraph(g, tgt)


def train_jepa(
    *,
    dataset: GraphDataset,
    encoder: nn.Module,
    ema_encoder: nn.Module,
    predictor: nn.Module,
    ema,
    epochs: int = 100,
    batch_size: int = 256,
    mask_ratio: float = 0.15,
    contiguous: bool = False,
    lr: float = 1e-4,
    device: str = "cuda",
    devices: int = 1,
    reg_lambda: float = 1e-4,
    use_wandb: bool = False,
    wandb_project: str = "m-jepa",
    wandb_tags: Optional[List[str]] = None,
    ckpt_path: Optional[str] = None,
    ckpt_every: int = 10,
    use_scheduler: bool = True,
    warmup_steps: int = 1000,
    use_amp: bool = True,
    random_rotate: bool = False,
    mask_angle: bool = False,
    perturb_dihedral: bool = False,
    resume_from: Optional[str] = None,
) -> List[float]:
    distributed = init_distributed()
    device_t = torch.device(device)
    if distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder.to(device_t),
            device_ids=[torch.cuda.current_device()] if device_t.type == "cuda" else None,
        )
        predictor = nn.parallel.DistributedDataParallel(
            predictor.to(device_t),
            device_ids=[torch.cuda.current_device()] if device_t.type == "cuda" else None,
        )
        ema_encoder = ema_encoder.to(device_t).eval()
        encoder.train()
        predictor.train()
    else:
        encoder.to(device_t).train()
        ema_encoder.to(device_t).eval()
        predictor.to(device_t).train()
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_t.type == "cuda")
    steps_per_epoch = max(1, math.ceil(len(dataset.graphs) / batch_size))
    total_steps = epochs * steps_per_epoch
    sch = cosine_with_warmup(opt, warmup_steps, total_steps) if use_scheduler else None
    wb = maybe_init_wandb(
        use_wandb,
        project=wandb_project,
        config=dict(method="jepa", lr=lr, mask_ratio=mask_ratio, contiguous=contiguous),
        tags=wandb_tags,
    )

    start_epoch = 1
    if resume_from and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from)
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        if "ema_encoder" in ckpt:
            ema_encoder.load_state_dict(ckpt["ema_encoder"])
        if "predictor" in ckpt:
            predictor.load_state_dict(ckpt["predictor"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and isinstance(scaler, torch.cuda.amp.GradScaler):
            scaler.load_state_dict(ckpt["scaler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

    losses: List[float] = []
    mse = nn.MSELoss()
    step = 0
    if ckpt_path:
        os.makedirs(ckpt_path, exist_ok=True)

    for ep in range(start_epoch, epochs + 1):
        ep_loss = 0.0
        data_iter = (
            list(DistributedSamplerList(dataset.graphs)) if distributed else dataset.graphs
        )
        for batch in _batch_iter(data_iter, batch_size):
            ctx_list, tgt_list = [], []
            for g in batch:
                if random_rotate or mask_angle or perturb_dihedral:
                    g = apply_graph_augmentations(
                        g,
                        rotate=random_rotate,
                        mask_angle=mask_angle,
                        perturb_dihedral=perturb_dihedral,
                    )
                g_ctx, g_tgt = _mask_subgraph(g, mask_ratio, contiguous)
                with torch.cuda.amp.autocast(
                    enabled=use_amp and device_t.type == "cuda"
                ):
                    h_c = encoder(g_ctx)
                    with torch.no_grad():
                        h_t = ema_encoder(g_tgt)
                ctx_list.append(_ensure_2d(h_c))
                tgt_list.append(_ensure_2d(h_t))

            h_c = torch.cat(ctx_list, dim=0).to(device_t)
            h_t = torch.cat(tgt_list, dim=0).to(device_t)
            with torch.cuda.amp.autocast(enabled=use_amp and device_t.type == "cuda"):
                pred = predictor(h_c)
                l2_reg = sum((p**2).sum() for p in predictor.parameters())
                loss = mse(pred, h_t) + reg_lambda * l2_reg

            opt.zero_grad(set_to_none=True)
            if isinstance(scaler, torch.cuda.amp.GradScaler):
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            if sch is not None:
                sch.step()

            ema.update(encoder)

            lv = float(loss.detach().cpu().item())
            ep_loss += lv
            step += 1
            if wb and is_main_process():
                wb.log(
                    {
                        "train/jepa_loss": lv,
                        "lr": float(opt.param_groups[0]["lr"]),
                        "step": step,
                        "epoch": ep,
                    }
                )

        ep_loss /= steps_per_epoch
        if is_main_process():
            losses.append(ep_loss)
            if wb:
                wb.log({"epoch/jepa_loss": ep_loss, "epoch": ep})
            if ckpt_path and (ep % ckpt_every == 0 or ep == epochs):
                save_checkpoint(
                    os.path.join(ckpt_path, f"jepa_ep{ep:04d}.pt"),
                    encoder=encoder.state_dict(),
                    ema_encoder=ema_encoder.state_dict(),
                    predictor=predictor.state_dict(),
                    optimizer=opt.state_dict(),
                    scaler=(
                        scaler.state_dict()
                        if isinstance(scaler, torch.cuda.amp.GradScaler)
                        else None
                    ),
                    epoch=ep,
                )
    try:
        if wb and is_main_process():
            wb.finish()
    except Exception:
        pass
    if distributed:
        cleanup()
    return losses


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
    devices: int = 1,
    temperature: float = 0.1,
    use_wandb: bool = False,
    wandb_project: str = "m-jepa",
    wandb_tags: Optional[List[str]] = None,
    ckpt_path: Optional[str] = None,
    ckpt_every: int = 10,
    use_scheduler: bool = True,
    warmup_steps: int = 1000,
    use_amp: bool = True,    
    random_rotate: bool = False,
    mask_angle: bool = False,
    perturb_dihedral: bool = False,
    resume_from: Optional[str] = None,
) -> List[float]:
    distributed = init_distributed()
    device_t = torch.device(device)
    if distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder.to(device_t),
            device_ids=[torch.cuda.current_device()] if device_t.type == "cuda" else None,
        )
        proj = nn.Sequential(
            nn.Linear(256, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        ).to(device_t)
        opt = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=lr)
        encoder.train()
        proj.train()
    else:
        encoder.to(device_t).train()
        proj = nn.Sequential(
            nn.Linear(256, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        ).to(device_t)
        opt = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_t.type == "cuda")
    steps_per_epoch = max(1, math.ceil(len(dataset.graphs) / batch_size))
    total_steps = epochs * steps_per_epoch
    sch = cosine_with_warmup(opt, warmup_steps, total_steps) if use_scheduler else None
    wb = maybe_init_wandb(
        use_wandb,
        project=wandb_project,
        config=dict(method="contrastive", lr=lr, mask_ratio=mask_ratio),
        tags=wandb_tags,
    )

    if resume_from and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from)
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        if "projector" in ckpt:
            proj.load_state_dict(ckpt["projector"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and isinstance(scaler, torch.cuda.amp.GradScaler):
            scaler.load_state_dict(ckpt["scaler"])

    losses: List[float] = []
    step = 0
    if ckpt_path:
        os.makedirs(ckpt_path, exist_ok=True)

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        data_iter = (
            list(DistributedSamplerList(dataset.graphs)) if distributed else dataset.graphs
        )
        for batch in _batch_iter(data_iter, batch_size):
            z1_list, z2_list = [], []
            for g in batch:
                if random_rotate or mask_angle or perturb_dihedral:
                    g = apply_graph_augmentations(
                        g,
                        rotate=random_rotate,
                        mask_angle=mask_angle,
                        perturb_dihedral=perturb_dihedral,
                    )
                v1, _ = _mask_subgraph(g, mask_ratio, contiguous=False)
                v2, _ = _mask_subgraph(g, mask_ratio, contiguous=False)
                with torch.cuda.amp.autocast(
                    enabled=use_amp and device_t.type == "cuda"
                ):
                    h1 = _ensure_2d(encoder(v1))
                    h2 = _ensure_2d(encoder(v2))
                    if isinstance(proj[0], nn.Linear) and proj[
                        0
                    ].in_features != h1.size(1):
                        proj[0] = nn.Linear(h1.size(1), proj[0].out_features).to(
                            device_t
                        )
                    z1_list.append(proj(h1))
                    z2_list.append(proj(h2))
            z1 = F.normalize(torch.cat(z1_list, dim=0), dim=-1)
            z2 = F.normalize(torch.cat(z2_list, dim=0), dim=-1)
            with torch.cuda.amp.autocast(enabled=use_amp and device_t.type == "cuda"):
                logits = z1 @ z2.t() / temperature
                target = torch.arange(z1.size(0), device=device_t)
                loss = 0.5 * (
                    F.cross_entropy(logits, target)
                    + F.cross_entropy(logits.t(), target)
                )
            opt.zero_grad(set_to_none=True)
            if isinstance(scaler, torch.cuda.amp.GradScaler):
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            if sch is not None:
                sch.step()
            lv = float(loss.detach().cpu().item())
            ep_loss += lv
            step += 1
            if wb and is_main_process():
                wb.log(
                    {
                        "train/contrastive_loss": lv,
                        "lr": float(opt.param_groups[0]["lr"]),
                        "step": step,
                        "epoch": ep,
                    }
                )
        ep_loss /= steps_per_epoch
        if is_main_process():
            losses.append(ep_loss)
            if wb:
                wb.log({"epoch/contrastive_loss": ep_loss, "epoch": ep})
            if ckpt_path and (ep % ckpt_every == 0 or ep == epochs):
                save_checkpoint(
                    os.path.join(ckpt_path, f"contrastive_ep{ep:04d}.pt"),
                    encoder=encoder.state_dict(),
                    projector=proj.state_dict(),
                    optimizer=opt.state_dict(),
                    scaler=(
                        scaler.state_dict()
                        if isinstance(scaler, torch.cuda.amp.GradScaler)
                        else None
                    ),
                    epoch=ep,
                )
    try:
        if wb and is_main_process():
            wb.finish()
    except Exception:
        pass
    if distributed:
        cleanup()
    return losses
