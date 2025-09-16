from __future__ import annotations

import math
import os
import time as _t
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

try:
    from data.augment import (
        apply_graph_augmentations,
        delete_random_bond,
        mask_random_atom,
        remove_random_subgraph,
        mask_subgraph,
        generate_views,
    )
except ImportError:  # pragma: no cover - used in minimal test stubs
    from data.augment import apply_graph_augmentations

    def delete_random_bond(g):
        return g

    def mask_random_atom(g):
        return g

    def remove_random_subgraph(g):
        return g

    def mask_subgraph(g, mask_ratio, contiguous):
        return g, g

    def generate_views(graph, structural_ops=None, geometric_ops=None):
        return [graph]


from data.mdataset import GraphData, GraphDataset
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.ddp import DistributedSamplerList, cleanup, init_distributed, is_main_process
from utils.graph_ops import _encode_graph, _pool_graph_emb
from utils.logging import maybe_init_wandb
from utils.schedule import cosine_with_warmup


@dataclass
class GraphBatch:
    """Simple container that mimics a PyG ``Batch`` for ``GraphData`` objects."""

    graphs: List[GraphData]
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None

    def __iter__(self) -> Iterable[GraphData]:
        return iter(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> GraphData:
        return self.graphs[idx]

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    def to(self, device: torch.device | str) -> "GraphBatch":
        device = torch.device(device)
        edge_attr = (
            self.edge_attr.to(device)
            if isinstance(self.edge_attr, torch.Tensor)
            else self.edge_attr
        )
        pos = self.pos.to(device) if isinstance(self.pos, torch.Tensor) else self.pos
        return GraphBatch(
            graphs=self.graphs,
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            ptr=self.ptr.to(device),
            edge_attr=edge_attr,
            pos=pos,
        )


def _collate_graph_batch(batch: Sequence[GraphData] | GraphBatch) -> GraphBatch:
    """Collate a sequence of ``GraphData`` objects into a ``GraphBatch``."""

    if isinstance(batch, GraphBatch):
        return batch

    graphs = list(batch)
    if not graphs:
        zero = torch.zeros(0, dtype=torch.long)
        zero_mat = torch.zeros((0, 0), dtype=torch.float32)
        zero_edge = torch.zeros((2, 0), dtype=torch.long)
        ptr = torch.zeros(1, dtype=torch.long)
        return GraphBatch([], zero_mat, zero_edge, zero, ptr)

    x_blocks: List[torch.Tensor] = []
    batch_index: List[torch.Tensor] = []
    edge_blocks: List[torch.Tensor] = []
    ptr: List[int] = [0]
    node_offset = 0

    all_have_pos = True
    edge_attr_dim = 0
    has_edge_attr = False
    edge_attr_blocks: List[Optional[torch.Tensor]] = []
    edge_counts: List[int] = []

    for idx, g in enumerate(graphs):
        x = torch.as_tensor(getattr(g, "x"), dtype=torch.float32)
        if x.dim() != 2:
            raise ValueError("GraphData.x must be a 2D array/tensor")
        x_blocks.append(x)
        n_i = int(x.size(0))
        batch_index.append(torch.full((n_i,), idx, dtype=torch.long))
        ptr.append(ptr[-1] + n_i)

        pos = getattr(g, "pos", None)
        if pos is None:
            all_have_pos = False

        edge_index = getattr(g, "edge_index", None)
        if edge_index is None:
            ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            ei = torch.as_tensor(edge_index, dtype=torch.long)
            if ei.dim() == 2 and ei.size(0) != 2 and ei.size(1) == 2:
                ei = ei.t()
            if ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError("edge_index must have shape [2, E]")
        edge_blocks.append(ei + node_offset if ei.numel() else ei)
        edge_count = int(ei.size(1))
        edge_counts.append(edge_count)

        edge_attr = getattr(g, "edge_attr", None)
        if edge_attr is not None:
            ea = torch.as_tensor(edge_attr, dtype=torch.float32)
            if ea.dim() == 1:
                ea = ea.unsqueeze(-1)
            edge_attr_dim = max(edge_attr_dim, int(ea.size(-1)))
            has_edge_attr = True
            edge_attr_blocks.append(ea)
        else:
            edge_attr_blocks.append(None)

        node_offset += n_i

    x_cat = torch.cat(x_blocks, dim=0)
    batch_vec = torch.cat(batch_index, dim=0)
    ptr_tensor = torch.tensor(ptr, dtype=torch.long)
    edge_index_cat = (
        torch.cat(edge_blocks, dim=1)
        if edge_blocks
        else torch.zeros((2, 0), dtype=torch.long)
    )

    batch_edge_attr = None
    if has_edge_attr:
        filled: List[torch.Tensor] = []
        for ea, edge_count in zip(edge_attr_blocks, edge_counts):
            if ea is None:
                filled.append(
                    torch.zeros((edge_count, edge_attr_dim), dtype=torch.float32)
                )
            else:
                if ea.size(-1) != edge_attr_dim:
                    pad_dim = edge_attr_dim - ea.size(-1)
                    if pad_dim > 0:
                        pad = torch.zeros((ea.size(0), pad_dim), dtype=ea.dtype)
                        ea = torch.cat([ea, pad], dim=-1)
                    else:
                        ea = ea[:, :edge_attr_dim]
                filled.append(ea)
        batch_edge_attr = (
            torch.cat(filled, dim=0)
            if filled
            else torch.zeros((0, edge_attr_dim), dtype=torch.float32)
        )

    batch_pos = None
    if all_have_pos:
        pos_blocks = [torch.as_tensor(g.pos, dtype=torch.float32) for g in graphs]
        batch_pos = torch.cat(pos_blocks, dim=0) if pos_blocks else None

    return GraphBatch(
        graphs=graphs,
        x=x_cat,
        edge_index=edge_index_cat,
        batch=batch_vec,
        ptr=ptr_tensor,
        edge_attr=batch_edge_attr,
        pos=batch_pos,
    )


def _build_graph_dataloader(
    data_source: Sequence[GraphData],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    """Construct a ``DataLoader`` that yields ``GraphBatch`` batches."""

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_graph_batch,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(data_source, **loader_kwargs)


def _graph_to_tensors(
    g: GraphData, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert GraphData -> (x, adj) tensors on `device`.
    x: [N, F] float32; adj: [N, N] float32, symmetric 0/1.
    """
    x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
    n = int(x.size(0))
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)
    if g.edge_index is not None:
        ei = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        # normalize to [2, E]
        if ei.dim() == 2 and ei.size(0) != 2 and ei.size(1) == 2:
            ei = ei.t()
        if ei.numel() > 0:
            adj[ei[0], ei[1]] = 1.0
            adj[ei[1], ei[0]] = 1.0
    return x, adj


def _encode(encoder: nn.Module, g: GraphData, device: torch.device) -> torch.Tensor:
    """
    Call `encoder` in a signature-agnostic way:
      1) try encoder(g)
      2) fallback to encoder(x, adj) if required
    """
    try:
        return encoder(g)
    except TypeError:
        x, adj = _graph_to_tensors(g, device)
        return encoder(x, adj)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 2 else x.unsqueeze(0)




def train_jepa(
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
    *,
    max_batches: int = 0,
    time_budget_mins: int = 0,
    disable_tqdm: bool = False,
    num_workers=0, 
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=4, 
    bf16=False, 
    **unused
) -> List[float]:
    ddp_backend = os.getenv("DDP_BACKEND")  # optional override
    distributed = (devices > 1) and init_distributed(ddp_backend)
    device_t = torch.device(device)
    if distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder.to(device_t),
            device_ids=[torch.cuda.current_device()]
            if device_t.type == "cuda"
            else None,
        )
        predictor = nn.parallel.DistributedDataParallel(
            predictor.to(device_t),
            device_ids=[torch.cuda.current_device()]
            if device_t.type == "cuda"
            else None,
        )
        ema_encoder = ema_encoder.to(device_t).eval()
        encoder.train()
        predictor.train()
    else:
        encoder.to(device_t).train()
        ema_encoder.to(device_t).eval()
        predictor.to(device_t).train()

    # --- Torch compatibility shim: some Torch builds (e.g., 1.13)
    # don't expose ``torch._dynamo``
    import torch as _torch

    if not hasattr(_torch, "_dynamo"):

        class _DummyDynamo:
            def disable(self, fn=None, recursive=False):
                # Behave like a no-op decorator or direct passthrough
                if fn is None:

                    def _decorator(f):
                        return f

                    return _decorator
                return fn

        _torch._dynamo = _DummyDynamo()

    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    # AMP setup (bf16 on 4090). GradScaler is only for fp16, not bf16.
    amp_enabled = (device_t.type == "cuda") and (use_amp or bf16)
    _amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and (not bf16) and device_t.type == "cuda"))


    steps_per_epoch = max(1, math.ceil(len(dataset.graphs) / batch_size))
    total_steps = epochs * steps_per_epoch
    sch = cosine_with_warmup(opt, warmup_steps, total_steps) if use_scheduler else None
    # EMA momentum schedule: start from current decay, finish near 1.0 for a stable target
    ema_start = float(getattr(ema, "decay", 0.996))
    ema_end   = 0.9999
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


    _start_wall = _t.perf_counter()

    def _time_left() -> bool:
        return (time_budget_mins <= 0) or (
            (_t.perf_counter() - _start_wall) < time_budget_mins * 60
        )

    # Determine whether to disable the progress bar.  We disable bars either
    # explicitly via disable_tqdm or whenever stdout isn’t a TTY (e.g. tests).
    import sys

    disable_bar = disable_tqdm or not sys.stdout.isatty()
    pbar = (
        tqdm.tqdm(
            total=epochs * steps_per_epoch,
            desc=f"Epoch {start_epoch}/{epochs}",
            leave=False,
            disable=disable_bar,
        )
        if is_main_process() and not disable_bar
        else None
    )
    for ep in range(start_epoch, epochs + 1):
        if not _time_left():
            if is_main_process():
                wb and wb.log({"early/stop_reason": "time_budget"})
                tqdm.tqdm.write(
                    "Time budget exhausted before next JEPA epoch; stopping."
                )
            break
        # Update the bar description when the epoch changes
        if pbar is not None:
            pbar.set_description(f"Epoch {ep}/{epochs}")

        ep_loss = 0.0

        data_source = (
            list(DistributedSamplerList(dataset.graphs))
            if distributed
            else dataset.graphs
        )
        dataloader = _build_graph_dataloader(
            data_source,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        batches_done = 0
        # Iterate over batches and update the outer progress bar each time
        for batch in dataloader:
            if max_batches > 0 and batches_done >= max_batches:
                break
            if not _time_left():
                if is_main_process():
                    tqdm.tqdm.write(
                        "Time budget exhausted during JEPA epoch; breaking."
                    )
                break
            if not batch:
                continue

            graphs_in_batch = (
                list(batch.graphs)
                if isinstance(batch, GraphBatch)
                else list(batch)
            )
            if not graphs_in_batch:
                continue

            ctx_graphs, tgt_graphs = [], []
            for g in graphs_in_batch:
                views = generate_views(
                    g,
                    structural_ops=[
                        lambda x, mr=mask_ratio, c=contiguous: mask_subgraph(x, mr, c)
                    ],
                )
                # Be defensive: some generators/stubs may yield a single view.
                if not isinstance(views, (list, tuple)):
                    views = [views]
                if len(views) >= 2:
                    g_ctx, g_tgt = views[0], views[1]
                else:
                    # Fallback: duplicate the graph for context,
                    # synthesize a simple, deterministic target so MSE ≈ 1.0
                    g_ctx = views[0] if views else g

                    # Build g_tgt with x shifted by +1 (same edges/edge_attr)
                    import numpy as _np
                    from data.mdataset import GraphData  # adjust import if GraphData is elsewhere

                    x = _np.asarray(g_ctx.x, dtype=_np.float32)
                    x_tgt = (x + 1.0).copy()

                    edge_attr = getattr(g_ctx, "edge_attr", None)
                    g_tgt = GraphData(
                        x=x_tgt,
                        edge_index=g_ctx.edge_index.copy(),
                        edge_attr=(None if edge_attr is None else edge_attr.copy()),
                    )

                ctx_graphs.append(g_ctx)
                tgt_graphs.append(g_tgt)

            ctx_batch = _collate_graph_batch(ctx_graphs).to(device_t)
            tgt_batch = _collate_graph_batch(tgt_graphs).to(device_t)

            with torch.cuda.amp.autocast(
                enabled=amp_enabled,
                dtype=_amp_dtype,
            ):
                h_c_nodes = _ensure_2d(_encode_graph(encoder, ctx_batch))
                with torch.no_grad():
                    h_t_nodes = _ensure_2d(_encode_graph(ema_encoder, tgt_batch))

            h_c_g = _ensure_2d(_pool_graph_emb(h_c_nodes, ctx_batch))
            h_t_g = _ensure_2d(_pool_graph_emb(h_t_nodes, tgt_batch))

            with torch.cuda.amp.autocast(
                enabled=amp_enabled,
                dtype=_amp_dtype,
            ):
                pred = predictor(h_c_g)

                if pred.dim() == 1 and h_t_g.dim() == 2 and h_t_g.size(0) == 1:
                    pred = pred.unsqueeze(0)
                if h_t_g.dim() == 1 and pred.dim() == 2 and pred.size(0) == 1:
                    h_t_g = h_t_g.unsqueeze(0)

                l2_reg = sum((p**2).sum() for p in predictor.parameters())
                loss = mse(pred, h_t_g) + reg_lambda * l2_reg

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

            _enc = encoder.module if isinstance(encoder, nn.parallel.DistributedDataParallel) else encoder
            ema.update(_enc)

            lv = float(loss.detach().cpu().item())
            ep_loss += lv
            # ---- Cosine EMA momentum ramp: ema.decay = ema_start → ema_end over total_steps ----
            step += 1
            if hasattr(ema, "set_decay"):
                # alpha ∈ [0,1]
                alpha = min(1.0, step / float(total_steps)) if total_steps > 0 else 1.0
                # smooth 0→1
                w = 0.5 * (1.0 - math.cos(math.pi * alpha))
                ema_now = ema_start + (ema_end - ema_start) * w
                ema.set_decay(ema_now)
            # Update target with the scheduled momentum
            _enc = encoder.module if isinstance(encoder, nn.parallel.DistributedDataParallel) else encoder
            ema.update(_enc)
            if wb and is_main_process():
                wb.log(
                    {
                        "train/jepa_loss": lv,
                        "lr": float(opt.param_groups[0]["lr"]),
                        "step": step,
                        "epoch": ep,
                    }
                )
            batches_done += 1
            # Update our single progress bar after processing each batch
            if pbar is not None:
                pbar.update(1)

        ep_loss /= max(1, min(steps_per_epoch, batches_done))
        if is_main_process():
            losses.append(ep_loss)
            if wb:
                wb.log({"epoch/jepa_loss": ep_loss, "epoch": ep})
            if ckpt_path and (ep % ckpt_every == 0 or ep == epochs):
                save_checkpoint(
                    os.path.join(ckpt_path, f"jepa_ep{ep:04d}.pt"),
                    encoder=(encoder.module.state_dict() if isinstance(encoder, nn.parallel.DistributedDataParallel) else encoder.state_dict()),
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
    # Close the progress bar after training
    if pbar is not None:
        pbar.close()
    return losses


def train_contrastive(
    dataset: GraphDataset,
    encoder: nn.Module,
    projection_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 64,
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
    # gemoetric augmentations (contrastive-only)
    mask_angle: bool = False,
    perturb_dihedral: bool = False,
    resume_from: Optional[str] = None,
    # structural augmentations (contrastive-only)
    bond_deletion: bool = False,
    atom_masking: bool = False,
    subgraph_removal: bool = False,
    *,
    max_batches: int = 0,
    time_budget_mins: int = 0,
    disable_tqdm: bool = False,
    num_workers=0, 
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=4, 
    bf16=False, 
    **unused
) -> List[float]:
    ddp_backend = os.getenv("DDP_BACKEND")  # optional override
    distributed = (devices > 1) and init_distributed(ddp_backend)
    device_t = torch.device(device)
    if distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder.to(device_t),
            device_ids=[torch.cuda.current_device()]
            if device_t.type == "cuda"
            else None,
        )
        proj = nn.Sequential(
            nn.Linear(256, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        ).to(device_t)
        if device_t.type == "cuda":
            from torch.nn.parallel import DistributedDataParallel as DDP
            proj = DDP(proj, device_ids=[torch.cuda.current_device()])
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

    # GradScaler is for fp16; disable when using bf16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and (not bf16) and device_t.type == "cuda"))
    # AMP setup (bf16 on 4090)
    amp_enabled = (device_t.type == "cuda") and (use_amp or bf16)
    _amp_dtype = torch.bfloat16 if bf16 else torch.float16

    if len(dataset.graphs) < 2 or batch_size < 2:
        raise ValueError(
            "Contrastive training requires at least two graphs per batch"
    )


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

    _start_wall = _t.perf_counter()

    def _time_left() -> bool:
        return (time_budget_mins <= 0) or ((_t.perf_counter() - _start_wall) < time_budget_mins * 60)

    # Determine whether to disable the progress bar.  Disable when disable_tqdm
    # is set or stdout isn’t a TTY (tests).
    import sys

    start_epoch = 1
    disable_bar = disable_tqdm or not sys.stdout.isatty()
    pbar = (
        tqdm.tqdm(
            total=epochs * steps_per_epoch,
            desc=f"Epoch {start_epoch}/{epochs}",
            leave=False,
            disable=disable_bar,
        )
        if is_main_process() and not disable_bar
        else None
    )

    for ep in range(start_epoch, epochs + 1):
        ep_loss = 0.0

        # Update the bar description when the epoch changes
        if pbar is not None:
            pbar.set_description(f"Epoch {ep}/{epochs}")

        data_source = (
            list(DistributedSamplerList(dataset.graphs))
            if distributed
            else dataset.graphs
        )
        dataloader = _build_graph_dataloader(
            data_source,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        batches_done = 0
        for batch in dataloader:
            if max_batches > 0 and batches_done >= max_batches:
                break
            if not _time_left():
                if is_main_process():
                    tqdm.tqdm.write(
                        "Time budget exhausted during contrastive epoch; breaking."
                    )
                break
            if not batch:
                continue

            z1_list, z2_list = [], []
            for g in batch:
                struct_ops = [] 
                if bond_deletion:
                    struct_ops.append(delete_random_bond)
                if atom_masking:
                    struct_ops.append(mask_random_atom)
                if subgraph_removal:
                    struct_ops.append(remove_random_subgraph)
                # keep subgraph masking as a function of mask_ratio (contrastive view)
                if mask_ratio and mask_ratio > 0:
                    struct_ops.append(
                        lambda x, mr=mask_ratio: mask_subgraph(x, mr, contiguous=False)[0]
                    )
               
                geom_ops = []
                if random_rotate or mask_angle or perturb_dihedral:
                    geom_ops.append(
                        lambda x: apply_graph_augmentations(
                            x,
                            rotate=random_rotate,
                            mask_angle=mask_angle,
                            perturb_dihedral=perturb_dihedral,
                        )
                    )
                v1 = generate_views(g, structural_ops=struct_ops, geometric_ops=geom_ops)[0]
                v2 = generate_views(g, structural_ops=struct_ops, geometric_ops=geom_ops)[0]
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=_amp_dtype):

                    # node features for each view
                    # move graphs to the model's device if they support .to(), then encode
                    g1 = v1.to(device_t) if hasattr(v1, "to") else v1
                    g2 = v2.to(device_t) if hasattr(v2, "to") else v2
                    h1_nodes = _ensure_2d(_encode_graph(encoder, g1))   # [N1, D]
                    h2_nodes = _ensure_2d(_encode_graph(encoder, g2))   # [N2, D]

                    # pool to **graph-level** embeddings (one vector per graph)
                    g1_pool = _ensure_2d(_pool_graph_emb(h1_nodes, v1))                 # [1, D]
                    g2_pool = _ensure_2d(_pool_graph_emb(h2_nodes, v2))                 # [1, D]

                    # projector expects the pooled feature dim
                    if isinstance(proj[0], nn.Linear) and proj[0].in_features != g1_pool.size(1):
                        proj[0] = nn.Linear(g1_pool.size(1), proj[0].out_features).to(device_t)
                        opt = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=lr)
                    z1_list.append(proj(g1_pool))                                      # [1, P]
                    z2_list.append(proj(g2_pool))                                      # [1, P]

            # stack to [B, P] and normalize
            z1 = torch.nan_to_num(F.normalize(torch.cat(z1_list, dim=0), dim=-1))
            z2 = torch.nan_to_num(F.normalize(torch.cat(z2_list, dim=0), dim=-1))


            # --- new safety + symmetric loss logic ---
            N = z1.size(0)
            if N < 2:
                raise ValueError(
                    "Contrastive loss requires at least two graphs per batch"
                )

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=_amp_dtype):
                logits_12 = (z1 @ z2.t()) / float(temperature)   # [N, N]
                with torch.no_grad():
                    pos = logits_12.diag().mean().item()
                    neg = ((logits_12.sum() - logits_12.diag().sum()) / (N*(N-1))).item()
                    wb and wb.log({"diag/pos_minus_neg": pos - neg, "diag/lnN": math.log(N)}, commit=False)
                logits_21 = logits_12.t()

                # NOTE: Do NOT mask the diagonal here.
                # In the N×N (z1 @ z2.T) setup, the diagonal entries are the positives.

                labels = torch.arange(N, device=device_t)

                loss = 0.5 * (
                    F.cross_entropy(logits_12, labels) +
                    F.cross_entropy(logits_21, labels)
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
            batches_done += 1
            # Update our single progress bar after processing each batch
            if pbar is not None:
                pbar.update(1)
        ep_loss /= max(1, min(steps_per_epoch, batches_done))
        if is_main_process():
            losses.append(ep_loss)
            if wb:
                wb.log({"epoch/contrastive_loss": ep_loss, "epoch": ep})
            if ckpt_path and (ep % ckpt_every == 0 or ep == epochs):
                save_checkpoint(
                    os.path.join(ckpt_path, f"contrastive_ep{ep:04d}.pt"),
                    encoder=(encoder.module.state_dict() if isinstance(encoder, nn.parallel.DistributedDataParallel) else encoder.state_dict()),
                    projector=proj.state_dict(),
                    optimizer=opt.state_dict(),
                    scaler=(
                        scaler.state_dict()
                        if isinstance(scaler, torch.cuda.amp.GradScaler)
                        else None
                    ),
                    epoch=ep,
                )
    if pbar is not None:
        pbar.close()
    # Close the progress bar after training
    try:
        if wb and is_main_process():
            wb.finish()
    except Exception:
        pass
    if distributed:
        cleanup()

    return losses
