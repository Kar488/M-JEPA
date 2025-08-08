
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
from typing import Callable, Iterable, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Models & training
try:
    from models.factory import build_encoder
except Exception:
    from models.encoder import GNNEncoder as _BasicEnc
    def build_encoder(gnn_type: str, input_dim: int, hidden_dim: int, num_layers: int, edge_dim: int | None = None):
        return _BasicEnc(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type)

from models.predictor import MLPPredictor
from models.ema import EMA
from training.unsupervised import train_jepa, train_contrastive
from training.supervised import train_linear_head
from training.baselines import pretrain_baseline


@dataclass(frozen=True)
class Config:
    mask_ratio: float
    contiguous: bool
    hidden_dim: int
    num_layers: int
    gnn_type: str
    ema_decay: float
    add_3d: bool
    pretrain_bs: int
    finetune_bs: int
    pretrain_epochs: int
    finetune_epochs: int
    lr: float


def _build_configs(
    mask_ratios: Iterable[float],
    contiguities: Iterable[bool],
    hidden_dims: Iterable[int],
    num_layers_list: Iterable[int],
    gnn_types: Iterable[str],
    ema_decays: Iterable[float],
    add_3d_options: Iterable[bool],
    pretrain_batch_sizes: Iterable[int],
    finetune_batch_sizes: Iterable[int],
    pretrain_epochs_options: Iterable[int],
    finetune_epochs_options: Iterable[int],
    lrs: Iterable[float],
) -> List[Config]:
    combos = product(
        mask_ratios, contiguities, hidden_dims, num_layers_list, gnn_types,
        ema_decays, add_3d_options, pretrain_batch_sizes, finetune_batch_sizes,
        pretrain_epochs_options, finetune_epochs_options, lrs
    )
    return [Config(*tpl) for tpl in combos]


def _edge_dim_or_none(ds) -> int | None:
    g0 = ds.graphs[0]
    return None if g0.edge_attr is None else int(g0.edge_attr.shape[1])


def _run_one_config_method(
    cfg: Config,
    method: str,
    dataset_fn: Callable[[bool], Any],
    task_type: str,
    seeds: Iterable[int],
    device: str,
    use_wandb: bool,
    ckpt_dir: str,
    ckpt_every: int,
    use_scheduler: bool,
    warmup_steps: int,
) -> Dict[str, Any]:
    ds = dataset_fn(cfg.add_3d)
    input_dim = int(ds.graphs[0].x.shape[1])
    edge_dim = _edge_dim_or_none(ds)

    seed_metrics = []
    for seed in seeds:
        np.random.seed(seed)

        if method.lower() == "jepa":
            encoder = build_encoder(gnn_type=cfg.gnn_type, input_dim=input_dim,
                                    hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, edge_dim=edge_dim)
            ema_encoder = build_encoder(gnn_type=cfg.gnn_type, input_dim=input_dim,
                                        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, edge_dim=edge_dim)
            ema = EMA(encoder, decay=cfg.ema_decay)
            predictor = MLPPredictor(embed_dim=cfg.hidden_dim, hidden_dim=cfg.hidden_dim * 2)

            try:
                train_jepa(
                    dataset=ds, encoder=encoder, ema_encoder=ema_encoder, predictor=predictor, ema=ema,
                    epochs=cfg.pretrain_epochs, batch_size=cfg.pretrain_bs, mask_ratio=cfg.mask_ratio,
                    contiguous=cfg.contiguous, lr=cfg.lr, device=device, reg_lambda=1e-4,
                    use_wandb=use_wandb, ckpt_path=f"{ckpt_dir}/jepa", ckpt_every=ckpt_every,
                    use_scheduler=use_scheduler, warmup_steps=warmup_steps
                )
            except TypeError:
                train_jepa(
                    dataset=ds, encoder=encoder, ema_encoder=ema_encoder, predictor=predictor, ema=ema,
                    epochs=cfg.pretrain_epochs, batch_size=cfg.pretrain_bs, mask_ratio=cfg.mask_ratio,
                    contiguous=cfg.contiguous, lr=cfg.lr, device=device, reg_lambda=1e-4
                )

        elif method.lower() == "contrastive":
            encoder = build_encoder(gnn_type=cfg.gnn_type, input_dim=input_dim,
                                    hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, edge_dim=edge_dim)
            try:
                train_contrastive(
                    dataset=ds, encoder=encoder, projection_dim=64, epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs, mask_ratio=cfg.mask_ratio, lr=cfg.lr, device=device,
                    temperature=0.1, use_wandb=use_wandb, ckpt_path=f"{ckpt_dir}/contrast", ckpt_every=ckpt_every,
                    use_scheduler=use_scheduler, warmup_steps=warmup_steps
                )
            except TypeError:
                train_contrastive(
                    dataset=ds, encoder=encoder, projection_dim=64, epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs, mask_ratio=cfg.mask_ratio, lr=cfg.lr, device=device,
                    temperature=0.1
                )

        else:
            # external baselines; wrapper falls back to internal contrastive if unavailable
            res = pretrain_baseline(method, dataset=ds, input_dim=input_dim, device=device, cfg=dict(
                hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, gnn_type=cfg.gnn_type,
                epochs=cfg.pretrain_epochs, batch_size=cfg.pretrain_bs, mask_ratio=cfg.mask_ratio, lr=cfg.lr,
                use_wandb=use_wandb, ckpt_path=f"{ckpt_dir}/{method}", ckpt_every=ckpt_every
            ))
            encoder = res.get("encoder", build_encoder(gnn_type=cfg.gnn_type, input_dim=input_dim,
                                                       hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, edge_dim=edge_dim))

        # Finetune linear head
        metrics = train_linear_head(dataset=ds, encoder=encoder, task_type=task_type,
                                    epochs=cfg.finetune_epochs, lr=5e-3, batch_size=cfg.finetune_bs, device=device)
        metrics = {k: float(v) for k, v in metrics.items() if k != "head"}
        seed_metrics.append(metrics)

    agg = {k: float(np.mean([m[k] for m in seed_metrics])) for k in seed_metrics[0].keys()}
    row = {**asdict(cfg), **agg, "method": method}
    return row


def run_grid_search(
    *,
    dataset_fn: Callable[[bool], Any],
    methods: Tuple[str, ...] = ("jepa",),
    task_type: str = "classification",
    seeds: Tuple[int, ...] = (42, 123, 456),
    mask_ratios: Tuple[float, ...] = (0.10, 0.15, 0.25),
    contiguities: Tuple[bool, ...] = (False, True),
    hidden_dims: Tuple[int, ...] = (128, 256),
    num_layers_list: Tuple[int, ...] = (2, 3),
    gnn_types: Tuple[str, ...] = ("mpnn", "gcn", "gat", "edge_mpnn"),
    ema_decays: Tuple[float, ...] = (0.95, 0.99),
    add_3d_options: Tuple[bool, ...] = (False, True),
    pretrain_batch_sizes: Tuple[int, ...] = (256,),
    finetune_batch_sizes: Tuple[int, ...] = (64,),
    pretrain_epochs_options: Tuple[int, ...] = (50,),
    finetune_epochs_options: Tuple[int, ...] = (30,),
    lrs: Tuple[float, ...] = (1e-4,),
    device: str = "cuda",
    n_jobs: int = 0,
    use_wandb: bool = False,
    ckpt_dir: str = "outputs/grid_ckpts",
    ckpt_every: int = 25,
    use_scheduler: bool = True,
    warmup_steps: int = 1000,
) -> pd.DataFrame:
    cfgs = _build_configs(
        mask_ratios, contiguities, hidden_dims, num_layers_list, gnn_types,
        ema_decays, add_3d_options, pretrain_batch_sizes, finetune_batch_sizes,
        pretrain_epochs_options, finetune_epochs_options, lrs
    )

    rows: List[Dict[str, Any]] = []
    if n_jobs and n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            rows = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run_one_config_method)(cfg, method, dataset_fn, task_type, seeds, device,
                                                use_wandb, ckpt_dir, ckpt_every, use_scheduler, warmup_steps)
                for cfg in cfgs for method in methods
            )
        except Exception:
            for cfg in cfgs:
                for method in methods:
                    rows.append(_run_one_config_method(cfg, method, dataset_fn, task_type, seeds, device,
                                                       use_wandb, ckpt_dir, ckpt_every, use_scheduler, warmup_steps))
    else:
        for cfg in cfgs:
            for method in methods:
                rows.append(_run_one_config_method(cfg, method, dataset_fn, task_type, seeds, device,
                                                   use_wandb, ckpt_dir, ckpt_every, use_scheduler, warmup_steps))

    return pd.DataFrame(rows)
