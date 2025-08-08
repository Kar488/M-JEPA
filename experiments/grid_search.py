
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
from typing import Callable, Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# Encoder factory (edge-aware included if available)
try:
    from models.factory import build_encoder
except Exception:
    from models.encoder import GNNEncoder as _BasicEnc
    def build_encoder(gnn_type: str, input_dim: int, hidden_dim: int, num_layers: int, edge_dim: Optional[int] = None):
        return _BasicEnc(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type)

from models.predictor import MLPPredictor
from models.ema import EMA
from training.unsupervised import train_jepa, train_contrastive
from training.supervised import train_linear_head

# Baseline adapter (CLI + sklearn heads on embeddings)
from experiments.baseline_integration import baseline_pretrain_and_embed
from training.train_on_embeddings import train_linear_on_embeddings_classification, train_linear_on_embeddings_regression


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


def _edge_dim_or_none(ds) -> Optional[int]:
    g0 = ds.graphs[0]
    return None if g0.edge_attr is None else int(g0.edge_attr.shape[1])


def _aggregate_seed_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = metrics_list[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"{k}_ci95"] = float(1.96 * (vals.std(ddof=1) / max(1, np.sqrt(len(vals))))) if len(vals) > 1 else 0.0
    return out


def _run_one_config_method(
    cfg: Config,
    method: str,
    unlabeled_dataset_fn: Callable[[bool], Any],
    eval_dataset_fn: Callable[[bool], Any],
    task_type: str,
    seeds: Iterable[int],
    device: str,
    use_wandb: bool,
    ckpt_dir: str,
    ckpt_every: int,
    use_scheduler: bool,
    warmup_steps: int,
    # baseline paths (files)
    baseline_unlabeled_file: Optional[str],
    baseline_eval_file: Optional[str],
    baseline_smiles_col: str,
    baseline_label_col: Optional[str],
    baseline_cfg: str = "adapters/config.yaml",
) -> Dict[str, Any]:
    ds_pre = unlabeled_dataset_fn(cfg.add_3d)
    ds_eval = eval_dataset_fn(cfg.add_3d)
    input_dim = int(ds_pre.graphs[0].x.shape[1])
    edge_dim = _edge_dim_or_none(ds_pre)

    seed_metrics: List[Dict[str, float]] = []

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
                    dataset=ds_pre, encoder=encoder, ema_encoder=ema_encoder, predictor=predictor, ema=ema,
                    epochs=cfg.pretrain_epochs, batch_size=cfg.pretrain_bs, mask_ratio=cfg.mask_ratio,
                    contiguous=cfg.contiguous, lr=cfg.lr, device=device, reg_lambda=1e-4,
                    use_wandb=use_wandb, ckpt_path=f"{ckpt_dir}/jepa", ckpt_every=ckpt_every,
                    use_scheduler=use_scheduler, warmup_steps=warmup_steps
                )
            except TypeError:
                train_jepa(
                    dataset=ds_pre, encoder=encoder, ema_encoder=ema_encoder, predictor=predictor, ema=ema,
                    epochs=cfg.pretrain_epochs, batch_size=cfg.pretrain_bs, mask_ratio=cfg.mask_ratio,
                    contiguous=cfg.contiguous, lr=cfg.lr, device=device, reg_lambda=1e-4
                )

            # Fine‑tune linear head on eval dataset
            metrics = train_linear_head(dataset=ds_eval, encoder=encoder, task_type=task_type,
                                        epochs=cfg.finetune_epochs, lr=5e-3, batch_size=cfg.finetune_bs, device=device)
            metrics = {k: float(v) for k, v in metrics.items() if k != "head"}
            seed_metrics.append(metrics)

        elif method.lower() == "contrastive":
            encoder = build_encoder(gnn_type=cfg.gnn_type, input_dim=input_dim,
                                    hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, edge_dim=edge_dim)
            try:
                train_contrastive(
                    dataset=ds_pre, encoder=encoder, projection_dim=64, epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs, mask_ratio=cfg.mask_ratio, lr=cfg.lr, device=device,
                    temperature=0.1, use_wandb=use_wandb, ckpt_path=f"{ckpt_dir}/contrast", ckpt_every=ckpt_every,
                    use_scheduler=use_scheduler, warmup_steps=warmup_steps
                )
            except TypeError:
                train_contrastive(
                    dataset=ds_pre, encoder=encoder, projection_dim=64, epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs, mask_ratio=cfg.mask_ratio, lr=cfg.lr, device=device,
                    temperature=0.1
                )

            metrics = train_linear_head(dataset=ds_eval, encoder=encoder, task_type=task_type,
                                        epochs=cfg.finetune_epochs, lr=5e-3, batch_size=cfg.finetune_bs, device=device)
            metrics = {k: float(v) for k, v in metrics.items() if k != "head"}
            seed_metrics.append(metrics)

        else:  # Baselines (MolCLR/GeomGCL/HiMol) via CLI + sklearn heads
            if baseline_unlabeled_file is None or baseline_eval_file is None or baseline_label_col is None:
                raise ValueError("Baselines require baseline_unlabeled_file, baseline_eval_file, and baseline_label_col.")
            # Pretrain (once) and export embeddings
            _, emb_file = baseline_pretrain_and_embed(
                method=method, unlabeled_file=baseline_unlabeled_file,
                smiles_eval_file=baseline_eval_file, cfg_path=baseline_cfg
            )
            # Load labels and embeddings
            if baseline_eval_file.endswith(".csv"):
                import pandas as pd
                y = pd.read_csv(baseline_eval_file)[baseline_label_col].to_numpy()
            else:
                import pandas as pd
                y = pd.read_parquet(baseline_eval_file)[baseline_label_col].to_numpy()
            import numpy as np
            X = np.load(emb_file) if emb_file.endswith(".npy") else pd.read_csv(emb_file).to_numpy()

            if task_type == "classification":
                m = train_linear_on_embeddings_classification(X, y)
            else:
                m = train_linear_on_embeddings_regression(X, y)
            seed_metrics.append({k: float(v) for k, v in m.items()})

    agg = _aggregate_seed_metrics(seed_metrics)
    row = {**asdict(cfg), **agg, "method": method, "seeds": len(list(seeds))}
    return row


def run_grid_search(
    *,
    unlabeled_dataset_fn: Callable[[bool], Any],
    eval_dataset_fn: Callable[[bool], Any],
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
    # baseline files
    baseline_unlabeled_file: Optional[str] = None,
    baseline_eval_file: Optional[str] = None,
    baseline_smiles_col: str = "smiles",
    baseline_label_col: Optional[str] = None,
    baseline_cfg: str = "adapters/config.yaml",
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
                delayed(_run_one_config_method)(cfg, method, unlabeled_dataset_fn, eval_dataset_fn, task_type, seeds, device,
                                                use_wandb, ckpt_dir, ckpt_every, use_scheduler, warmup_steps,
                                                baseline_unlabeled_file, baseline_eval_file, baseline_smiles_col,
                                                baseline_label_col, baseline_cfg)
                for cfg in cfgs for method in methods
            )
        except Exception:
            for cfg in cfgs:
                for method in methods:
                    rows.append(_run_one_config_method(cfg, method, unlabeled_dataset_fn, eval_dataset_fn, task_type, seeds, device,
                                                       use_wandb, ckpt_dir, ckpt_every, use_scheduler, warmup_steps,
                                                       baseline_unlabeled_file, baseline_eval_file, baseline_smiles_col,
                                                       baseline_label_col, baseline_cfg))
    else:
        for cfg in cfgs:
            for method in methods:
                rows.append(_run_one_config_method(cfg, method, unlabeled_dataset_fn, eval_dataset_fn, task_type, seeds, device,
                                                   use_wandb, ckpt_dir, ckpt_every, use_scheduler, warmup_steps,
                                                   baseline_unlabeled_file, baseline_eval_file, baseline_smiles_col,
                                                   baseline_label_col, baseline_cfg))

    return pd.DataFrame(rows)
