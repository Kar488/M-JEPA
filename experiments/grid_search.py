from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd 

import torch
from torch_geometric.loader import DataLoader as GeoLoader

# Encoder factory
try:
    from models.factory import build_encoder
except Exception:
    from models.encoder import GNNEncoder as _BasicEnc

    def build_encoder(
        gnn_type: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        edge_dim: Optional[int] = None,
    ):
        return _BasicEnc(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
        )


# Baselines (CLI or native adapters)
from experiments.baseline_integration import baseline_pretrain_and_embed
from models.ema import EMA
from models.predictor import MLPPredictor
from training.supervised import train_linear_head
from training.train_on_embeddings import (
    train_linear_on_embeddings_classification,
    train_linear_on_embeddings_regression,
)
from training.unsupervised import train_contrastive, train_jepa

# Optional probing & clustering
try:
    from experiments.probing import (
        clustering_quality,
        compute_embeddings,
        linear_probe_classification,
        linear_probe_regression,
    )

    _HAS_PROBE = True
except Exception:
    _HAS_PROBE = False


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
        mask_ratios,
        contiguities,
        hidden_dims,
        num_layers_list,
        gnn_types,
        ema_decays,
        add_3d_options,
        pretrain_batch_sizes,
        finetune_batch_sizes,
        pretrain_epochs_options,
        finetune_epochs_options,
        lrs,
    )
    return [Config(*tpl) for tpl in combos]


def _edge_dim_or_none(ds) -> Optional[int]:
    g0 = ds.graphs[0]
    return None if g0.edge_attr is None else int(g0.edge_attr.shape[1])


def _aggregate_seed_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    # union of keys (probing/cluster may be absent for some seeds)
    keys = sorted({k for m in metrics_list for k in m.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list if k in m], dtype=np.float64)
        if len(vals) == 0:
            continue
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"{k}_ci95"] = (
            float(1.96 * (vals.std(ddof=1) / max(1, np.sqrt(len(vals)))))
            if len(vals) > 1
            else 0.0
        )
    return out

def _ensure_loader(obj: Any, batch_size: int, shuffle: bool) -> Optional[GeoLoader]:
    if obj is None:
        return None
    # Already a loader (iterable but not indexable)
    if hasattr(obj, "__iter__") and not hasattr(obj, "__getitem__"):
        return obj  # assume it's a loader
    # Likely a dataset: wrap it
    return GeoLoader(obj, batch_size=batch_size, shuffle=shuffle)

def _normalize_ds_to_loaders(ds: Any, pre_bs: int, ft_bs: int) -> Tuple[Any, Any, Any]:
    tr, va, te = _normalize_ds(ds)
    tr = _ensure_loader(tr, pre_bs, True)
    va = _ensure_loader(va, ft_bs, False)
    te = _ensure_loader(te, ft_bs, False)
    return tr, va, te

def _infer_dims_from_loader(obj: Any) -> Tuple[Optional[int], Optional[int]]:
    if obj is None:
        return None, None
    # If loader: get first batch; if dataset: get first sample
    batch = None
    if hasattr(obj, "__iter__") and not hasattr(obj, "__getitem__"):  # loader
        it = iter(obj)
        try:
            batch = next(it)
        except StopIteration:
            return None, None
    else:  # dataset or single Data
        try:
            batch = obj[0]
        except Exception:
            batch = obj
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    x = getattr(batch, "x", None)
    edge_attr = getattr(batch, "edge_attr", None)
    in_dim = int(x.size(-1)) if x is not None else None
    edge_dim = int(edge_attr.size(-1)) if edge_attr is not None else None
    return in_dim, edge_dim

def _edge_dim_or_none(ds_pre: Any) -> Optional[int]:
    """Your existing util—kept for the non-prebuilt path. Implement as before."""
    try:
        ea = ds_pre.graphs[0].edge_attr
        return int(ea.shape[1]) if ea is not None else None
    except Exception:
        return None

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
    baseline_unlabeled_file: Optional[str],
    baseline_eval_file: Optional[str],
    baseline_smiles_col: str,
    baseline_label_col: Optional[str],
    baseline_cfg: str = "adapters/config.yaml",
    use_scaffold: bool = False,
    prebuilt_loaders: Optional[Tuple[Any, Any, Any]] = None,

) -> Dict[str, Any]:
    
    if prebuilt_loaders is not None:
        train_loader, val_loader, test_loader = prebuilt_loaders
        input_dim, edge_dim = _infer_dims_from_loader(train_loader)
    else:
        ds_pre = unlabeled_dataset_fn(cfg.add_3d)
        ds_eval = eval_dataset_fn(cfg.add_3d)
        input_dim = int(ds_pre.graphs[0].x.shape[1])
        edge_dim = _edge_dim_or_none(ds_pre)

    seed_metrics: List[Dict[str, float]] = []
    for seed in seeds:
        np.random.seed(seed)
        if method.lower() == "jepa":
            encoder = build_encoder(
                gnn_type=cfg.gnn_type,
                input_dim=input_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                edge_dim=edge_dim,
            )
            ema_encoder = build_encoder(
                gnn_type=cfg.gnn_type,
                input_dim=input_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                edge_dim=edge_dim,
            )
            ema = EMA(encoder, decay=cfg.ema_decay)
            predictor = MLPPredictor(
                embed_dim=cfg.hidden_dim, hidden_dim=cfg.hidden_dim * 2
            )

            try:
                train_jepa(
                    dataset=ds_pre,
                    encoder=encoder,
                    ema_encoder=ema_encoder,
                    predictor=predictor,
                    ema=ema,
                    epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs,
                    mask_ratio=cfg.mask_ratio,
                    contiguous=cfg.contiguous,
                    lr=cfg.lr,
                    device=device,
                    reg_lambda=1e-4,
                    use_wandb=use_wandb,
                    ckpt_path=f"{ckpt_dir}/jepa",
                    ckpt_every=ckpt_every,
                    use_scheduler=use_scheduler,
                    warmup_steps=warmup_steps,
                )
            except TypeError:
                train_jepa(
                    dataset=ds_pre,
                    encoder=encoder,
                    ema_encoder=ema_encoder,
                    predictor=predictor,
                    ema=ema,
                    epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs,
                    mask_ratio=cfg.mask_ratio,
                    contiguous=cfg.contiguous,
                    lr=cfg.lr,
                    device=device,
                    reg_lambda=1e-4,
                )

            m = train_linear_head(
                dataset=ds_eval,
                encoder=encoder,
                task_type=task_type,
                epochs=cfg.finetune_epochs,
                lr=5e-3,
                batch_size=cfg.finetune_bs,
                device=device,
                use_scaffold=use_scaffold,
            )
            row = {k: float(v) for k, v in m.items() if k != "head"}

            if _HAS_PROBE:
                X = compute_embeddings(
                    ds_eval, encoder, batch_size=cfg.finetune_bs, device=device
                )
                if task_type == "classification":
                    row.update(
                        linear_probe_classification(
                            X,
                            ds_eval.labels.astype(int),
                            smiles=ds_eval.smiles,
                            use_scaffold=use_scaffold,
                        )
                    )
                else:
                    row.update(
                        linear_probe_regression(
                            X,
                            ds_eval.labels.astype(float),
                            smiles=ds_eval.smiles,
                            use_scaffold=use_scaffold,
                        )
                    )
                row.update(clustering_quality(X, n_clusters=10))

            seed_metrics.append(row)

        elif method.lower() == "contrastive":
            encoder = build_encoder(
                gnn_type=cfg.gnn_type,
                input_dim=input_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                edge_dim=edge_dim,
            )
            try:
                train_contrastive(
                    dataset=ds_pre,
                    encoder=encoder,
                    projection_dim=64,
                    epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs,
                    mask_ratio=cfg.mask_ratio,
                    lr=cfg.lr,
                    device=device,
                    temperature=0.1,
                    use_wandb=use_wandb,
                    ckpt_path=f"{ckpt_dir}/contrast",
                    ckpt_every=ckpt_every,
                    use_scheduler=use_scheduler,
                    warmup_steps=warmup_steps,
                )
            except TypeError:
                train_contrastive(
                    dataset=ds_pre,
                    encoder=encoder,
                    projection_dim=64,
                    epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs,
                    mask_ratio=cfg.mask_ratio,
                    lr=cfg.lr,
                    device=device,
                    temperature=0.1,
                )

            m = train_linear_head(
                dataset=ds_eval,
                encoder=encoder,
                task_type=task_type,
                epochs=cfg.finetune_epochs,
                lr=5e-3,
                batch_size=cfg.finetune_bs,
                device=device,
                use_scaffold=use_scaffold,
            )
            row = {k: float(v) for k, v in m.items() if k != "head"}

            if _HAS_PROBE:
                X = compute_embeddings(
                    ds_eval, encoder, batch_size=cfg.finetune_bs, device=device
                )
                if task_type == "classification":
                    row.update(
                        linear_probe_classification(
                            X,
                            ds_eval.labels.astype(int),
                            smiles=ds_eval.smiles,
                            use_scaffold=use_scaffold,
                        )
                    )
                else:
                    row.update(
                        linear_probe_regression(
                            X,
                            ds_eval.labels.astype(float),
                            smiles=ds_eval.smiles,
                            use_scaffold=use_scaffold,
                        )
                    )
                row.update(clustering_quality(X, n_clusters=10))

            seed_metrics.append(row)

        else:  # MolCLR / GeomGCL / HiMol via adapters
            if (
                baseline_unlabeled_file is None
                or baseline_eval_file is None
                or baseline_label_col is None
            ):
                raise ValueError(
                    "Baselines require baseline_unlabeled_file, baseline_eval_file, and baseline_label_col."
                )
            _, emb_file = baseline_pretrain_and_embed(
                method=method,
                unlabeled_file=baseline_unlabeled_file,
                smiles_eval_file=baseline_eval_file,
                cfg_path=baseline_cfg,
            )
            if baseline_eval_file.endswith(".csv"):
                y = pd.read_csv(baseline_eval_file)[baseline_label_col].to_numpy()
            else:
                y = pd.read_parquet(baseline_eval_file)[baseline_label_col].to_numpy()
            X = (
                np.load(emb_file)
                if emb_file.endswith(".npy")
                else pd.read_csv(emb_file).to_numpy()
            )
            if task_type == "classification":
                m = train_linear_on_embeddings_classification(X, y)
            else:
                m = train_linear_on_embeddings_regression(X, y)
            row = {f"probe_{k}": float(v) for k, v in m.items()}
            if _HAS_PROBE:
                row.update(clustering_quality(X, n_clusters=10))
            seed_metrics.append(row)

    agg = _aggregate_seed_metrics(seed_metrics)
    row = {**asdict(cfg), **agg, "method": method, "seeds": len(list(seeds))}
    return row

def _normalize_ds(ds: Any) -> Tuple[Any, Any, Any]:
    if isinstance(ds, dict):
        return ds.get("train") or ds.get("train_loader"), \
               ds.get("val") or ds.get("valid") or ds.get("val_loader"), \
               ds.get("test") or ds.get("test_loader")
    if isinstance(ds, (list, tuple)):
        if len(ds) == 3: return ds[0], ds[1], ds[2]
        if len(ds) == 2: return ds[0], ds[1], None
        if len(ds) == 1: return ds[0], None, None
    return ds, None, None

def _cfg_get(cfg: Any, key: str, default=None):
    # supports both dict-like and attr-like configs
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def run_grid_search(
    *,
    dataset_fn: Optional[Callable[..., Any]] = None,
    unlabeled_dataset_fn: Optional[Callable[[bool], Any]] = None,
    eval_dataset_fn: Optional[Callable[[bool], Any]] = None,

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
    baseline_unlabeled_file: Optional[str] = None,
    baseline_eval_file: Optional[str] = None,
    baseline_smiles_col: str = "smiles",
    baseline_label_col: Optional[str] = None,
    baseline_cfg: str = "adapters/config.yaml",
    out_csv: Optional[str] = None,
    use_scaffold: bool = False,
) -> pd.DataFrame:
    cfgs = _build_configs(
        mask_ratios,
        contiguities,
        hidden_dims,
        num_layers_list,
        gnn_types,
        ema_decays,
        add_3d_options,
        pretrain_batch_sizes,
        finetune_batch_sizes,
        pretrain_epochs_options,
        finetune_epochs_options,
        lrs,
    )

    # ---------------- dataset wiring ----------------
    # We ONLY decide which path to use here; we DO NOT build datasets yet.
    use_single_builder = dataset_fn is not None
        
    rows: List[Dict[str, Any]] = []
    for cfg in cfgs:
        add_3d = _cfg_get(cfg, "add_3d", False)  # tests sweep over this

        # If tests provided dataset_fn, build loaders per-config
        prebuilt_loaders = None
        if use_single_builder:
            pre_bs = getattr(cfg, "pretrain_batch_size", 32)
            ft_bs  = getattr(cfg, "finetune_batch_size", 32)
            try:
                ds = dataset_fn(add_3d=add_3d)   # preferred
            except TypeError:
                ds = dataset_fn(add_3d)          # positional fallback
            train_loader, val_loader, test_loader = _normalize_ds_to_loaders(ds, pre_bs, ft_bs)
            prebuilt_loaders = (train_loader, val_loader, test_loader)

        for method in methods:
            rows.append(
                _run_one_config_method(
                    cfg,
                    method,
                    unlabeled_dataset_fn,
                    eval_dataset_fn,
                    task_type,
                    seeds,
                    device,
                    use_wandb,
                    ckpt_dir,
                    ckpt_every,
                    use_scheduler,
                    warmup_steps,
                    baseline_unlabeled_file,
                    baseline_eval_file,
                    baseline_smiles_col,
                    baseline_label_col,
                    baseline_cfg,
                    use_scaffold=use_scaffold,
                    prebuilt_loaders=prebuilt_loaders,
                )
            )

    df = pd.DataFrame(rows)
    df["best_metric"] = ""
    metrics_max = [
        "roc_auc_mean",
        "pr_auc_mean",
        "acc_mean",
        "probe_roc_auc_mean",
        "probe_pr_auc_mean",
        "probe_acc_mean",
        "r2_mean",
        "probe_r2_mean",
        "cluster_silhouette_mean",
    ]
    metrics_min = [
        "rmse_mean",
        "mae_mean",
        "probe_rmse_mean",
        "probe_mae_mean",
    ]
    best_rows: List[Dict[str, Any]] = []
    for m in metrics_max:
        if m in df.columns:
            idx = df[m].idxmax()
            row = df.loc[idx].to_dict()
            row["best_metric"] = m
            best_rows.append(row)
    for m in metrics_min:
        if m in df.columns:
            idx = df[m].idxmin()
            row = df.loc[idx].to_dict()
            row["best_metric"] = m
            best_rows.append(row)
    if best_rows:
        df = pd.concat([df, pd.DataFrame(best_rows)], ignore_index=True)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df
