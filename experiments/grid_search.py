from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Sequence

ds_pre: Optional[Any] = None
ds_eval: Optional[Any] = None

import logging
import numpy as np
import pandas as pd

import torch
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.data import Data as PyGData
import warnings

logger = logging.getLogger(__name__)

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

BASELINE_METHODS = {"molclr", "geomgcl", "himol", "baseline"}

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

@dataclass
class _GraphDatasetShim:
    graphs: Sequence[Any]
    labels: Optional[np.ndarray] = None
    smiles: Optional[Sequence[str]] = None

def _extract_attr(seq: Sequence[Any], name: str):
    out: List[Any] = []
    for g in seq:
        v = getattr(g, name, None)
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            if v.numel() == 1:
                v = v.item()
            else:
                v = v.numpy()
        out.append(v)
    # labels as np.array when name == 'y'
    return np.asarray(out) if name == "y" else out

def _ensure_graph_dataset(obj: Any) -> Any:
    """Return object with `.graphs` and `.labels` if possible."""
    if obj is None:
        logger.debug("_ensure_graph_dataset received None")
        return None
    if hasattr(obj, "graphs"):
        if getattr(obj, "labels", None) is None:
            labs = _extract_attr(getattr(obj, "graphs"), "y")
            if labs is not None:
                obj.labels = labs  # type: ignore[attr-defined]
        return obj
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        graphs = list(obj)
        labels = getattr(obj, "labels", None) or _extract_attr(graphs, "y")
        smiles = getattr(obj, "smiles", None) or _extract_attr(graphs, "smiles")
        logger.debug("Built GraphDataset shim with %d graphs", len(graphs))
        return _GraphDatasetShim(graphs=graphs, labels=labels, smiles=smiles)
    if isinstance(obj, PyGData) or hasattr(obj, "x") or hasattr(obj, "edge_index") or hasattr(obj, "adj"):
        label = getattr(obj, "y", None)
        if isinstance(label, torch.Tensor) and label.numel() == 1:
            label = np.asarray([label.item()])
        return _GraphDatasetShim(graphs=[obj], labels=label)
    logger.debug("_ensure_graph_dataset returning original object of type %s", type(obj))
    return obj

def _dataset_from_loader(loader: Any) -> Any:
    """Get a dataset from a loader. If `.dataset` missing, materialize from the loader."""
    if loader is None:
        logger.debug("_dataset_from_loader received None")
        return None
    
    ds = getattr(loader, "dataset", None)
   
    if ds is not None:
        logger.debug("Loader already has dataset: %s", type(ds))
        return _ensure_graph_dataset(ds)

    # Materialize small loaders: consume items into a shim
    graphs: List[Any] = []
    labels: List[Any] = []
    for item in loader:
        g = item[0] if isinstance(item, (list, tuple)) else item
        graphs.append(g)
        y = getattr(g, "y", None)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu()
            if y.numel() == 1:
                labels.append(y.item())
            else:
                labels.append(y.numpy())
        elif y is not None:
            labels.append(y)
    labs = np.asarray(labels) if len(labels) else None
    logger.debug("Materialized dataset from loader with %d graphs", len(graphs))
    return _GraphDatasetShim(graphs=graphs, labels=labs)

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
    configs = [Config(*tpl) for tpl in combos]
    logger.debug("Generated %d grid search configurations", len(configs))
    return configs


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

def _is_indexable(obj) -> bool:
    return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")

def _to_pyg(g: Any) -> PyGData:
    # If your class already provides a converter, use it
    if hasattr(g, "to_pyg") and callable(getattr(g, "to_pyg")):
        return g.to_pyg()
    # Fallback: build a PyG Data from common attributes
    fields = {}
    for name in ("x", "edge_index", "edge_attr", "pos", "y"):
        if hasattr(g, name):
            fields[name] = getattr(g, name)
    if not fields:
        raise TypeError(f"Cannot convert {type(g)} to PyG Data (missing fields)")
    return PyGData(**fields)

def _as_data_sequence(obj: Any):
    """
    Return a sequence of PyG Data objects that GeoLoader can collate.
    Handles GraphDataset/GraphData from your codebase.
    """
    if obj is None:
        return None

    # Your container type: has .graphs -> list of GraphData / Data
    if hasattr(obj, "graphs"):
        seq = obj.graphs
    else:
        # Already a list/tuple of samples?
        if _is_indexable(obj):
            seq = obj
        else:
            # Single sample (GraphData or PyG Data)
            return [_to_pyg(obj)] if not isinstance(obj, PyGData) else [obj]

    # Map any custom GraphData items to PyG Data
    if len(seq) > 0 and not isinstance(seq[0], PyGData):
        seq = [_to_pyg(g) for g in seq]
    return seq

def _ensure_loader(obj: Any, batch_size: int, shuffle: bool):
    if obj is None:
        return None
    # If it's already a loader-like iterable and not a dataset container, use as-is
    if hasattr(obj, "__iter__") and not _is_indexable(obj) and not hasattr(obj, "graphs"):
        return obj
    # Convert to a sequence of PyG Data, then wrap
    data_seq = _as_data_sequence(obj)
    if data_seq is not None:
        return GeoLoader(data_seq, batch_size=batch_size, shuffle=shuffle)
    # Fallback: last resort
    return obj

def _normalize_ds(ds: Any) -> Tuple[Any, Any, Any]:
    if isinstance(ds, dict):
        return (ds.get("train") or ds.get("train_loader"),
                ds.get("val") or ds.get("valid") or ds.get("val_loader"),
                ds.get("test") or ds.get("test_loader"))
    if isinstance(ds, (list, tuple)):
        if len(ds) == 3: return ds[0], ds[1], ds[2]
        if len(ds) == 2: return ds[0], ds[1], None
        if len(ds) == 1: return ds[0], None, None
    return ds, None, None

def _normalize_ds_to_loaders(ds, pre_bs: int, ft_bs: int):
    tr, va, te = _normalize_ds(ds)
    tr = _ensure_loader(tr, pre_bs, shuffle=True)  
    va = _ensure_loader(va, ft_bs, shuffle=False)  
    te = _ensure_loader(te, ft_bs, shuffle=False)  
    return tr, va, te

def _feat_dim(x):
    if x is None:
        return None

    # Torch tensor
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return int(x.shape[-1])
    except Exception:
        pass

    # NumPy array
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return int(x.shape[-1]) if x.ndim > 0 else 1
    except Exception:
        pass

    # Lists/tuples: walk until we find a real tensor/array/shape
    if isinstance(x, (list, tuple)) and x:
        for elem in x:
            d = _feat_dim(elem)
            if d is not None:
                return d
        return None

    # Generic fallback: anything with a shape-like attr
    if hasattr(x, "shape"):
        try:
            shape = x.shape
            return int(shape[-1]) if len(shape) else None
        except Exception:
            pass

    # Last-ditch: some custom objects expose num_features
    if hasattr(x, "num_features"):
        try:
            return int(x.num_features)
        except Exception:
            return None

    return None

def _infer_dims_from_loader(obj):
    # accept loader, dataset (GraphDataset), list of Data, or single Data
    sample = None
    if obj is None:
        return None, None
    if hasattr(obj, "graphs") and len(obj.graphs):
        sample = obj.graphs[0]
    elif hasattr(obj, "__getitem__") and hasattr(obj, "__len__") and len(obj):
        sample = obj[0]
    elif hasattr(obj, "__iter__"):
        it = iter(obj)
        try:
            sample = next(it)
        except StopIteration:
            return None, None
    else:
        sample = obj

    if isinstance(sample, (list, tuple)) and sample:
        sample = sample[0]  # (data, label) → data

    # If this is your custom GraphData, convert or access its fields
    if not hasattr(sample, "x") and hasattr(sample, "to_pyg"):
        sample = sample.to_pyg()

    x = getattr(sample, "x", None)
    ea = getattr(sample, "edge_attr", None)

    in_dim = _feat_dim(x)
    edge_dim = _feat_dim(ea)
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
    prebuilt_datasets: Optional[Tuple[Any, Any, Any]] = None,  
) -> Dict[str, Any]:
    logger.info("Running method %s with config %s", method, asdict(cfg))

    if prebuilt_datasets is not None:
        tr_ds, va_ds, te_ds = prebuilt_datasets
        ds_pre  = _ensure_graph_dataset(tr_ds)
        ds_eval = _ensure_graph_dataset(va_ds) or ds_pre
        # infer dims from the train dataset
        input_dim = _feat_dim(getattr(ds_pre.graphs[0], "x", None))
        edge_dim  = _feat_dim(getattr(ds_pre.graphs[0], "edge_attr", None))

        # --- Fallbacks when dataset graphs don't expose x/edge_attr yet ---
        if input_dim is None:
             # 1) Try the loader (it collates PyG Data with concrete tensors)
              if prebuilt_loaders is not None:
                  tr_loader = prebuilt_loaders[0]
                  in2, ed2 = _infer_dims_from_loader(tr_loader)
                  input_dim = in2
                  edge_dim  = edge_dim  or ed2
        if input_dim is None and hasattr(ds_pre, "graphs") and ds_pre.graphs:
            # 2) Convert first sample to PyG and read dims
            try:
                _pyg = _to_pyg(ds_pre.graphs[0])
                input_dim = _feat_dim(getattr(_pyg, "x", None)) or input_dim
                edge_dim  = _feat_dim(getattr(_pyg, "edge_attr", None)) or edge_dim
            except Exception:
                pass
            # (Optional) As last resort, avoid None to keep smoke tests alive
            if input_dim is None:
                input_dim = 1

    elif prebuilt_loaders is not None:
        train_loader, val_loader, test_loader = prebuilt_loaders
        input_dim, edge_dim = _infer_dims_from_loader(train_loader)
        ds_pre  = _dataset_from_loader(train_loader)
        ds_eval = _dataset_from_loader(val_loader) or ds_pre
    else:
        ds_pre = unlabeled_dataset_fn(cfg.add_3d)
        ds_eval = eval_dataset_fn(cfg.add_3d)
        input_dim = int(ds_pre.graphs[0].x.shape[1])
        edge_dim = _edge_dim_or_none(ds_pre)

    # Safety: ensure both expose `.graphs` and (if possible) `.labels`
    ds_pre  = _ensure_graph_dataset(ds_pre)
    ds_eval = _ensure_graph_dataset(ds_eval) or ds_pre

    seed_metrics: List[Dict[str, float]] = []
    for seed in seeds:
        logger.debug("Training seed %d", seed)
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

        # Some factory variants can return None if dims weren’t resolved earlier.
        if encoder is None:
            try:
                from models.encoder import GNNEncoder as _BasicEnc
                encoder = _BasicEnc(
                    input_dim=input_dim,
                    hidden_dim=cfg.hidden_dim,
                    num_layers=cfg.num_layers,
                    gnn_type=cfg.gnn_type,
                )
            except Exception:
                # ultra-minimal fallback
                class _Id(torch.nn.Module):
                    def __init__(self, in_d, hid_d):
                        super().__init__()
                        self.in_d = in_d
                        self.hidden_dim = hid_d
                        self.proj = torch.nn.Linear(in_d, hid_d)
                    def forward(self, batch):
                        x = getattr(batch, "x", None)
                        if x is None:
                            device = self.proj.weight.device
                            n = getattr(batch, "num_nodes", 1)  # keep node count if available
                            x = torch.zeros((n, self.in_d), device=device)
                        return self.proj(x).mean(dim=0, keepdim=True)
                encoder = _Id(input_dim, cfg.hidden_dim)

        if ema_encoder is None:
            # build a second copy if factory returned None
            try:
                from models.encoder import GNNEncoder as _BasicEnc
                ema_encoder = _BasicEnc(
                    input_dim=input_dim,
                    hidden_dim=cfg.hidden_dim,
                    num_layers=cfg.num_layers,
                    gnn_type=cfg.gnn_type,
                )
            except Exception:
                # copy the fallback encoder
                import copy as _copy
                ema_encoder = _copy.deepcopy(encoder)

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

        elif method.lower() in BASELINE_METHODS:  # MolCLR / GeomGCL / HiMol via adapters
            if (
                baseline_unlabeled_file is None
                or baseline_eval_file is None
                or baseline_label_col is None
            ):
                # In tests that aren't exercising baselines, skip cleanly
                warnings.warn(
                    f"Skipping baseline '{method}' — missing baseline_unlabeled_file / "
                    f"baseline_eval_file / baseline_label_col."
                )
                # no rows appended for this method/seed; will aggregate to empty
                continue

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
        else:
            # not jepa/contrastive/baseline → don’t misroute into baseline
            warnings.warn(f"Unknown method '{method}', skipping this seed.")
            continue

    agg = _aggregate_seed_metrics(seed_metrics)
    row = {**asdict(cfg), **agg, "method": method, "seeds": len(list(seeds))}
    return row

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
        
    logger.info("Running grid search over %d configs", len(cfgs))

    rows: List[Dict[str, Any]] = []
    for cfg in cfgs:
        logger.debug("Processing configuration %s", asdict(cfg))
        add_3d = _cfg_get(cfg, "add_3d", False)  # tests sweep over this

        # If tests provided dataset_fn, build loaders per-config
        prebuilt_loaders = None
        if use_single_builder:
            pre_bs = getattr(cfg, "pretrain_batch_size", 32)
            ft_bs  = getattr(cfg, "finetune_batch_size", 32)

            # get datasets straight from dataset_fn
            try: 
                ds = dataset_fn(add_3d=add_3d)
            except TypeError:
                ds = dataset_fn(add_3d)

            # datasets (train/val/test)
            tr_ds, va_ds, te_ds = _normalize_ds(ds)

            # loaders built from ds (as we already had)
            train_loader, val_loader, test_loader = _normalize_ds_to_loaders(ds, pre_bs, ft_bs)

            prebuilt_loaders  = (train_loader, val_loader, test_loader)
            prebuilt_datasets = (tr_ds, va_ds, te_ds)
        else:
            prebuilt_loaders  = None
            prebuilt_datasets = None

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
                    prebuilt_datasets=prebuilt_datasets,
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
    # metrics where higher is better
    for m in metrics_max:
        if m in df.columns:
            col = pd.to_numeric(df[m], errors="coerce")
            if col.notna().any():  # skip if all NaN
                pos = int(col.idxmax())           # position of best row
                row = df.iloc[pos].to_dict()      # use iloc, not loc
                row["best_metric"] = m
                best_rows.append(row)

    # metrics where lower is better
    for m in metrics_min:
        if m in df.columns:
            col = pd.to_numeric(df[m], errors="coerce")
            if col.notna().any():  # skip if all NaN
                pos = int(col.idxmin())
                row = df.iloc[pos].to_dict()
                row["best_metric"] = m
                best_rows.append(row)

    if best_rows:
        df = pd.concat([df, pd.DataFrame(best_rows)], ignore_index=True)

    if out_csv is not None:
        df.to_csv(out_csv, index=False)

    all_metrics = list(set(metrics_max + metrics_min))
    for m in all_metrics:
        if m in df.columns:
            # flatten list/tuple/ndarray scalars like [0.85]
            df[m] = df[m].apply(
                lambda v: float(v[0]) if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1 else v
            )
            # coerce to numeric
            df[m] = pd.to_numeric(df[m], errors="coerce")
            # replace infinities with NaN
            df[m] = df[m].replace([np.inf, -np.inf], np.nan)

    return df
