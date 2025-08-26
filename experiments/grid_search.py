from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from typing import (  # noqa: E501
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
import torch
import tqdm
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as GeoLoader 
from experiments.baseline_integration import baseline_pretrain_and_embed
from models.ema import EMA
from models.predictor import MLPPredictor
from training.supervised import train_linear_head
from training.train_on_embeddings import (
    train_linear_on_embeddings_classification,
    train_linear_on_embeddings_regression,
)
from training.unsupervised import train_contrastive, train_jepa
from data.augment import AugmentationConfig

ds_pre: Optional[Any] = None
ds_eval: Optional[Any] = None

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

logger = logging.getLogger(__name__)

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
    augmentations: AugmentationConfig
    pretrain_bs: int
    finetune_bs: int
    pretrain_epochs: int
    finetune_epochs: int
    lr: float
    temperature: float


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


from collections.abc import Mapping
def _ensure_graph_dataset(obj: Any) -> Any:
    if obj is None:
        logger.debug("_ensure_graph_dataset received None")
        return None

    if hasattr(obj, "graphs"):
        if getattr(obj, "labels", None) is None:
            labs = _extract_attr(getattr(obj, "graphs"), "y")
            if labs is not None:
                obj.labels = labs  # type: ignore[attr-defined]
        return obj

    # --- handle single graph BEFORE indexable detection ---
    if (
        isinstance(obj, PyGData)
        or hasattr(obj, "x")
        or hasattr(obj, "edge_index")
        or hasattr(obj, "adj")
    ):
        label = getattr(obj, "y", None)
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu()
            if label.numel() == 1:
                label = np.asarray([label.item()])
            else:
                label = label.numpy()
        return _GraphDatasetShim(graphs=[obj], labels=label)

    # If it's a mapping (e.g., dict), leave it alone
    if isinstance(obj, Mapping):
        return obj

    # Indexable container → shim (exclude strings/bytes/mappings)
    if (
        hasattr(obj, "__len__")
        and hasattr(obj, "__getitem__")
        and not isinstance(obj, (str, bytes, Mapping))
    ):
        graphs = list(obj)
        labels = getattr(obj, "labels", None) or _extract_attr(graphs, "y")
        smiles_attr = getattr(obj, "smiles", None)
        smiles = smiles_attr or _extract_attr(graphs, "smiles")
        logger.debug("Built GraphDataset shim with %d graphs", len(graphs))
        return _GraphDatasetShim(graphs=graphs, labels=labels, smiles=smiles)

    logger.debug("_ensure_graph_dataset returning original object of type %s", type(obj))
    return obj


def _dataset_from_loader(loader: Any) -> Any:
    """Get a dataset from a loader.

    If `.dataset` missing, materialize from the loader.
    """
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
    logger.debug(
        "Materialized dataset from loader with %d graphs",
        len(graphs),
    )
    return _GraphDatasetShim(graphs=graphs, labels=labs)


def _build_configs(
    mask_ratios: Iterable[float],
    contiguities: Iterable[bool],
    hidden_dims: Iterable[int],
    num_layers_list: Iterable[int],
    gnn_types: Iterable[str],
    ema_decays: Iterable[float],
    add_3d_options: Iterable[bool],
    augmentation_options: Iterable[AugmentationConfig],
    pretrain_batch_sizes: Iterable[int],
    finetune_batch_sizes: Iterable[int],
    pretrain_epochs_options: Iterable[int],
    finetune_epochs_options: Iterable[int],
    lrs: Iterable[float],
    temperatures: Iterable[float],
) -> List[Config]:
    combos = product(
        mask_ratios,
        contiguities,
        hidden_dims,
        num_layers_list,
        gnn_types,
        ema_decays,
        add_3d_options,
        augmentation_options,
        pretrain_batch_sizes,
        finetune_batch_sizes,
        pretrain_epochs_options,
        finetune_epochs_options,
        lrs,
        temperatures,
    )
    configs = [
        Config(
            mask_ratio=mask_ratio,
            contiguous=contiguous,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            ema_decay=ema_decay,
            add_3d=add_3d,
            augmentations=aug,            # <-- pass the whole AugmentationConfig
            pretrain_bs=pre_bs,
            finetune_bs=finetune_bs,
            pretrain_epochs=pre_epochs,
            finetune_epochs=finetune_epochs,
            lr=lr,
            temperature=temperature,
        )
        for (
            mask_ratio,
            contiguous,
            hidden_dim,
            num_layers,
            gnn_type,
            ema_decay,
            add_3d,
            aug,
            pre_bs,
            finetune_bs,
            pre_epochs,
            finetune_epochs,
            lr,
            temperature,
        ) in combos
    ]
    logger.debug("Generated %d grid search configurations", len(configs))
    return configs


def _aggregate_seed_metrics(
    metrics_list: List[Dict[str, float]],
) -> Dict[str, float]:
    # union of keys (probing/cluster may be absent for some seeds)
    keys = sorted({k for m in metrics_list for k in m.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array(
            [m[k] for m in metrics_list if k in m],
            dtype=np.float64,
        )
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
        msg = f"Cannot convert {type(g)} to PyG Data (missing fields)"
        raise TypeError(msg)
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
    # If it's already a loader-like iterable and not a dataset
    # container, use as-is
    if (
        hasattr(obj, "__iter__")
        and not _is_indexable(obj)
        and not hasattr(obj, "graphs")
    ):
        return obj
    # Convert to a sequence of PyG Data, then wrap
    data_seq = _as_data_sequence(obj)
    if data_seq is not None:
        return GeoLoader(data_seq, batch_size=batch_size, shuffle=shuffle)
    # Fallback: last resort
    return obj


def _normalize_ds(ds: Any) -> Tuple[Any, Any, Any]:
    if isinstance(ds, dict):
        return (
            ds.get("train") or ds.get("train_loader"),
            ds.get("val") or ds.get("valid") or ds.get("val_loader"),
            ds.get("test") or ds.get("test_loader"),
        )
    if isinstance(ds, (list, tuple)):
        if len(ds) == 3:
            return ds[0], ds[1], ds[2]
        if len(ds) == 2:
            return ds[0], ds[1], None
        if len(ds) == 1:
            return ds[0], None, None
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
    """Your existing util—kept for the non-prebuilt path.
    Implement as before."""
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
    # fast-path caps
    target_pretrain_samples: int = 0,
    max_pretrain_batches: int = 0,
    max_finetune_batches: int = 0,
    time_left: Optional[Callable[[], float]] = None,
    # performance knobs
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    bf16: bool = False,
) -> Dict[str, Any]:
    logger.info("Running method %s with config %s", method, asdict(cfg))

    if prebuilt_datasets is not None:
        tr_ds, va_ds, te_ds = prebuilt_datasets
        ds_pre = _ensure_graph_dataset(tr_ds)
        ds_eval = _ensure_graph_dataset(va_ds) or ds_pre
        # infer dims from the train dataset
        input_dim = _feat_dim(getattr(ds_pre.graphs[0], "x", None))
        edge_dim = _feat_dim(getattr(ds_pre.graphs[0], "edge_attr", None))

        # --- Fallbacks when dataset graphs don't expose x/edge_attr yet ---
        if input_dim is None:
            # 1) Try the loader (it collates PyG Data with concrete tensors)
            if prebuilt_loaders is not None:
                tr_loader = prebuilt_loaders[0]
                in2, ed2 = _infer_dims_from_loader(tr_loader)
                input_dim = in2
                edge_dim = edge_dim or ed2
        if input_dim is None and hasattr(ds_pre, "graphs") and ds_pre.graphs:
            # 2) Convert first sample to PyG and read dims
            try:
                _pyg = _to_pyg(ds_pre.graphs[0])
                input_dim = _feat_dim(getattr(_pyg, "x", None)) or input_dim
                edge_attr_dim = _feat_dim(getattr(_pyg, "edge_attr", None))
                edge_dim = edge_attr_dim or edge_dim
            except Exception:
                pass
            # (Optional) As last resort, avoid None to keep smoke tests alive
            if input_dim is None:
                input_dim = 1

    elif prebuilt_loaders is not None:
        train_loader, val_loader, test_loader = prebuilt_loaders
        input_dim, edge_dim = _infer_dims_from_loader(train_loader)
        ds_pre = _dataset_from_loader(train_loader)
        ds_eval = _dataset_from_loader(val_loader) or ds_pre
    else:
        ds_pre = unlabeled_dataset_fn(cfg.add_3d)
        ds_eval = eval_dataset_fn(cfg.add_3d)
        input_dim = int(ds_pre.graphs[0].x.shape[1])
        edge_dim = _edge_dim_or_none(ds_pre)

    # Safety: ensure both expose `.graphs` and (if possible) `.labels`
    ds_pre = _ensure_graph_dataset(ds_pre)
    ds_eval = _ensure_graph_dataset(ds_eval) or ds_pre

    # ---- per-config pretrain cap ----
    max_batches_for_trial = max_pretrain_batches
    if target_pretrain_samples > 0:
        try:
            ds_len = len(ds_pre.graphs)
        except Exception:
            ds_len = 0
        if ds_len > 0 and cfg.pretrain_bs > 0:
            batches_from_target = math.ceil(target_pretrain_samples / cfg.pretrain_bs)
            batches_from_data = cfg.pretrain_epochs * math.ceil(ds_len / cfg.pretrain_bs)
            cap = min(batches_from_target, batches_from_data)
            max_batches_for_trial = (
                cap if max_pretrain_batches <= 0 else min(cap, max_pretrain_batches)
            )

    seed_metrics: List[Dict[str, float]] = []

    for seed in seeds:
        logger.debug("Training seed %d", seed)
        np.random.seed(seed)
        if method.lower() == "jepa":
            row = {}
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

            # If ema_encoder factory returned None, build a copy now
            if ema_encoder is None:
                try:
                    from models.encoder import GNNEncoder as _BasicEnc

                    ema_encoder = _BasicEnc(
                        input_dim=input_dim,
                        hidden_dim=cfg.hidden_dim,
                        num_layers=cfg.num_layers,
                        gnn_type=cfg.gnn_type,
                    )
                except Exception:
                    import copy as _copy

                    ema_encoder = _copy.deepcopy(encoder)

            # --- ensure models are on device BEFORE creating EMA ---
            _dev = torch.device(device)
            encoder = encoder.to(_dev)
            ema_encoder = ema_encoder.to(_dev)
            predictor = MLPPredictor(
                embed_dim=cfg.hidden_dim, hidden_dim=cfg.hidden_dim * 2
            ).to(_dev)
            # EMA clones from encoder; now buffers land on the right device
            ema = EMA(encoder, decay=cfg.ema_decay)

            remaining = time_left() if time_left is not None else float("inf")
            if remaining <= 0:
                logger.info("Time budget exhausted before JEPA pretraining; stopping.")
                break
            _tb = 0 if math.isinf(remaining) else remaining
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
                    max_batches=max_batches_for_trial,
                    time_budget_mins=_tb,
                    # perf knobs
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    bf16=bf16,
                )
            except TypeError:
                # Backward-compatible call
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

            remaining = time_left() if time_left is not None else float("inf")
            if remaining <= 0:
                logger.info("Time budget exhausted before fine-tuning; stopping.")
                break
            _tb = 0 if math.isinf(remaining) else remaining
            try:
                m = train_linear_head(
                    dataset=ds_eval,
                    encoder=encoder,
                    task_type=task_type,
                    epochs=cfg.finetune_epochs,
                    lr=5e-3,
                    batch_size=cfg.finetune_bs,
                    device=device,
                    use_scaffold=use_scaffold,
                    max_batches=max_finetune_batches,
                    time_budget_mins=_tb,
                    # perf knobs
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    bf16=bf16,
                )
                row = {k: float(v) for k, v in m.items() if k != "head"}
            except TypeError:
                logger.exception("Fine-tuning failed in JEPA")
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
            row = {}
            encoder = build_encoder(
                gnn_type=cfg.gnn_type,
                input_dim=input_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                edge_dim=edge_dim,
            )
            remaining = time_left() if time_left is not None else float("inf")
            if remaining <= 0:
                logger.info(
                    "Time budget exhausted before contrastive pretraining; stopping."
                )
                break
            _tb = 0 if math.isinf(remaining) else remaining
            # normalize aug names (supports rotate/dihedral or random_rotate/perturb_dihedral)
            _rot = getattr(cfg.augmentations, "random_rotate", getattr(cfg.augmentations, "rotate", False))
            _dih = getattr(cfg.augmentations, "perturb_dihedral", getattr(cfg.augmentations, "dihedral", False))
            _ang = getattr(cfg.augmentations, "mask_angle", False)
            try:
                train_contrastive(
                    dataset=ds_pre,
                    encoder=encoder,
                    projection_dim=64,
                    epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs,
                    mask_ratio=cfg.mask_ratio,
                    lr=cfg.lr,
                    temperature=cfg.temperature,
                    device=device,
                    use_wandb=use_wandb,
                    ckpt_path=f"{ckpt_dir}/contrast",
                    ckpt_every=ckpt_every,
                    use_scheduler=use_scheduler,
                    warmup_steps=warmup_steps,
                    random_rotate=_rot,
                    mask_angle=_ang,
                    perturb_dihedral=_dih,
                    max_batches=max_batches_for_trial,
                    time_budget_mins=_tb,
                    # perf knobs
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    bf16=bf16,
                )
            except TypeError:
                # older signature without extra knobs
                train_contrastive(
                    dataset=ds_pre,
                    encoder=encoder,
                    projection_dim=64,
                    epochs=cfg.pretrain_epochs,
                    batch_size=cfg.pretrain_bs,
                    mask_ratio=cfg.mask_ratio,
                    lr=cfg.lr,
                    temperature=cfg.temperature,
                    device=device,
                    random_rotate=_rot,
                    mask_angle=_ang,
                    perturb_dihedral=_dih,
                )
            remaining = time_left() if time_left is not None else float("inf")
            if remaining <= 0:
                logger.info("Time budget exhausted before fine-tuning; stopping.")
                break
            _tb = 0 if math.isinf(remaining) else remaining
            try:
                m = train_linear_head(
                    dataset=ds_eval,
                    encoder=encoder,
                    task_type=task_type,
                    epochs=cfg.finetune_epochs,
                    lr=5e-3,
                    batch_size=cfg.finetune_bs,
                    device=device,
                    use_scaffold=use_scaffold,
                    max_batches=max_finetune_batches,
                    time_budget_mins=_tb,
                    # perf knobs
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    bf16=bf16,
                )
                row = {k: float(v) for k, v in m.items() if k != "head"}
            except TypeError:
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

        elif (
            method.lower() in BASELINE_METHODS
        ):  # MolCLR / GeomGCL / HiMol via adapters
            if (
                baseline_unlabeled_file is None
                or baseline_eval_file is None
                or baseline_label_col is None
            ):
                # In tests that aren't exercising baselines, skip cleanly
                warnings.warn(
                    (
                        "Skipping baseline '%s' — missing "
                        "baseline_unlabeled_file / "
                        "baseline_eval_file / "
                        "baseline_label_col."
                    )
                    % method
                )
                # no rows appended for this method/seed;
                # will aggregate to empty
                continue

            _, emb_file = baseline_pretrain_and_embed(
                method=method,
                unlabeled_file=baseline_unlabeled_file,
                smiles_eval_file=baseline_eval_file,
                cfg_path=baseline_cfg,
            )
            if baseline_eval_file.endswith(".csv"):
                df_eval = pd.read_csv(baseline_eval_file)
            else:
                df_eval = pd.read_parquet(baseline_eval_file)
            y = df_eval[baseline_label_col].to_numpy()
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
    # materialize seeds to count reliably (iterables can be one-shot)
    _seeds = tuple(seeds) if not isinstance(seeds, tuple) else seeds
    row = {**asdict(cfg), **agg, "method": method, "seeds": len(_seeds)}
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
    augmentation_options: Tuple[AugmentationConfig, ...] = (
        AugmentationConfig(False, False, False),
    ),
    pretrain_batch_sizes: Tuple[int, ...] = (256,),
    finetune_batch_sizes: Tuple[int, ...] = (64,),
    pretrain_epochs_options: Tuple[int, ...] = (50,),
    finetune_epochs_options: Tuple[int, ...] = (30,),
    lrs: Tuple[float, ...] = (1e-4,),
    temperatures: Tuple[float, ...] = (0.1,0.2),
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
    # --- fast-path caps for grid-search ---
    target_pretrain_samples: int = 0,
    max_pretrain_batches: int = 0,
    max_finetune_batches: int = 0,
    time_budget_mins: int = 0,
    disable_tqdm: bool = False,
    # performance knobs to propagate into trainers
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    bf16: bool = False,
) -> pd.DataFrame:
    

    # ---- robust timing helpers (monotonic) ----
    start = time.perf_counter()
    time_budget_mins = float(time_budget_mins or 0)
    deadline = start + (time_budget_mins * 60.0) if time_budget_mins > 0 else None
    def budget_exhausted() -> bool:
        # 0.25s cushion avoids boundary flicker
        return deadline is not None and time.perf_counter() >= (deadline - 0.25)
    def time_left() -> float:
        """Minutes remaining (∞ if no budget)."""
        if deadline is None:
            return float("inf")
        return max(0.0, (deadline - time.perf_counter()) / 60.0)

    if "contrastive" not in {m.lower() for m in methods}:
        augmentation_options = (AugmentationConfig(False, False, False),)

    cfgs = _build_configs(
        mask_ratios,
        contiguities,
        hidden_dims,
        num_layers_list,
        gnn_types,
        ema_decays,
        add_3d_options,
        augmentation_options,
        pretrain_batch_sizes,
        finetune_batch_sizes,
        pretrain_epochs_options,
        finetune_epochs_options,
        lrs,
        temperatures,
    )

    # ---------------- dataset wiring ----------------
    # We ONLY decide which path to use here; we DO NOT build datasets yet.
    use_single_builder = dataset_fn is not None

    logger.info("Running grid search over %d configs", len(cfgs))

    rows: List[Dict[str, Any]] = []
    if (
        time_budget_mins
        or max_pretrain_batches
        or max_finetune_batches
        or target_pretrain_samples
    ):
        logger.info(
            "Grid caps: time_budget=%s min, target_pretrain_samples=%s, "
            "max_pretrain_batches=%s, max_finetune_batches=%s",
            time_budget_mins,
            target_pretrain_samples,
            max_pretrain_batches,
            max_finetune_batches,
        )

    import sys

    total_pairs = len(cfgs) * len(methods)
    disable_bar = disable_tqdm or (not sys.stdout.isatty())
    pbar = None if disable_bar else tqdm.tqdm(total=total_pairs, leave=False)

    processed = 0
    total = len(cfgs) * len(methods)
    stop_early = False

    for cfg in cfgs:
        logger.debug("Processing configuration %s", asdict(cfg))
        add_3d = _cfg_get(cfg, "add_3d", False)  # tests sweep over this

        # If tests provided dataset_fn, build loaders per-config
        prebuilt_loaders = None
        if use_single_builder:
            pre_bs = cfg.pretrain_bs
            ft_bs  = cfg.finetune_bs

            # get datasets straight from dataset_fn
            try:
                ds = dataset_fn(add_3d=add_3d)
            except TypeError:
                ds = dataset_fn(add_3d)

            # datasets (train/val/test)
            tr_ds, va_ds, te_ds = _normalize_ds(ds)

            # loaders built from ds (as we already had)
            train_loader, val_loader, test_loader = _normalize_ds_to_loaders(
                ds, pre_bs, ft_bs
            )

            prebuilt_loaders = (train_loader, val_loader, test_loader)
            prebuilt_datasets = (tr_ds, va_ds, te_ds)
        else:
            prebuilt_loaders = None
            prebuilt_datasets = None

        for method in methods:
            # Check BEFORE launching the next trial
            if budget_exhausted():
                elapsed = (time.perf_counter() - start) / 60.0
                logger.info(
                    "Time budget exhausted; processed %d/%d configs in %.2f/%.2f min.",
                    processed, total, elapsed, time_budget_mins
                )
                stop_early = True
                break

            res = _run_one_config_method(
                    cfg, method, unlabeled_dataset_fn, eval_dataset_fn, task_type, seeds,
                    device, use_wandb, ckpt_dir, ckpt_every, use_scheduler, warmup_steps,
                    baseline_unlabeled_file, baseline_eval_file, baseline_smiles_col,
                    baseline_label_col, baseline_cfg, use_scaffold,
                    prebuilt_loaders, prebuilt_datasets,
                    target_pretrain_samples, max_pretrain_batches, max_finetune_batches, time_left,
                    num_workers, pin_memory, persistent_workers, prefetch_factor, bf16
                )
            # ensure required keys present for downstream selectors/tests
            row = dict(res) if isinstance(res, dict) else {"result": res}
            row.setdefault("method", method)
            rows.append(row)

            processed += 1
            if pbar is not None:
                pbar.update(1)

        if stop_early:
            break

    if pbar is not None:
        pbar.close()

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
        "rmse_mean","mae_mean","probe_rmse_mean","probe_mae_mean",
        "brier_mean","probe_brier_mean",
    ]
    best_rows: List[Dict[str, Any]] = []
    # metrics where higher is better
    for m in metrics_max:
        if m in df.columns:
            col = pd.to_numeric(df[m], errors="coerce")
            if col.notna().any():  # skip if all NaN
                pos = int(col.idxmax())  # position of best row
                row = df.iloc[pos].to_dict()  # use iloc, not loc
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

    # ---- Centralized selection policy (no env/config) ----
    TIE_EPS = 0.01  # 1% window

    def _first_valid_metric(df: pd.DataFrame, candidates):
        """Return first metric name that exists and has at least one non-NaN value."""
        cols = set(df.columns)
        for m in candidates:
            if m in cols:
                s = pd.to_numeric(df[m], errors="coerce")
                if s.notna().any():
                    return m
        return None
    
    def _selection_policy(df: pd.DataFrame, task_type: str):
        """
        Returns: primary_metric (str), maximize_primary (bool),
                tiebreakers: List[Tuple[column_name (str), maximize (bool)]]
        """
        cols = set(df.columns)
        is_reg = (task_type or "").lower().startswith("regress")
        if is_reg:
            # Regression: RMSE (min), tie by MAE (min)
            primary = _first_valid_metric(df, ["probe_rmse_mean", "rmse_mean"])
            if primary is None:
                raise ValueError("Regression selection requires probe_rmse_mean or rmse_mean.")
            tb = _first_valid_metric(df, ["probe_mae_mean", "mae_mean"])
            tiebreakers = [(tb, False)] if tb else []
            return primary, False, tiebreakers
        else:
            # Classification: ROC-AUC (max), tie by Brier (min) then PR-AUC (max)
            primary = _first_valid_metric(df, ["probe_roc_auc_mean", "roc_auc_mean", "auroc_mean"])
            if primary is None:
                raise ValueError("Classification selection requires roc_auc_mean/probe_roc_auc_mean/auroc_mean.")
            tb1 = _first_valid_metric(df, ["probe_brier_mean", "brier_mean"])
            tb2 = _first_valid_metric(df, ["probe_pr_auc_mean", "pr_auc_mean"])
            tiebreakers = []
            if tb1: tiebreakers.append((tb1, False))  # minimize Brier
            if tb2: tiebreakers.append((tb2, True))   # maximize PR-AUC
            return primary, True, tiebreakers


    # ----- Final primary selection (with optional tie-breaker) -----
    try:
        primary, maximize, tiebreakers = _selection_policy(df, task_type)
    except ValueError:
        # No expected selection metrics (e.g., test stub only returns "metric").
        # Just return the raw dataframe so callers/tests can inspect it.
        return df
    
    dfx = df.copy()
    dfx[primary] = pd.to_numeric(dfx[primary], errors="coerce")
    dfx = dfx.dropna(subset=[primary])
    if dfx.empty:
        raise ValueError(f"No valid values for primary metric '{primary}'")
    best_idx = dfx[primary].idxmax() if maximize else dfx[primary].idxmin()
    best_val = float(dfx.loc[best_idx, primary])

    # tie window
    close = dfx[dfx[primary] >= best_val * (1.0 - TIE_EPS)].copy() if maximize \
            else dfx[dfx[primary] <= best_val * (1.0 + TIE_EPS)].copy()
    suffix = primary
    for tb, tb_max in tiebreakers:
        if not tb or tb not in close.columns:
            continue
        close.loc[:, tb] = pd.to_numeric(close[tb], errors="coerce")
        close_tb = close.dropna(subset=[tb])
        if close_tb.empty or len(close_tb) <= 1:
            continue
        best_idx = close_tb[tb].idxmax() if tb_max else close_tb[tb].idxmin()
        suffix = f"{primary}+tie:{tb}"
        break
    final_row = df.loc[best_idx].to_dict()
    final_row["best_metric"] = suffix
    df = pd.concat([df, pd.DataFrame([final_row])], ignore_index=True)

    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        # annotate resulting table with caps (useful for debugging)
    df["cap_time_mins"] = time_budget_mins
    df["cap_pretrain_batches"] = max_pretrain_batches
    df["cap_target_pretrain_samples"] = target_pretrain_samples
    df["cap_finetune_batches"] = max_finetune_batches

    all_metrics = list(set(metrics_max + metrics_min))
    for m in all_metrics:
        if m in df.columns:
            # flatten list/tuple/ndarray scalars like [0.85]
            df[m] = df[m].apply(
                lambda v: (
                    float(v[0])
                    if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1
                    else v
                )
            )
            # coerce to numeric
            df[m] = pd.to_numeric(df[m], errors="coerce")
            # replace infinities with NaN
            df[m] = df[m].replace([np.inf, -np.inf], np.nan)

    return df
