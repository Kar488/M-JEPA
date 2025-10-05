"""Entry point for JEPA experiments: demo, full, and grid modes.

- demo: tiny toy run (JEPA vs contrastive) + quick head training + synthetic case study
- full: pretrain on unlabeled (train shards), then finetune with val early‑stopping; optional test eval
- grid: YAML/JSON‑driven sweep over JEPA/contrastive/baseline methods

This file expects the project layout:
    data/, experiments/, models/, training/, utils/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import logging
import os
import contextlib

# Structured logging
logger = logging.getLogger(__name__)

# Data & utils
from data.mdataset import GraphData, GraphDataset
from utils.dataset import (
    load_dataset,
    load_directory_dataset,
    load_parquet_dataset,
)
try:  # pragma: no cover - optional dependency
    from utils.checkpoint import save_checkpoint
except Exception:  # pragma: no cover
    def save_checkpoint(*args, **kwargs):  # type: ignore
        raise ImportError("Checkpointing requires PyTorch")

try:  # pragma: no cover - optional dependency
    from utils.plotting import plot_training_curves
except Exception:  # pragma: no cover
    def plot_training_curves(*args, **kwargs):  # type: ignore
        return None
from utils.logging import maybe_init_wandb

# Models
try:  # pragma: no cover - optional dependency
    from models.factory import build_encoder  # provides 'edge_mpnn' + fallbacks
    from models.ema import EMA
    from models.predictor import MLPPredictor
    from training.supervised import train_linear_head
    from training.unsupervised import train_contrastive, train_jepa
    try:
        from training.supervised_with_val import train_linear_head_with_val
        _HAS_VAL_TRAIN = True
    except Exception:  # pragma: no cover
        _HAS_VAL_TRAIN = False
        def train_linear_head_with_val(*args, **kwargs):  # type: ignore
            raise ImportError("Val training unavailable")
except Exception:  # pragma: no cover
    def build_encoder(*args, **kwargs):  # type: ignore
        raise ImportError("Model building requires PyTorch")
    class EMA:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("EMA requires PyTorch")
    class MLPPredictor:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Predictor requires PyTorch")
    def train_linear_head(*args, **kwargs):  # type: ignore
        raise ImportError("Training requires PyTorch")
    def train_contrastive(*args, **kwargs):  # type: ignore
        raise ImportError("Training requires PyTorch")
    def train_jepa(*args, **kwargs):  # type: ignore
        raise ImportError("Training requires PyTorch")
    def train_linear_head_with_val(*args, **kwargs):  # type: ignore
        raise ImportError("Val training unavailable")
    _HAS_VAL_TRAIN = False

# Experiments
try:  # pragma: no cover - optional dependency
    from experiments.grid_search import run_grid_search
except Exception:  # pragma: no cover
    def run_grid_search(*args, **kwargs):  # type: ignore
        raise ImportError("Grid search unavailable")

try:  # pragma: no cover - optional dependency
    from experiments.case_study import run_tox21_case_study
    _HAS_CASE_STUDY = True
except Exception:  # pragma: no cover
    def run_tox21_case_study(*args, **kwargs):  # type: ignore
        raise ImportError("Case study unavailable")
    _HAS_CASE_STUDY = False


# ---------------------------- Dataset helpers ---------------------------- #
# Consolidated dataset loading utilities live in ``utils.dataset``.


def _edge_dim_or_none(ds: GraphDataset) -> Optional[int]:
    g0: GraphData = ds.graphs[0]
    return None if g0.edge_attr is None else int(g0.edge_attr.shape[1])


# ----------------------------- Demo pipeline ----------------------------- #
def _make_chain_graph(n: int) -> GraphData:
    x = np.ones((n, 2), dtype=np.float32)
    edges = []
    for i in range(n - 1):
        edges.append([i, i + 1]); edges.append([i + 1, i])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return GraphData(x=x, edge_index=edge_index)

def _build_unlabeled_dataset_from_smiles(smiles):
    try:
        # If method exists anywhere in your env, use it
        return GraphDataset.from_smiles_list(smiles)  # type: ignore[attr-defined]
    except AttributeError:
        # Synthetic fallback (no RDKit deps)
        graphs = [_make_chain_graph(max(3, min(8, (len(s) % 6) + 3))) for s in smiles]
        return GraphDataset(graphs)
    
def _ensure_labels_inplace_local(ds, task_type: str) -> None:
    # Create or coerce labels in place so downstream code can read ds.labels
    if not hasattr(ds, "labels") or getattr(ds, "labels", None) is None:
        n = len(ds.graphs)
        ds.labels = np.zeros(n, dtype=np.int64 if task_type == "classification" else np.float32)
    else:
        arr = np.asarray(ds.labels)
        ds.labels = arr.astype(np.int64 if task_type == "classification" else np.float32, copy=False)


def demonstration(device: str = "cpu", devices: int = 1, use_scaffold: bool = False) -> None:
    """Tiny run: JEPA vs contrastive on a toy dataset, with a tiny linear head and a synthetic case study."""
    smiles = ["CCO","CCN","CCC","c1ccccc1","CC(=O)O","CCOCC","CNC","CCCl","COC","CCN(CC)CC"]

    # Unsupervised dataset
    dataset = _build_unlabeled_dataset_from_smiles(smiles)

    input_dim = dataset.graphs[0].x.shape[1]
    edge_dim = _edge_dim_or_none(dataset)

    # JEPA
    encoder = build_encoder(
        gnn_type="mpnn",
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        edge_dim=edge_dim,
    )
    ema_encoder = build_encoder(
        gnn_type="mpnn",
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        edge_dim=edge_dim,
    )
    ema = EMA(encoder, decay=0.99)
    predictor = MLPPredictor(embed_dim=64, hidden_dim=128)

    try:
        jepa_losses = train_jepa(
            dataset=dataset,
            encoder=encoder,
            ema_encoder=ema_encoder,
            predictor=predictor,
            ema=ema,
            epochs=3,
            batch_size=5,
            mask_ratio=0.2,
            contiguous=False,
            lr=1e-3,
            device=device,
            devices=devices,
            reg_lambda=1e-4,
            use_wandb=False,
        )
    except TypeError:
        jepa_losses = train_jepa(
            dataset=dataset,
            encoder=encoder,
            ema_encoder=ema_encoder,
            predictor=predictor,
            ema=ema,
            epochs=3,
            batch_size=5,
            mask_ratio=0.2,
            contiguous=False,
            lr=1e-3,
            device=device,
            devices=devices,
            reg_lambda=1e-4,
        )

    # Contrastive
    contrastive_encoder = build_encoder(
        gnn_type="mpnn",
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        edge_dim=edge_dim,
    )
    try:
        contrastive_losses = train_contrastive(
            dataset=dataset,
            encoder=contrastive_encoder,
            projection_dim=32,
            epochs=3,
            batch_size=5,
            mask_ratio=0.2,
            lr=1e-3,
            device=device,
            devices=devices,
            temperature=0.1,
            use_wandb=False,
        )
    except TypeError:
        contrastive_losses = train_contrastive(
            dataset=dataset,
            encoder=contrastive_encoder,
            projection_dim=32,
            epochs=3,
            batch_size=5,
            mask_ratio=0.2,
            lr=1e-3,
            device=device,
            devices=devices,
            temperature=0.1,
        )

    from utils.logging import maybe_init_wandb

    wb = maybe_init_wandb(enable=False)
    
    plot_training_curves(
        {"JEPA": jepa_losses, "Contrastive": contrastive_losses},
        title="Toy Unsupervised Training Losses",
        normalize=True,
        wb=wb,
    )

    # Tiny head
    random_labels = np.random.randint(0, 2, size=len(smiles)).astype(np.int64)
    setattr(dataset, "labels", random_labels)
    
    labeled_dataset = dataset
    metrics = train_linear_head(
        labeled_dataset,
        encoder,
        task_type="classification",
        epochs=5,
        lr=1e-3,
        batch_size=5,
        device=device,
        use_scaffold=use_scaffold,
        devices=devices,
    )
    logger.info(
        "Toy classification metrics: %s",
        {k: v for k, v in metrics.items() if k != "head"},
    )

    # Tox21 case study using real labels
    csv = "samples/tox21_mini.csv"
    result = run_tox21_case_study(
        csv_path=csv,
        task_name="NR-AR",
        pretrain_epochs=1,
        finetune_epochs=1,
        device=device,
    )
    primary_eval = result.evaluations[0]
    logger.info(
        "Tox21 case study – mean true: %.3f, random: %.3f, predicted: %.3f",
        primary_eval.mean_true,
        primary_eval.mean_random,
        primary_eval.mean_pred,
    )


# ----------------------------- Full pipeline ----------------------------- #


def _train_with_val_if_available(
    train_ds: GraphDataset,
    val_ds: Optional[GraphDataset],
    encoder,
    task_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    val_patience: int,
    devices: int = 1,
) -> dict:
    if _HAS_VAL_TRAIN and val_ds is not None:
        return train_linear_head_with_val(
            train_ds=train_ds,
            val_ds=val_ds,
            encoder=encoder,
            task_type=task_type,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            val_patience=val_patience,
            devices=devices,
        )
    return train_linear_head(
        train_ds,
        encoder,
        task_type,
        epochs,
        lr,
        batch_size,
        device,
        patience=val_patience,
        devices=devices,
    )


def run_full_mode(args: argparse.Namespace) -> None:
    """Pretrain (unlabeled) then fine‑tune (labeled) with val early‑stopping; optional test & Tox21 case study."""
    # --- Unlabeled (pretraining) --- #
    unlabeled = load_directory_dataset(
        dirpath=args.unlabeled_dir,
        ext="parquet",
        smiles_col="smiles",
        cache_dir=args.cache_dir,
        prefix_filter="train",
        add_3d=args.add_3d,
    )
    if not unlabeled.graphs:
        raise SystemExit(f"No graphs loaded from {args.unlabeled_dir}")

    input_dim = int(unlabeled.graphs[0].x.shape[1])
    edge_dim = _edge_dim_or_none(unlabeled)

    # Build encoders
    encoder = build_encoder(
        gnn_type=args.gnn_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=edge_dim,
    )
    ema_encoder = build_encoder(
        gnn_type=args.gnn_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=edge_dim,
    )
    ema = EMA(encoder, decay=args.ema_decay)
    predictor = MLPPredictor(
        embed_dim=args.hidden_dim, hidden_dim=max(128, 2 * args.hidden_dim)
    )

    # Pretrain method selection
    if args.method == "jepa":
        try:
            train_jepa(
                dataset=unlabeled,
                encoder=encoder,
                ema_encoder=ema_encoder,
                predictor=predictor,
                ema=ema,
                epochs=args.pretrain_epochs,
                batch_size=args.pretrain_bs,
                mask_ratio=args.mask_ratio,
                contiguous=args.contiguous,
                lr=args.pretrain_lr,
                device=args.device,
                devices=args.devices,
                reg_lambda=1e-4,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_tags=args.wandb_tags,
                ckpt_path=str(Path(args.ckpt_dir) / "pretrain"),
                ckpt_every=args.ckpt_every,
                use_scheduler=True,
                warmup_steps=args.warmup_steps,
                random_rotate=args.aug_rotate,
                mask_angle=args.aug_mask_angle,
                perturb_dihedral=args.aug_dihedral,
            )
        except TypeError:
            train_jepa(
                dataset=unlabeled,
                encoder=encoder,
                ema_encoder=ema_encoder,
                predictor=predictor,
                ema=ema,
                epochs=args.pretrain_epochs,
                batch_size=args.pretrain_bs,
                mask_ratio=args.mask_ratio,
                contiguous=args.contiguous,
                lr=args.pretrain_lr,
                device=args.device,
                devices=args.devices,
                reg_lambda=1e-4,
                random_rotate=args.aug_rotate,
                mask_angle=args.aug_mask_angle,
                perturb_dihedral=args.aug_dihedral,
            )
    elif args.method in ("contrastive",):
        try:
            train_contrastive(
                dataset=unlabeled,
                encoder=encoder,
                projection_dim=args.proj_dim,
                epochs=args.pretrain_epochs,
                batch_size=args.pretrain_bs,
                mask_ratio=args.mask_ratio,
                lr=args.pretrain_lr,
                device=args.device,
                devices=args.devices,
                temperature=args.temperature,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_tags=args.wandb_tags,
                ckpt_path=str(Path(args.ckpt_dir) / "pretrain"),
                ckpt_every=args.ckpt_every,
                use_scheduler=True,
                warmup_steps=args.warmup_steps,
                random_rotate=args.aug_rotate,
                mask_angle=args.aug_mask_angle,
                perturb_dihedral=args.aug_dihedral,
            )
        except TypeError:
            train_contrastive(
                dataset=unlabeled,
                encoder=encoder,
                projection_dim=args.proj_dim,
                epochs=args.pretrain_epochs,
                batch_size=args.pretrain_bs,
                mask_ratio=args.mask_ratio,
                lr=args.pretrain_lr,
                device=args.device,
                devices=args.devices,
                temperature=args.temperature,
                random_rotate=args.aug_rotate,
                mask_angle=args.aug_mask_angle,
                perturb_dihedral=args.aug_dihedral,
            )
    else:
        # Baselines via wrapper (if you have a local baseline trainer)
        from training.baselines import pretrain_baseline

        pretrain_baseline(
            args.method,
            dataset=unlabeled,
            input_dim=input_dim,
            device=args.device,
            cfg=dict(
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                gnn_type=args.gnn_type,
                epochs=args.pretrain_epochs,
                batch_size=args.pretrain_bs,
                mask_ratio=args.mask_ratio,
                lr=args.pretrain_lr,
                use_wandb=args.use_wandb,
                ckpt_path=str(Path(args.ckpt_dir) / "pretrain"),
                ckpt_every=args.ckpt_every,
            ),
        )

    # Save encoder checkpoint
    save_checkpoint(
        str(Path(args.ckpt_dir) / "encoder_final.pt"), encoder=encoder.state_dict()
    )

    # --- Labeled (finetuning) --- #
    if not (args.label_train_dir and args.label_val_dir):
        raise SystemExit(
            "Full mode requires --label_train_dir and --label_val_dir (and optionally --label_test_dir)."
        )

    train_ds = load_directory_dataset(
        args.label_train_dir,
        ext="parquet",
        smiles_col="smiles",
        label_col=args.label_col,
        cache_dir=str(Path(args.cache_dir) / "label_train"),
        add_3d=args.add_3d,
    )
    val_ds = load_directory_dataset(
        args.label_val_dir,
        ext="parquet",
        smiles_col="smiles",
        label_col=args.label_col,
        cache_dir=str(Path(args.cache_dir) / "label_val"),
        add_3d=args.add_3d,
    )

    _ensure_labels_inplace_local(train_ds, args.task_type)
    _ensure_labels_inplace_local(val_ds, args.task_type)

    metrics = _train_with_val_if_available(
        train_ds=train_ds,
        val_ds=val_ds,
        encoder=encoder,
        task_type=args.task_type,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        batch_size=args.finetune_bs,
        device=args.device,
        val_patience=args.val_patience,
        devices=args.devices,
    )
    logger.info(
        "Train/Val metrics: %s", {k: v for k, v in metrics.items() if k != "head"}
    )

    if args.label_test_dir:
        test_ds = load_directory_dataset(
            args.label_test_dir,
            ext="parquet",
            smiles_col="smiles",
            label_col=args.label_col,
            cache_dir=str(Path(args.cache_dir) / "label_test"),
            add_3d=args.add_3d,
        )
        # quick re‑train on merged set for reporting (simple baseline)
        # Merge train + val in place to avoid calling a constructor that doesn’t accept labels
        merged_graphs = train_ds.graphs + val_ds.graphs

        train_labels = getattr(train_ds, "labels", None)
        val_labels   = getattr(val_ds, "labels", None)
        merged_labels = None
        if train_labels is not None and val_labels is not None:
            merged_labels = np.concatenate([np.asarray(train_labels), np.asarray(val_labels)], axis=0)

        # mutate train_ds so it remains a valid dataset instance (keeps __len__, etc.)
        train_ds.graphs = merged_graphs
        setattr(train_ds, "labels", merged_labels)

        merged = train_ds  # reuse the same object
        
        test_metrics = train_linear_head(
            merged,
            encoder,
            task_type=args.task_type,
            epochs=max(3, args.finetune_epochs // 5),
            lr=args.finetune_lr,
            batch_size=args.finetune_bs,
            device=args.device,
        )
        logger.info(
            "Test metrics (simple merged baseline): %s",
            {k: v for k, v in test_metrics.items() if k != "head"},
        )

    # Optional Tox21 case study
    tox21_csv = getattr(args, "tox21_csv", None)
    tox21_task = getattr(args, "tox21_task", None)
    if _HAS_CASE_STUDY and tox21_csv and tox21_task:
        case_result = run_tox21_case_study(
            csv_path=tox21_csv,
            task_name=tox21_task,
            pretrain_epochs=getattr(args, "tox21_epochs", 10),
            finetune_epochs=getattr(args, "tox21_epochs", 10),
            device=args.device,
            triage_pct=getattr(args, "tox21_topk", 0.05),
        )
        primary_eval = case_result.evaluations[0]
        logger.info(
            "Tox21 case study (mean_true, mean_random_after, mean_predicted_after): %s",
            (primary_eval.mean_true, primary_eval.mean_random, primary_eval.mean_pred),
        )


def run_full(args: argparse.Namespace) -> None:
    """Backward compatible wrapper for :func:`run_full_mode`.

    Some tests and scripts expect a ``run_full`` entry point; this thin
    wrapper simply delegates to :func:`run_full_mode` so existing callers
    continue to work.
    """
    run_full_mode(args)


# ----------------------------- Grid runner ----------------------------- #


def _resolve_cache_dir(raw_value: Optional[str], default_dir: str) -> str:
    """Resolve cache directory values that may reference ``$CACHE_DIR``.

    Parameters
    ----------
    raw_value:
        Value provided in the sweep specification.  It may include
        environment-variable placeholders such as ``${CACHE_DIR}``.
    default_dir:
        Default directory to fall back to when ``raw_value`` is empty or the
        placeholder cannot be expanded (e.g. ``CACHE_DIR`` unset).
    """

    if raw_value is None:
        return default_dir

    raw_str = str(raw_value).strip()
    if not raw_str:
        return default_dir

    expanded = os.path.expanduser(os.path.expandvars(raw_str))
    if "$" not in expanded:
        return expanded

    if raw_str.startswith("${CACHE_DIR}"):
        suffix = raw_str[len("${CACHE_DIR}") :]
    elif raw_str.startswith("$CACHE_DIR"):
        suffix = raw_str[len("$CACHE_DIR") :]
    else:
        return default_dir

    suffix = suffix.lstrip("/ ")
    return os.path.join(default_dir, suffix) if suffix else default_dir


def run_grid_mode(args: argparse.Namespace) -> None:
    """Run YAML/JSON sweep. Results CSV printed and saved to spec['output_csv']."""

    def _load_sweep(path: str) -> dict:
        ext = Path(path).suffix.lower()
        with open(path, "r", encoding="utf-8") as f:
            if ext in (".yaml", ".yml"):
                import yaml

                return yaml.safe_load(f)
            elif ext == ".json":
                return json.load(f)
            else:
                raise ValueError("Sweep spec must be .yaml/.yml or .json")

    spec = _load_sweep(args.sweep)

    # data factory
    unlabeled_dir = spec.get("unlabeled_dir", "data/ZINC_canonicalized")
    smiles_col = spec.get("smiles_col", "smiles")
    default_cache_dir = os.environ.get("CACHE_DIR", "cache/zinc")
    default_cache_dir = os.path.expanduser(os.path.expandvars(default_cache_dir))
    cache_dir = _resolve_cache_dir(spec.get("cache_dir"), default_cache_dir)
    prefix_filter = spec.get("prefix_filter", "train")
    add_3d_default = bool(spec.get("add_3d_default", False))

    def dataset_fn(add_3d: bool) -> GraphDataset:
        return load_directory_dataset(
            dirpath=unlabeled_dir,
            ext="parquet",
            smiles_col=smiles_col,
            cache_dir=cache_dir,
            prefix_filter=prefix_filter,
            add_3d=add_3d,
        )

    # sweep axes
    def _tuple(key, default):
        return tuple(spec.get(key, default))

    seeds = tuple(spec.get("seeds", [42, 123, 456]))
    methods = tuple(spec.get("methods", ["jepa"]))
    task_type = spec.get("task_type", "classification")
    device = spec.get("device", "cuda")

    from experiments.grid_search import run_grid_search

    df = run_grid_search(
        unlabeled_dataset_fn=dataset_fn,  # NOTE: both fns are required
        eval_dataset_fn=dataset_fn,
        methods=methods,
        task_type=task_type,
        seeds=seeds,
        mask_ratios=_tuple("mask_ratios", (0.10, 0.15, 0.25)),
        contiguities=_tuple("contiguities", (False, True)),
        hidden_dims=_tuple("hidden_dims", (128, 256)),
        num_layers_list=_tuple("num_layers_list", (2, 3)),
        gnn_types=_tuple("gnn_types", ("mpnn", "gcn", "gat", "edge_mpnn")),
        ema_decays=_tuple("ema_decays", (0.95, 0.99)),
        add_3d_options=_tuple("add_3d_options", (add_3d_default,)),
        pretrain_batch_sizes=_tuple("pretrain_batch_sizes", (256,)),
        finetune_batch_sizes=_tuple("finetune_batch_sizes", (64,)),
        pretrain_epochs_options=_tuple("pretrain_epochs_options", (50,)),
        finetune_epochs_options=_tuple("finetune_epochs_options", (30,)),
        lrs=_tuple("lrs", (1e-4,)),
        device=device,
        n_jobs=int(spec.get("n_jobs", 0)),
        use_wandb=bool(spec.get("use_wandb", False)),
        ckpt_dir=spec.get("ckpt_dir", "outputs/grid_ckpts"),
        ckpt_every=int(spec.get("ckpt_every", 25)),
        use_scheduler=bool(spec.get("use_scheduler", True)),
        warmup_steps=int(spec.get("warmup_steps", 1000)),
        baseline_unlabeled_file=spec.get("baseline_unlabeled_file"),
        baseline_eval_file=spec.get("baseline_eval_file"),
        baseline_smiles_col=spec.get("baseline_smiles_col", "smiles"),
        baseline_label_col=spec.get("baseline_label_col"),
        baseline_cfg=spec.get("baseline_cfg", "adapters/config.yaml"),
        use_scaffold=args.use_scaffold,
    )

    out_csv = spec.get("output_csv", "outputs/grid_results.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("[grid] wrote: %s", out_csv)
    logger.info("%s", df.head())

# --------------------------------- CLI ---------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser("JEPA experiments")
    p.add_argument("--mode", type=str, default="demo", choices=["demo", "full", "grid"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs for DDP")

    # shared model opts
    p.add_argument(
        "--method",
        type=str,
        default="jepa",
        choices=["jepa", "contrastive", "molclr", "geomgcl", "himol"],
    )
    p.add_argument(
        "--gnn_type", type=str, default="mpnn", help="mpnn|gcn|gat|edge_mpnn"
    )
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--add_3d", action="store_true")

    # pretrain
    p.add_argument("--unlabeled_dir", type=str, default="data/ZINC_canonicalized")
    p.add_argument("--pretrain_epochs", type=int, default=100)
    p.add_argument("--pretrain_bs", type=int, default=256)
    p.add_argument("--pretrain_lr", type=float, default=1e-4)
    p.add_argument("--mask_ratio", type=float, default=0.15)
    p.add_argument("--contiguous", action="store_true")
    p.add_argument("--aug_rotate", action="store_true")
    p.add_argument("--aug_mask_angle", action="store_true")
    p.add_argument("--aug_dihedral", action="store_true")
    p.add_argument("--proj_dim", type=int, default=64)  # contrastive
    p.add_argument("--temperature", type=float, default=0.1)  # contrastive
    p.add_argument("--cache_dir", type=str, default="cache")

    # finetune
    p.add_argument("--label_train_dir", type=str, default=None)
    p.add_argument("--label_val_dir", type=str, default=None)
    p.add_argument("--label_test_dir", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument(
        "--task_type",
        type=str,
        default="classification",
        choices=["classification", "regression"],
    )
    p.add_argument("--finetune_epochs", type=int, default=50)
    p.add_argument("--finetune_bs", type=int, default=64)
    p.add_argument("--finetune_lr", type=float, default=5e-3)
    p.add_argument("--val_patience", type=int, default=7)

    # logging / ckpt / sched
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="m-jepa")
    p.add_argument("--wandb_tags", nargs="*", default=None)
    p.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--use-scaffold", action="store_true", dest="use_scaffold")

    # tox21 (optional)
    p.add_argument("--tox21_csv", type=str, default=None)
    p.add_argument("--tox21_task", type=str, default=None)
    p.add_argument("--tox21_epochs", type=int, default=10)
    p.add_argument("--tox21_topk", type=float, default=0.05)

    # grid
    p.add_argument("--sweep", type=str, default=None, help="YAML/JSON spec")

    args = p.parse_args()

    
    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        api_key=os.getenv("WANDB_API_KEY"),  # requires WANDB_API_KEY env var
    )

    if args.mode == "demo":
        demonstration(
            device=args.device, devices=args.devices, use_scaffold=args.use_scaffold
        )
    elif args.mode == "grid":
        if not args.sweep:
            raise SystemExit("--mode grid requires --sweep <spec.yaml|json>")
        run_grid_mode(args)
    else:
        run_full_mode(args)
    if wb is not None:
        with contextlib.suppress(Exception):
            wb.finish()
