# flake8: noqa

"""End-to-end JEPA training and evaluation pipeline.

This script orchestrates self‑supervised JEPA pretraining, fine‑tuning/evaluation,
benchmarking and optional case study on Tox21.  Rather than duplicating
core logic, it delegates to reusable modules defined in the repository.  All
hyper‑parameters live in `default.yaml` and can be overridden via CLI.

Stages:
    - `pretrain`: run JEPA on unlabelled data, optionally contrastive baseline.
    - `finetune`: train a linear head on labelled data, averaging metrics over multiple seeds.
    - `evaluate`: same as finetune but without saving the head.
    - `benchmark`: compare JEPA vs contrastive encoders on the same dataset, reporting the better.
    - `tox21`: run a real case study on a Tox21 CSV.

If available, grid search and case study helpers from `experiments` are used.

Each major step is logged to Weights & Biases when enabled. Distinct exit
codes are used so that GitHub Actions can determine which stage failed.
"""


from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from data.augment import iter_augmentation_options

# Attempt to import reusable components from the package.
try:
    from data.mdataset import GraphData, GraphDataset
    from data.augment import AugmentationConfig

except Exception:
    GraphData = GraphDataset = None  # type: ignore[assignment]
    load_directory_dataset = None  # type: ignore[assignment]

    @dataclass(frozen=True)
    class AugmentationConfig:
        rotate: bool = False
        mask_angle: bool = False
        dihedral: bool = False

        @classmethod
        def from_dict(cls, cfg: Optional[dict] = None) -> "AugmentationConfig":
            cfg = cfg or {}
            return cls(
                rotate=bool(cfg.get("rotate", False)),
                mask_angle=bool(cfg.get("mask_angle", False)),
                dihedral=bool(cfg.get("dihedral", False)),
            )


# Models
try:
    from models.factory import build_encoder  # provides 'edge_mpnn' + fallbacks
except Exception:
    # fallback to basic encoder if factory not present
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


try:
    from models.ema import EMA  # type: ignore[assignment]
    from models.predictor import MLPPredictor  # type: ignore[assignment]
except Exception:
    EMA = None  # type: ignore[assignment]
    MLPPredictor = None  # type: ignore[assignment]

# --- Minimal linear head builder (works for classification & regression) ---
try:
    # If you later add a proper head somewhere, import it here:
    from models.heads import build_linear_head  # type: ignore
except Exception:
    build_linear_head = None  # type: ignore[assignment]

    import torch.nn as nn

    class _LinearHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.fc(x)

    def build_linear_head(
        in_dim: int, num_classes: int, task_type: str = "classification"
    ):
        """
        Returns a simple linear probe:
        - classification: out_dim = num_classes
        - regression: out_dim = 1
        """
        out_dim = num_classes if task_type == "classification" else 1
        return _LinearHead(in_dim, out_dim)


try:
    from training.unsupervised import (  # type: ignore[assignment]
        train_contrastive,
        train_jepa,
    )
except Exception:
    train_jepa = None  # type: ignore[assignment]
    train_contrastive = None  # type: ignore[assignment]

try:
    from training.supervised import train_linear_head  # type: ignore[assignment]
except Exception:
    train_linear_head = None  # type: ignore[assignment]

try:
    from experiments.case_study import run_tox21_case_study  # type: ignore[assignment]
except Exception:
    run_tox21_case_study = None  # type: ignore[assignment]

try:
    from experiments.grid_search import run_grid_search  # type: ignore[assignment]
except Exception:
    run_grid_search = None  # type: ignore[assignment]

try:
    from utils.logging import maybe_init_wandb  # type: ignore[assignment]
except Exception:
    # Provide a stub if W&B logging isn't available
    def maybe_init_wandb(*args, **kwargs):  # type: ignore[assignment]
        class DummyWB:
            def log(self, *a, **k):
                pass

            def finish(self) -> None:
                pass

        return DummyWB()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Mock test support utils
# ---------------------------------------------------------------------------


def _maybe_to(module, device):
    """Call .to(device) if present (tests use dummy encoders without .to)."""
    if hasattr(module, "to"):
        module.to(device)
    return module


def _maybe_labels(ds):
    """Best-effort extraction of labels from various dataset shapes.
    Returns a NumPy array or None if not available."""
    import numpy as _np

    for attr in ("y", "labels", "targets"):
        if hasattr(ds, attr):
            try:
                return _np.asarray(getattr(ds, attr))
            except Exception:
                return None
    return None


def _load_state_dict_forgiving(module, state):
    """Call load_state_dict with strict=False when supported; fall back otherwise."""
    try:
        module.load_state_dict(state, strict=False)
    except TypeError:
        module.load_state_dict(state)


def _safe_load_checkpoint(path: str, device: str):
    """
    Best-effort checkpoint loader:
      - Returns the loaded state (or {"encoder": {}}) for valid .pt files
      - Returns {} if the file is not a valid PyTorch checkpoint (common in smoke tests)
    This keeps CLI/tests from crashing when a stub file is used.
    """
    try:
        return torch.load(path, map_location=device)
    except Exception as e:
        logger.warning(
            "Could not load checkpoint %r (%s). Proceeding with random init (test/smoke mode).",
            path,
            e,
        )
        # Return empty dict so load_state_dict is a no-op if needed
        return {"encoder": {}}


def _infer_num_classes(labeled) -> int:
    """Best-effort class count. Falls back to 2 if we can't see labels."""
    # 1) explicit attributes
    for attr in ("num_classes", "n_classes", "classes"):
        if hasattr(labeled, attr):
            try:
                n = int(getattr(labeled, attr))
                if n > 0:
                    return n
            except Exception:
                pass
    # 2) try labels array
    import numpy as np

    y = _maybe_labels(labeled)  # your helper from the previous step
    if y is None:
        return 2
    try:
        y = np.asarray(y)
        if y.size == 0:
            return 2
        if y.ndim > 1:
            y = y[:, 0]
        # robust to non-integer labels
        uniq = np.unique(y[~np.isnan(y)]) if y.dtype.kind in "fc" else np.unique(y)
        if uniq.dtype.kind in "iu":
            return int(uniq.max() + 1)
        return int(len(uniq)) or 2
    except Exception:
        return 2


def _iter_params(m):
    ps = getattr(m, "parameters", None)
    if callable(ps):
        try:
            return list(ps())
        except Exception:
            return []
    return []


def _maybe_state_dict(obj):
    if obj is None:
        return None
    """Return obj.state_dict() if available, else None (for test dummies)."""
    sd = getattr(obj, "state_dict", None)
    if callable(sd):
        try:
            return sd()
        except Exception:
            return None
    return None


def load_directory_dataset(
    dirpath: str,
    ext: str = "parquet",
    smiles_col: str = "smiles",
    label_col: Optional[str] = None,
    cache_dir: Optional[str] = None,
    prefix_filter: Optional[str] = None,
    add_3d: bool = False,
    random_seed: Optional[int] = None,
    n_rows_per_file: Optional[int] = None,
    max_graphs: Optional[int] = None,
    num_workers: int = 0,
) -> GraphDataset:
    return GraphDataset.from_directory(
        dirpath=dirpath,
        ext=ext,
        smiles_col=smiles_col,
        label_col=label_col,
        cache_dir=cache_dir,
        add_3d=add_3d,
        random_seed=random_seed,
        prefix_filter=prefix_filter,
        n_rows_per_file=n_rows_per_file,
        max_graphs=max_graphs,
        num_workers=num_workers,
    )


def load_parquet_dataset(
    filepath: str,
    smiles_col: str = "smiles",
    label_col: Optional[str] = None,
    cache_dir: Optional[str] = None,
    add_3d: bool = False,
    random_seed: Optional[int] = None,
    n_rows: Optional[int] = None,
) -> GraphDataset:
    return GraphDataset.from_parquet(
        filepath=filepath,
        smiles_col=smiles_col,
        label_col=label_col,
        cache_dir=cache_dir,
        add_3d=add_3d,
        random_seed=random_seed,
        n_rows=n_rows,
    )


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Load defaults eagerly.  These are used as defaults for CLI arguments.
CONFIG = load_config(Path(__file__).with_name("default.yaml"))
DEFAULT_AUG = AugmentationConfig.from_dict(
    CONFIG.get("pretrain", {}).get("augmentations", {})
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean and std for each metric across runs.

    Excludes the key 'head' if present.
    """
    if not metrics_list:
        return {}
    out: Dict[str, float] = {}
    keys = sorted({k for d in metrics_list for k in d.keys() if k != "head"})
    for k in keys:
        vals = np.array([d[k] for d in metrics_list if k in d], dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def resolve_device(preferred: str) -> str:
    """Return a valid PyTorch device string."""
    if preferred and preferred != "cpu" and torch.cuda.is_available():
        return preferred
    return "cpu"


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_pretrain(args: argparse.Namespace) -> None:
    """Self‑supervised pretraining of a JEPA encoder and optional contrastive baseline."""
    logger.info("Starting pretrain with args: %s", args)
    if load_directory_dataset is None or build_encoder is None or train_jepa is None:
        logger.error("Pretraining modules are unavailable.")
        sys.exit(2)

    # W&B run
    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "unlabeled_dir": args.unlabeled_dir,
            "gnn_type": args.gnn_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "mask_ratio": args.mask_ratio,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "ema_decay": args.ema_decay,
            "contrastive": args.contrastive,
        },
    )

    from utils.checkpoint import load_checkpoint, save_checkpoint

    # Resume state
    args.ckpt_dir = getattr(args, "ckpt_dir", "ckpts/pretrain")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_every = max(1, int(getattr(args, "save_every", 1)))
    start_epoch = 0
    if getattr(args, "resume_ckpt", None):
        wb.log({"phase": "pretrain", "status": "resume", "ckpt": args.resume_ckpt})
        ckpt_state = load_checkpoint(args.resume_ckpt)
    else:
        ckpt_state = {}

    # Load unlabeled dataset
    try:
        unlabeled = load_directory_dataset(
            args.unlabeled_dir,
            add_3d=args.add_3d,
            num_workers=getattr(args, "num_workers", 0),
            cache_dir=getattr(args, "cache_dir", None),
        )  # type: ignore[arg-type]

        # Sample a subset of the unlabeled dataset if requested.  Use getattr to
        # avoid AttributeError when the caller hasn’t set sample_unlabeled.
        sample_ul = getattr(args, "sample_unlabeled", 0)
        if (
            sample_ul
            and hasattr(unlabeled, "__len__")
            and len(unlabeled) > sample_ul
            and hasattr(unlabeled, "random_subset")
        ):
            unlabeled = unlabeled.random_subset(sample_ul, seed=42)

        wb.log({"phase": "data_load", "unlabeled_graphs": len(unlabeled)})
    except Exception:
        logger.exception("Failed to load unlabeled dataset")
        wb.log({"phase": "data_load", "status": "error"})
        sys.exit(1)

    input_dim = unlabeled.graphs[0].x.shape[1]
    edge_dim = (
        None
        if unlabeled.graphs[0].edge_attr is None
        else unlabeled.graphs[0].edge_attr.shape[1]
    )
    device = resolve_device(args.device)

    # Build encoder and EMA copy
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
    ema_helper = EMA(encoder, decay=args.ema_decay)  # type: ignore[call-arg]
    predictor = MLPPredictor(embed_dim=args.hidden_dim, hidden_dim=args.hidden_dim * 2)  # type: ignore[call-arg]

    # If resuming, load model/optimizer states
    if ckpt_state:
        if "encoder" in ckpt_state:
            encoder._load_state_dict_forgiving(ckpt_state["encoder"])
        if "ema_encoder" in ckpt_state:
            ema_encoder._load_state_dict_forgiving(ckpt_state["ema_encoder"])
        if "predictor" in ckpt_state:
            predictor._load_state_dict_forgiving(ckpt_state["predictor"])
        if "ema" in ckpt_state and hasattr(ema_helper, "load_state_dict"):
            ema_helper._load_state_dict_forgiving(ckpt_state["ema"])
        start_epoch = ckpt_state.get("epoch", 0) + 1

    # Pretrain JEPA
    try:
        wb.log({"phase": "pretrain", "status": "start"})
        for epoch in range(start_epoch, args.epochs):
            train_jepa(
                dataset=unlabeled,
                encoder=encoder,
                ema_encoder=ema_encoder,
                predictor=predictor,
                ema=ema_helper,
                epochs=1,  # one epoch per loop so we can checkpoint each epoch
                max_batches=getattr(
                    args, "max_pretrain_batches", 0
                ),  # ensure it does not crash for unit tests
                time_budget_mins=getattr(
                    args, "time_budget_mins", 0
                ),  # ensure it does not crash for unit tests
                batch_size=args.batch_size,
                mask_ratio=args.mask_ratio,
                contiguous=args.contiguous,
                lr=args.lr,
                device=device,
                reg_lambda=1e-4,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_tags=args.wandb_tags,
                disable_tqdm=True,  # suppress single‑epoch progress bars
            )
            # save after each epoch (or every N epochs)
            if (epoch + 1) % save_every == 0 or (epoch + 1) == args.epochs:
                save_checkpoint(
                    os.path.join(args.ckpt_dir, f"pt_epoch_{epoch+1}.pt"),
                    epoch=epoch,
                    encoder=encoder.state_dict(),
                    ema_encoder=ema_encoder.state_dict(),
                    predictor=(
                        predictor.state_dict()
                        if hasattr(predictor, "state_dict")
                        else None
                    ),
                    ema=ema_helper.state_dict()
                    if hasattr(ema_helper, "state_dict")
                    else None,
                )
        wb.log({"phase": "pretrain", "status": "success"})
    except Exception:
        logger.exception("JEPA pretraining failed")
        wb.log({"phase": "pretrain", "status": "error"})
        sys.exit(2)
    
    aug_cfg = AugmentationConfig(
        rotate=args.aug_rotate,
        mask_angle=args.aug_mask_angle,
        dihedral=args.aug_dihedral,
    )

    # Optionally run contrastive baseline
    if args.contrastive:
        cont_encoder = build_encoder(
            gnn_type=args.gnn_type,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            edge_dim=edge_dim,
        )
        try:
            wb.log({"phase": "pretrain_contrastive", "status": "start"})
            train_contrastive(  # type: ignore[call-arg]
                dataset=unlabeled,
                encoder=cont_encoder,
                epochs=args.epochs,
                batch_size=args.batch_size,
                mask_ratio=args.mask_ratio,
                lr=args.lr,
                device=device,
                use_wandb=args.use_wandb,
                random_rotate=aug_cfg.rotate,
                mask_angle=aug_cfg.mask_angle,
                perturb_dihedral=aug_cfg.dihedral,
                wandb_project=args.wandb_project,
                wandb_tags=args.wandb_tags,
                disable_tqdm=True,  # suppress single‑epoch progress bars
            )
            wb.log({"phase": "pretrain_contrastive", "status": "success"})
        except Exception:
            logger.exception("Contrastive pretraining failed")
            wb.log({"phase": "pretrain_contrastive", "status": "error"})
            sys.exit(2)

    # Save checkpoints
    ckpt_base = args.output
    torch.save({"encoder": encoder.state_dict()}, ckpt_base)
    wb.log({"jepa_checkpoint": ckpt_base})
    if args.contrastive:
        cont_path = f"{os.path.splitext(ckpt_base)[0]}_contrastive.pt"
        torch.save({"encoder": cont_encoder.state_dict()}, cont_path)
        wb.log({"contrastive_checkpoint": cont_path})

    wb.finish()


def cmd_finetune(args: argparse.Namespace) -> None:
    """Fine‑tune a linear head on labelled data across multiple seeds resume & checkpoints."""
    logger.info("Starting finetune with args: %s", args)

    from utils.checkpoint import load_checkpoint, save_checkpoint

    # Directories / resume
    args.ckpt_dir = getattr(args, "ckpt_dir", "ckpts/finetune")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    resume_state = {}

    if getattr(args, "resume_ckpt", None):
        # wb may not exist yet; use logger or postpone this log until after wb is created
        logger.info("[finetune] resuming from %s", args.resume_ckpt)
        resume_state = load_checkpoint(args.resume_ckpt)

    if (
        load_directory_dataset is None
        or build_encoder is None
        or train_linear_head is None
    ):
        wb.error("Fine‑tuning modules are unavailable.")
        sys.exit(3)

    # Determine seeds: CLI overrides config
    seeds: List[int]
    if args.seeds is not None and len(args.seeds) > 0:
        seeds = args.seeds
    else:
        seeds = CONFIG.get("finetune", {}).get("seeds", [0])  # type: ignore[assignment]

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "labeled_dir": args.labeled_dir,
            "gnn_type": args.gnn_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "task_type": args.task_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "ema_decay": args.ema_decay,
            "seeds": seeds,
        },
    )

    # Load labelled dataset
    try:
        labeled = load_directory_dataset(
            args.labeled_dir,
            label_col=args.label_col,
            add_3d=args.add_3d,
            num_workers=getattr(args, "num_workers", 0),
            cache_dir=getattr(args, "cache_dir", None),
        )  # type: ignore[arg-type]

        # Sample a subset of labeled graphs if requested.  Use getattr to
        # handle cases where sample_labeled isn’t provided.
        sample_lb = getattr(args, "sample_labeled", 0)
        if (
            sample_lb
            and hasattr(labeled, "__len__")
            and len(labeled) > sample_lb
            and hasattr(labeled, "random_subset")
        ):
            labeled = labeled.random_subset(sample_lb, seed=42)

        wb.log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset")
        wb.log({"phase": "data_load", "status": "error"})
        sys.exit(1)

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = (
        None
        if labeled.graphs[0].edge_attr is None
        else labeled.graphs[0].edge_attr.shape[1]
    )
    device = resolve_device(args.device)

    # Aggregate metrics across seeds
    metrics_runs: List[Dict[str, float]] = []

    # choose metric & direction
    metric_name = getattr(args, "metric", "val_loss")
    higher_is_better = metric_name.lower() in {"acc", "accuracy", "auc", "auroc", "f1"}

    def _is_better(curr, best):
        return (curr > best) if higher_is_better else (curr < best)

    save_every = max(1, int(getattr(args, "save_every", 1)))

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)  # okay if no CUDA; harmless on CPU
        except Exception:
            pass

        encoder = build_encoder(
            gnn_type=args.gnn_type,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            edge_dim=edge_dim,
        )
        # ensure modules on device
        _maybe_to(encoder, device)

        # Load checkpoint
        # Load pretrained encoder weights (from pretrain output)
        if getattr(args, "encoder", None):
            enc_state = _safe_load_checkpoint(args.encoder, device)
            if enc_state is not None:
                logger.info("[finetune] loaded encoder from %s", args.encoder)
                _load_state_dict_forgiving(
                    encoder,
                    enc_state if "encoder" not in enc_state else enc_state["encoder"],
                )
            else:
                logger.warning("Encoder not loaded; proceeding with random init")

        # If resuming a fine-tune checkpoint, it may contain a fresher encoder
        if "encoder" in resume_state:
            logger.info("Overriding encoder from resume checkpoint")
            _load_state_dict_forgiving(encoder, resume_state["encoder"])
        if "head" in resume_state and hasattr(head, "load_state_dict"):
            _load_state_dict_forgiving(head, resume_state["head"])

        # Build linear head for fine-tuning
        # compute num_classes robustly for classification; for regression we won’t use it
        _in_dim = getattr(encoder, "hidden_dim", getattr(args, "hidden_dim", None))
        assert (
            _in_dim is not None
        ), "hidden dim unknown (encoder.hidden_dim or args.hidden_dim required)"
        if args.task_type == "classification":
            # robust class count
            num_classes = _infer_num_classes(labeled)

            # optional label stats (never break if missing)
            y_arr = _maybe_labels(labeled)
            if y_arr is not None:
                try:
                    # simple example: fraction of positives if binary labels
                    import numpy as _np

                    arr = _np.asarray(y_arr)
                    if arr.ndim > 1:
                        arr = arr[:, 0]
                    pos_frac = float((arr > 0).mean())
                    if "wb" in locals() and wb:
                        wb.log({"dataset/pos_frac": pos_frac})
                except Exception:
                    # Never let metrics logging break training
                    pass

            head = build_linear_head(
                in_dim=_in_dim, num_classes=num_classes, task_type="classification"
            )
        else:
            # regression
            head = build_linear_head(
                in_dim=_in_dim, num_classes=1, task_type="regression"
            )

        _maybe_to(head, device)

        # Optimizer & scheduler
        params = _iter_params(encoder) + _iter_params(head)
        optimizer = (
            torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4) if params else None
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        # If resuming, load head/optim/scheduler from checkpoint
        if "head" in resume_state:
            head.load_state_dict(resume_state["head"], strict=False)
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"], strict=False)
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"], strict=False)

        # epoch to start from (resume file stores last finished epoch)
        start_epoch = int(resume_state.get("epoch", -1)) + 1
        if start_epoch < 0:
            start_epoch = 0

        try:
            wb.log({"phase": f"finetune_{seed}", "status": "start"})

            # per-seed checkpoint dir (avoid overwriting across seeds)
            seed_dir = os.path.join(args.ckpt_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            # initialize best depending on direction
            best_metric = -float("inf") if higher_is_better else float("inf")

            for epoch in range(start_epoch, args.epochs):
                metrics = train_linear_head(
                    dataset=labeled,
                    encoder=encoder,
                    head=head,
                    task_type=args.task_type,
                    epochs=1,
                    max_batches=getattr(
                        args, "max_pretrain_batches", 0
                    ),  # ensure it does not crash for unit tests
                    time_budget_mins=getattr(
                        args, "time_budget_mins", 0
                    ),  # ensure it does not crash for unit tests
                    lr=args.lr,
                    batch_size=args.batch_size,
                    device=device,
                    patience=args.patience,
                    devices=args.devices,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                current = metrics.get(metric_name, None)
                if current is not None:
                    # normalize to float
                    try:
                        current = float(current)
                    except Exception:
                        current = None
                if current is not None and _is_better(current, best_metric):
                    best_metric = current
                    save_checkpoint(
                        os.path.join(seed_dir, "ft_best.pt"),
                        epoch=epoch,
                        encoder=encoder.state_dict(),
                        head=head.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        best_metric=best_metric,
                    )
                # periodic (and last-epoch) snapshot
                if ((epoch + 1) % save_every == 0) or ((epoch + 1) == args.epochs):
                    save_payload = {"epoch": epoch}
                    for name, obj in (("encoder", encoder), ("head", head)):
                        sd = _maybe_state_dict(obj)
                        if sd is not None:
                            save_payload[name] = sd
                    if len(save_payload) > 1:
                        save_checkpoint(
                            os.path.join(seed_dir, f"ft_epoch_{epoch+1}.pt"),
                            **save_payload,
                        )
                # advance LR schedule after the epoch
                try:
                    scheduler.step()
                except Exception:
                    pass
            wb.log({"phase": f"finetune_{seed}", "status": "success"})
            metrics_runs.append({k: v for k, v in metrics.items() if k != "head"})
        except Exception:
            logger.exception(f"Fine‑tuning failed on seed {seed}")
            wb.log({"phase": f"finetune_{seed}", "status": "error"})
            sys.exit(3)

    agg = aggregate_metrics(metrics_runs)
    for k, v in agg.items():
        wb.log({f"metric/{k}": v})
    wb.finish()


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a pretrained encoder by training a linear probe across seeds."""
    logger.info("Starting evaluate with args: %s", args)
    # Reuse finetune implementation with a different default config section
    cmd_finetune(args)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Compare JEPA and contrastive encoders on the same labelled dataset  with flexible loading + report.

    Runs training across seeds and reports which method yields better
    performance based on ROC‑AUC (classification) or RMSE (regression).
    """

    logger.info("Starting benchmark with args: %s", args)
    if (
        load_directory_dataset is None
        or build_encoder is None
        or train_linear_head is None
    ):
        logger.warning("Benchmark modules are unavailable.")
        sys.exit(6)

    seeds: List[int]
    if args.seeds is not None and len(args.seeds) > 0:
        seeds = args.seeds
    else:
        seeds = CONFIG.get("benchmark", {}).get("seeds", [0])  # type: ignore[assignment]

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "labeled_dir": args.labeled_dir,
            "task_type": args.task_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seeds": seeds,
        },
    )

    import json
    import os
    import time

    import numpy as np
    import torch

    from utils.checkpoint import load_checkpoint  # for fine-tuned ckpt (encoder+head)

    # --- paths / report ---
    args.report_dir = getattr(args, "report_dir", "reports")
    os.makedirs(args.report_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_stem = getattr(args, "report_stem", f"benchmark_{timestamp}")
    report_json = os.path.join(args.report_dir, report_stem + ".json")
    report_csv = os.path.join(args.report_dir, report_stem + ".csv")

    try:
        labeled = load_directory_dataset(
            args.labeled_dir,
            label_col=args.label_col,
            add_3d=args.add_3d,
            num_workers=getattr(args, "num_workers", 0),
            cache_dir=getattr(args, "cache_dir", None),
        )  # type: ignore[arg-type]
        wb.log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset for benchmarking")
        wb.log({"phase": "data_load", "status": "error"})
        sys.exit(1)

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = (
        None
        if labeled.graphs[0].edge_attr is None
        else labeled.graphs[0].edge_attr.shape[1]
    )
    device = resolve_device(args.device)

    # Prepare results dict
    all_results: Dict[str, Dict[str, float]] = {}
    from typing import Any, Dict

    def evaluate_state(
        state_obj: Dict[str, Any] | Any, method_name: str
    ) -> Dict[str, float]:
        """
        Evaluate an already-loaded state object (either a raw encoder state_dict or a
        dict with key 'encoder'). Always trains a fresh linear head for fairness.
        """
        metrics_runs: List[Dict[str, float]] = []
        for seed in seeds:
            # Repro
            torch.manual_seed(seed)
            np.random.seed(seed)
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

            # Build & load encoder
            enc = build_encoder(
                gnn_type=args.gnn_type,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                edge_dim=edge_dim,
            )
            if isinstance(state_obj, dict) and "encoder" in state_obj:
                _load_state_dict_forgiving(enc, state_obj["encoder"])
            else:
                _load_state_dict_forgiving(enc, state_obj)
            _maybe_to(enc, device)

            # Train fresh head and log metrics
            mets = train_linear_head(
                dataset=labeled,
                encoder=enc,
                task_type=args.task_type,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
                patience=args.patience,
                devices=args.devices,
            )
            metrics_runs.append({k: v for k, v in mets.items() if k != "head"})

        agg = aggregate_metrics(metrics_runs)
        for k, v in agg.items():
            wb.log({f"{method_name}/{k}": v})
        return agg

    # Thin wrappers that load, then call evaluate_state
    def evaluate_encoder(ckpt_path: str, method_name: str) -> Dict[str, float]:
        state = _safe_load_checkpoint(ckpt_path, device)
        return evaluate_state(state, method_name)

    def evaluate_finetuned(ft_ckpt_path: str) -> Dict[str, float]:
        try:
            state = load_checkpoint(ft_ckpt_path)
        except Exception:
            logger.exception("Failed to load fine-tuned checkpoint: %s", ft_ckpt_path)
            return {}
        return evaluate_state(state, "finetuned")

    wb.log({"phase": "benchmark", "status": "start"})
    # Evaluate JEPA
    agg_jepa = evaluate_encoder(args.jepa_encoder, "jepa")
    all_results["jepa"] = agg_jepa

    # Evaluate contrastive
    agg_cont: Dict[str, float] = {}
    if args.contrastive_encoder:
        agg_cont = evaluate_encoder(args.contrastive_encoder, "contrastive")
        all_results["contrastive"] = agg_cont

    # Optional: evaluate a fine-tuned checkpoint that already has a head
    agg_ft: Dict[str, float] = {}
    if getattr(args, "ft_ckpt", None):
        agg_ft = evaluate_finetuned(args.ft_ckpt)
        if agg_ft:
            all_results["finetuned"] = agg_ft

    # Decide which is better
    verdict = "jepa"
    if agg_cont:
        # Choose metric based on task
        if args.task_type == "classification":
            # Higher AUC/ACC is better
            key = (
                "roc_auc_mean"
                if "roc_auc_mean" in agg_jepa
                else ("acc_mean" if "acc_mean" in agg_jepa else None)
            )
            if key and agg_cont.get(key, float("-inf")) > agg_jepa.get(
                key, float("-inf")
            ):
                verdict = "contrastive"
        else:
            # Lower RMSE/MAE is better
            key = (
                "rmse_mean"
                if "rmse_mean" in agg_jepa
                else ("mae_mean" if "mae_mean" in agg_jepa else None)
            )
            if key and agg_cont.get(key, float("inf")) < agg_jepa.get(
                key, float("inf")
            ):
                verdict = "contrastive"

    # If finetuned was evaluated, compare it too
    if "finetuned" in all_results:
        if args.task_type == "classification":
            key = (
                "roc_auc_mean"
                if "roc_auc_mean" in agg_jepa
                else ("acc_mean" if "acc_mean" in agg_jepa else None)
            )
            if key and all_results["finetuned"].get(
                key, float("-inf")
            ) > all_results.get(verdict, {}).get(key, float("-inf")):
                verdict = "finetuned"
        else:
            key = (
                "rmse_mean"
                if "rmse_mean" in agg_jepa
                else ("mae_mean" if "mae_mean" in agg_jepa else None)
            )
            if key and all_results["finetuned"].get(
                key, float("inf")
            ) < all_results.get(verdict, {}).get(key, float("inf")):
                verdict = "finetuned"

    wb.log({"phase": "benchmark", "status": "success", "best_method": verdict})
    logger.info(f"Benchmark completed. Best method: {verdict}")

    # --- Write JSON/CSV report with all results + verdict ---
    try:
        payload = {"results": all_results, "best_method": verdict}
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        # CSV: method,metric,value
        import csv

        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "metric", "value"])
            for method, mets in all_results.items():
                for k, v in mets.items():
                    w.writerow([method, k, v])
        logger.info("Wrote reports: %s , %s", report_json, report_csv)
    except Exception:
        logger.warning("Failed to write reports", exc_info=True)
    finally:
        wb.finish()


def cmd_tox21(args: argparse.Namespace) -> None:
    """Run the Tox21 ranking case study."""
    logger.info("Starting Tox21 case study with args: %s", args)
    if run_tox21_case_study is None:
        logger.error("Case study module is unavailable.")
        sys.exit(5)

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "csv": args.csv,
            "task": args.task,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "num_top_exclude": args.num_top_exclude,
        },
    )

    try:
        wb.log({"phase": "tox21", "status": "start"})
        mean_true, mean_random, mean_jepa, baseline_means = run_tox21_case_study(
            csv_path=args.csv,
            task_name=args.task,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            num_top_exclude=args.num_top_exclude,
            device=resolve_device(args.device),
        )
        # Assemble a single metrics dictionary so all values appear on the same
        # W&B step.  We prefix baseline keys for clarity.  This allows
        # convenient visualisation of all outputs together in the W&B UI.
        metrics = {
            "phase": "tox21",
            "status": "success",
            "mean_true": mean_true,
            "mean_random_after": mean_random,
            "mean_jepa_after": mean_jepa,
        }
        for name, val in baseline_means.items():
            metrics[f"baseline/{name}"] = val
        wb.log(metrics)
    except Exception:
        logger.warning("Tox21 case study failed")
        wb.log({"phase": "tox21", "status": "error"})
        sys.exit(5)
    finally:
        wb.finish()


def cmd_grid_search(args: argparse.Namespace) -> None:
    """Run a hyper‑parameter sweep using the ``run_grid_search`` helper.

    Scenarios:
        #   1) --dataset-dir: same dataset used for both pretraining & eval
        #   2) --unlabeled-dir / --labeled-dir: separate datasets
        # Each closure must accept add_3d and return GraphDataset.

    The search
    space is configurable via CLI flags.

    Results are logged to Weights &
    Biases if enabled and optionally written to a CSV file.  When the grid
    search completes, the best configuration and its metric are reported.
    """
    # Skip grid search if results already exist
    if (
        not getattr(args, "force_refresh", False)
        and args.out_csv
        and args.best_config_out
    ):
        if os.path.exists(args.out_csv) and os.path.exists(args.best_config_out):
            logger.info(
                "Skipping grid search because %s and %s already exist.",
                args.out_csv,
                args.best_config_out,
            )
            return
    # If the experiments module is unavailable, abort with a distinct exit code
    logger.info("Starting grid search with args: %s", args)
    if run_grid_search is None:
        logger.error(
            "Grid search functionality is unavailable. Install the experiments package or check the import."
        )
        sys.exit(7)

    # Convert numerical lists to tuples and boolean flags
    contiguities = tuple(bool(c) for c in args.contiguities)
    add_3d_opts = tuple(bool(a) for a in args.add_3d_options)
    aug_configs = tuple(
        iter_augmentation_options(
            getattr(args, "aug_rotate_options", [0]),
            getattr(args, "aug_mask_angle_options", [0]),
            getattr(args, "aug_dihedral_options", [0]),
        )
    )

    if "contrastive" not in {m.lower() for m in args.methods}:
        aug_configs = tuple(iter_augmentation_options([0], [0], [0]))
    seeds: tuple
    # Determine seeds: use CLI if provided, otherwise fall back to configuration defaults
    if args.seeds is not None and len(args.seeds) > 0:
        seeds = tuple(args.seeds)
    else:
        seeds = tuple(CONFIG.get("finetune", {}).get("seeds", [42, 123, 456]))

    cache_dir = (
        None
        if getattr(args, "no_cache", False)
        else (getattr(args, "cache_dir", None) or "cache/graphs")
    )
    if cache_dir:
        logger.info("Using cache directory %s", cache_dir)
    else:
        logger.info("Graph caching disabled")

    # Create dataset loader closures for run_grid_search without post-hoc
    # sampling. ``sample-unlabeled`` and ``sample-labeled`` act as ``max_graphs``
    # limits and ``n_rows_per_file`` bounds rows per file.
    n_rows_per_file = getattr(args, "n_rows_per_file", None)
    sample_ul = getattr(args, "sample_unlabeled", 0) or None
    sample_lb = getattr(args, "sample_labeled", 0) or None

    _dataset_fn = None
    _unlabeled_fn = None
    _eval_fn = None

    if args.dataset_dir:

        def _dataset_fn(add_3d: bool = False):  # type: ignore[override]
            t0 = time.time()
            ds = load_directory_dataset(
                args.dataset_dir,
                label_col=args.label_col,
                add_3d=add_3d,
                smiles_col=getattr(args, "smiles_col", "smiles"),
                n_rows_per_file=n_rows_per_file,
                max_graphs=max(sample_ul or 0, sample_lb or 0) or None,
                num_workers=args.num_workers,
                cache_dir=cache_dir,
            )
            dt = time.time() - t0
            logger.info(
                "Loaded dataset in %.2fs (%s graphs)",
                dt,
                len(ds) if hasattr(ds, "__len__") else "unknown",
            )
            return ds

        if not args.unlabeled_dir:
            _unlabeled_fn = lambda add_3d=False: _dataset_fn(add_3d=add_3d)
        if not args.labeled_dir:
            _eval_fn = lambda add_3d=False: _dataset_fn(add_3d=add_3d)

    if args.unlabeled_dir:

        def _unlabeled_fn(add_3d: bool = False):
            logger.info(
                "Loading unlabeled (cap=%s, workers=%s)…", sample_ul, args.num_workers
            )
            t0 = time.time()
            ds = load_directory_dataset(
                args.unlabeled_dir,
                add_3d=add_3d,
                smiles_col=getattr(args, "smiles_col", "smiles"),
                n_rows_per_file=n_rows_per_file,
                max_graphs=sample_ul,
                num_workers=args.num_workers,
                cache_dir=cache_dir,
            )
            dt = time.time() - t0
            logger.info(
                "Loaded unlabeled dataset in %.2fs (%s graphs)",
                dt,
                len(ds) if hasattr(ds, "__len__") else "unknown",
            )
            return ds

    if args.labeled_dir:

        def _eval_fn(add_3d: bool = False):
            logger.info(
                "Loading labeled (cap=%s, workers=%s)…", sample_lb, args.num_workers
            )
            t0 = time.time()
            ds = load_directory_dataset(
                args.labeled_dir,
                label_col=args.label_col,
                add_3d=add_3d,
                smiles_col=getattr(args, "smiles_col", "smiles"),
                n_rows_per_file=n_rows_per_file,
                max_graphs=sample_lb,
                num_workers=args.num_workers,
                cache_dir=cache_dir,
            )
            dt = time.time() - t0
            logger.info(
                "Loaded labeled dataset in %.2fs (%s graphs)",
                dt,
                len(ds) if hasattr(ds, "__len__") else "unknown",
            )
            return ds

    if _unlabeled_fn is None:
        logger.info(
            "Grid search requires at least one dataset source: --dataset-dir or "
            "(--unlabeled-dir and/or --labeled-dir). possibly running in unit test mode"
        )
        _unlabeled_fn = lambda add_3d=False: None  # noqa: E731
    if _eval_fn is None:
        logger.info(
            "Grid search requires at least one dataset source: --dataset-dir or "
            "(--unlabeled-dir and/or --labeled-dir). possibly running in unit test mode"
        )
        _eval_fn = lambda add_3d=False: None  # noqa: E731
    # Initialise optional W&B run for grid search
    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "dataset_dir": args.dataset_dir,
            "unlabeled_dir": args.unlabeled_dir,
            "labeled_dir": args.labeled_dir,
            "task_type": args.task_type,
            "methods": args.methods,
            "mask_ratios": args.mask_ratios,
            "contiguities": args.contiguities,
            "hidden_dims": args.hidden_dims,
            "num_layers_list": args.num_layers_list,
            "gnn_types": args.gnn_types,
            "ema_decays": args.ema_decays,
            "add_3d_options": args.add_3d_options,
            "pretrain_batch_sizes": args.pretrain_batch_sizes,
            "finetune_batch_sizes": args.finetune_batch_sizes,
            "pretrain_epochs_options": args.pretrain_epochs_options,
            "finetune_epochs_options": args.finetune_epochs_options,
            "learning_rates": args.learning_rates,
            "seeds": seeds,
        },
    )

    # --- safe wandb helpers (avoid crashes if inner code closed the run) ---
    def _wb_active(w):
        try:
            return hasattr(w, "run") and getattr(w, "run", None) is not None
        except Exception:
            return False

    # --- safe wandb logger: try if .log exists; swallow real-W&B preinit errors ---
    def _wb_log(w, payload):
        if hasattr(w, "log"):
            try:
                w.log(payload)
            except Exception as e:
                logger.warning("Skipping wandb.log: %s", e)

    _wb_log(wb, {"phase": "grid_search", "status": "start"})

    try:
        df = run_grid_search(
            dataset_fn=_dataset_fn,
            unlabeled_dataset_fn=_unlabeled_fn,
            eval_dataset_fn=_eval_fn,
            methods=tuple(args.methods),
            task_type=args.task_type,
            seeds=seeds,
            mask_ratios=tuple(args.mask_ratios),
            contiguities=contiguities,
            hidden_dims=tuple(args.hidden_dims),
            num_layers_list=tuple(args.num_layers_list),
            gnn_types=tuple(args.gnn_types),
            ema_decays=tuple(args.ema_decays),
            add_3d_options=add_3d_opts,
            augmentation_options=aug_configs,
            pretrain_batch_sizes=tuple(args.pretrain_batch_sizes),
            finetune_batch_sizes=tuple(args.finetune_batch_sizes),
            pretrain_epochs_options=tuple(args.pretrain_epochs_options),
            finetune_epochs_options=tuple(args.finetune_epochs_options),
            lrs=tuple(args.learning_rates),
            device=args.device,
            use_wandb=args.use_wandb,
            ckpt_dir=args.ckpt_dir,
            ckpt_every=args.ckpt_every,
            use_scheduler=args.use_scheduler,
            warmup_steps=args.warmup_steps,
            out_csv=args.out_csv,
            # If your run_grid_search signature supports these, they’ll be used;
            # otherwise they’ll be ignored (or remove them here).
            max_pretrain_batches=getattr(args, "max_pretrain_batches", 0),
            max_finetune_batches=getattr(args, "max_finetune_batches", 0),
            time_budget_mins=getattr(args, "time_budget_mins", 0),
            disable_tqdm=not getattr(args, "force_tqdm", False)
            and not sys.stdout.isatty(),
        )
        # Log each row to W&B for comprehensive visualisation.  We assign a
        # unique identifier to each configuration using its index.  This
        # produces a separate log entry per configuration, enabling plots and
        # tables in the W&B UI.
        best_conf = None
        if df is not None and not df.empty:
            for idx, row in df.iterrows():
                # Prepare a metrics dict excluding non-numeric entries and
                # include the index as "config_id".  Flatten any lists or
                # arrays to scalars when possible.
                metrics_dict = {"config_id": int(idx)}
                for col, val in row.items():
                    if isinstance(val, (list, tuple)) and len(val) == 1:
                        val = val[0]
                    metrics_dict[col] = val
                _wb_log(wb, metrics_dict)
            best_conf = df.iloc[-1].to_dict()
            logger.info("Grid search completed. Best configuration: %s", best_conf)
            # Optionally write the best configuration to a JSON file for later use.
            if args.best_config_out:
                try:
                    import json

                    with open(args.best_config_out, "w", encoding="utf-8") as f:
                        json.dump(best_conf, f, indent=2)
                    logger.info("Saved best configuration to %s", args.best_config_out)
                except Exception:
                    logger.exception("Failed to write best configuration to JSON")
        else:
            logger.info("Grid search returned no results.")
        _wb_log(wb, {"phase": "grid_search", "status": "success", "best": best_conf})
    except Exception:
        logger.exception("Grid search failed")
        try:
            active = hasattr(wb, "run") and getattr(wb, "run", None) is not None
        except Exception:
            active = False
        if active and hasattr(wb, "log"):
            try:
                _wb_log(wb, {"phase": "grid_search", "status": "error"})
            except Exception as e:
                logger.warning("Skipping wandb.log in error path: %s", e)
        # exit with distinct code for grid search failures
        sys.exit(7)
    finally:
        try:
            if hasattr(wb, "finish"):
                wb.finish()
        except Exception as e:
            logger.warning("Skipping wandb.finish(): %s", e)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser, section: str) -> None:
    """Add arguments common to multiple commands.

    Defaults are taken from the given config section.
    """
    # Model hyperparameters
    model_cfg = CONFIG.get("model", {})
    p.add_argument(
        "--gnn-type",
        type=str,
        default=model_cfg.get("gnn_type", "mpnn"),
        help="GNN encoder type",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=model_cfg.get("hidden_dim", 64),
        help="Hidden dimension size",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=model_cfg.get("num_layers", 2),
        help="Number of GNN layers",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=model_cfg.get("ema_decay", 0.99),
        help="EMA decay rate",
    )
    # Data augmentations and options
    p.add_argument(
        "--add-3d", action="store_true", help="Augment with 3D coordinate featurisation"
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Process pool workers for SMILES conversion (0=serial)",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache processed graphs",
    )
    p.add_argument(
        "--contiguous",
        action="store_true",
        help="Use contiguous subgraph masking (JEPA)",
    )
    p.add_argument(
        "--aug-rotate",
        action="store_true",
        default=DEFAULT_AUG.rotate,
        help="Randomly rotate coordinates during pretraining",
    )
    p.add_argument(
        "--aug-mask-angle",
        action="store_true",
        default=DEFAULT_AUG.mask_angle,
        help="Mask bond angles during pretraining",
    )
    p.add_argument(
        "--aug-dihedral",
        action="store_true",
        default=DEFAULT_AUG.dihedral,
        help="Perturb dihedral angles during pretraining",
    )
    # Optimisation
    sec_cfg = CONFIG.get(section, {})
    p.add_argument(
        "--epochs",
        type=int,
        default=sec_cfg.get("epochs", 1),
        help="Number of training epochs",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=sec_cfg.get("batch_size", 32),
        help="Batch size",
    )
    p.add_argument(
        "--lr", type=float, default=sec_cfg.get("lr", 1e-3), help="Learning rate"
    )
    # Seeds for downstream evaluation
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Random seeds for averaging results",
    )
    # Device & DDP
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs for DDP")
    # W&B
    wandb_cfg = CONFIG.get("wandb", {})
    p.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=wandb_cfg.get("project", "m-jepa"),
        help="W&B project name",
    )
    p.add_argument(
        "--wandb-tags",
        nargs="*",
        default=wandb_cfg.get("tags", []),
        help="Tags for W&B run",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="JEPA training and evaluation pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Pretrain subcommand
    pre = sub.add_parser("pretrain", help="Self‑supervised pretraining")
    pre.add_argument(
        "--unlabeled-dir",
        required=True,
        help="Directory of unlabeled graphs (.parquet or .csv)",
    )
    pre.add_argument(
        "--output",
        type=str,
        default="encoder.pt",
        help="Where to save the JEPA encoder checkpoint",
    )
    pre.add_argument(
        "--contrastive", action="store_true", help="Also run a contrastive baseline"
    )
    pre.add_argument(
        "--ckpt-dir",
        type=str,
        default="ckpts/pretrain",
        help="Directory to save pretrain checkpoints",
    )
    pre.add_argument(
        "--resume-ckpt",
        type=str,
        default="",
        help="Resume pretraining from a checkpoint",
    )
    pre.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save a pretrain checkpoint every N epochs",
    )
    pre.add_argument(
        "--mask-ratio",
        type=float,
        default=0.15,
        help="Fraction of nodes to mask in each view (JEPA/contrastive).",
    )

    _add_common_args(pre, "pretrain")
    pre.set_defaults(func=cmd_pretrain)

    # Fine‑tune subcommand
    ft = sub.add_parser("finetune", help="Fine‑tune a linear head on labelled data")
    ft.add_argument(
        "--labeled-dir",
        required=True,
        help="Directory of labelled graphs (.parquet or .csv)",
    )
    ft.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Label column name in input files",
    )
    ft.add_argument(
        "--encoder", required=True, help="Path to a pretrained encoder checkpoint (.pt)"
    )

    ft.add_argument(
        "--ckpt-dir",
        type=str,
        default="ckpts/finetune",
        help="dir to write fine-tune checkpoints",
    )
    ft.add_argument(
        "--resume-ckpt",
        type=str,
        default="",
        help="resume fine-tune from this checkpoint",
    )
    ft.add_argument(
        "--save-every", type=int, default=1, help="save checkpoint every N epochs"
    )
    ft.add_argument(
        "--save-final", action="store_true", help="also save ft_last.pt at the end"
    )
    ft.add_argument(
        "--metric", type=str, default="val_loss", choices=["val_loss", "acc", "auroc"]
    )

    ft.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        default="classification",
    )
    ft.add_argument(
        "--patience",
        type=int,
        default=CONFIG.get("finetune", {}).get("patience", 10),
        help="Early stopping patience",
    )
    _add_common_args(ft, "finetune")
    ft.set_defaults(func=cmd_finetune)

    # Evaluate subcommand (alias for finetune)
    ev = sub.add_parser(
        "evaluate", help="Evaluate a pretrained encoder via a linear probe"
    )
    ev.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs")
    ev.add_argument("--label-col", type=str, default="label", help="Label column name")
    ev.add_argument(
        "--encoder", required=True, help="Path to a pretrained encoder checkpoint (.pt)"
    )
    ev.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        default="classification",
    )
    ev.add_argument(
        "--patience",
        type=int,
        default=CONFIG.get("evaluate", {}).get("patience", 10),
        help="Early stopping patience",
    )
    _add_common_args(ev, "evaluate")
    ev.set_defaults(func=cmd_evaluate)

    # Benchmark subcommand
    bench = sub.add_parser(
        "benchmark", help="Compare JEPA and contrastive encoders on labelled data"
    )
    bench.add_argument(
        "--labeled-dir", required=True, help="Directory of labelled graphs"
    )
    bench.add_argument(
        "--label-col", type=str, default="label", help="Label column name"
    )
    bench.add_argument(
        "--jepa-encoder", required=True, help="Path to a JEPA encoder checkpoint (.pt)"
    )
    bench.add_argument(
        "--contrastive-encoder",
        required=False,
        help="Path to a contrastive encoder checkpoint (.pt)",
    )
    bench.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        default="classification",
    )
    bench.add_argument(
        "--patience",
        type=int,
        default=CONFIG.get("benchmark", {}).get("patience", 10),
        help="Early stopping patience",
    )

    bench.add_argument(
        "--ft-ckpt",
        type=str,
        default="",
        help="fine-tuned checkpoint (expects encoder and optionally head)",
    )  # ?
    bench.add_argument(
        "--report-dir", type=str, default="reports", help="where to write JSON/CSV"
    )
    bench.add_argument(
        "--report-stem",
        type=str,
        default="",
        help="filename stem; defaults to timestamped benchmark_*",
    )

    _add_common_args(bench, "benchmark")
    bench.set_defaults(func=cmd_benchmark)

    # Tox21 case study
    tox = sub.add_parser("tox21", help="Run the Tox21 case study experiment")
    tox.add_argument(
        "--csv",
        required=True,
        help="Path to the Tox21 CSV containing SMILES and labels",
    )
    tox.add_argument(
        "--task", required=True, help="Name of the toxicity column to predict"
    )
    case_cfg = CONFIG.get("case_study", {})
    tox.add_argument(
        "--pretrain-epochs",
        type=int,
        default=case_cfg.get("pretrain_epochs", 5),
        help="JEPA pretrain epochs for case study",
    )
    tox.add_argument(
        "--finetune-epochs",
        type=int,
        default=case_cfg.get("finetune_epochs", 20),
        help="Epochs to train regression head in case study",
    )
    tox.add_argument(
        "--num-top-exclude",
        type=int,
        default=case_cfg.get("num_top_exclude", 10),
        help="Top‑k toxic compounds to exclude when ranking",
    )
    _add_common_args(tox, "case_study")
    tox.set_defaults(func=cmd_tox21)

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------
    # This subcommand exposes the hyper‑parameter sweep functionality from
    # ``experiments.grid_search``.  It allows a user to optimise JEPA
    # pretraining and downstream evaluation parameters across a user‑defined
    # search space.  The defaults mirror those in ``run_grid_search`` but
    # can be overridden on the CLI.  The dataset is specified via
    # ``--dataset-dir`` and will be loaded with the same loader used in
    # other stages.  Seeds and search ranges can also be customised.
    grid = sub.add_parser(
        "grid-search",
        help="Perform hyper‑parameter grid search for JEPA using run_grid_search",
    )
    grid.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="Column name that contains molecule SMILES.",
    )
    grid.add_argument(
        "--dataset-dir",
        required=False,
        default=None,
        help="Path to a graph dataset used for both pretraining and evaluation. "
        "If omitted, you must specify --unlabeled-dir and/or --labeled-dir.",
    )
    grid.add_argument(
        "--unlabeled-dir",
        type=str,
        default=None,
        help=(
            "Directory of an unlabeled graph dataset for JEPA pretraining "
            "(e.g. ZINC/PubChem)."
        ),
    )
    grid.add_argument(
        "--labeled-dir",
        type=str,
        default=None,
        help=(
            "Directory of a labeled graph dataset for downstream evaluation "
            "(e.g. MoleculeNet)."
        ),
    )
    grid.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of the label column in the dataset (ignored for unlabeled data)",
    )
    grid.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached graph data (defaults to cache/graphs)",
    )
    grid.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable graph caching during grid search",
    )
    grid.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for SMILES featurization",
    )
    grid.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        default="classification",
        help="Task type for downstream evaluation",
    )
    grid.add_argument(
        "--methods",
        nargs="+",
        default=["jepa"],
        help="Names of methods to include in the sweep (e.g. jepa contrastive)",
    )
    grid.add_argument(
        "--mask-ratios",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.25],
        help="List of mask ratios to sweep over",
    )
    grid.add_argument(
        "--contiguities",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Contiguity flags (0 for False, 1 for True) to sweep over",
    )
    grid.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 256],
        help="Hidden dimensions to sweep over",
    )
    grid.add_argument(
        "--num-layers-list",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Number of GNN layers to sweep over",
    )
    grid.add_argument(
        "--gnn-types",
        nargs="+",
        default=["mpnn", "gcn", "gat", "edge_mpnn"],
        help="GNN architectures to sweep over",
    )
    grid.add_argument(
        "--ema-decays",
        type=float,
        nargs="+",
        default=[0.95, 0.99],
        help="EMA decay rates to sweep over",
    )
    grid.add_argument(
        "--add-3d-options",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Whether to include 3D features (0 for False, 1 for True)",
    )
    grid.add_argument(
        "--aug-rotate-options",
        type=int,
        nargs="+",
        default=[int(DEFAULT_AUG.rotate)],
        help="Apply random rotation augmentation (0 for False, 1 for True)",
    )
    grid.add_argument(
        "--aug-mask-angle-options",
        type=int,
        nargs="+",
        default=[int(DEFAULT_AUG.mask_angle)],
        help="Apply angle masking augmentation (0 for False, 1 for True)",
    )
    grid.add_argument(
        "--aug-dihedral-options",
        type=int,
        nargs="+",
        default=[int(DEFAULT_AUG.dihedral)],
        help="Apply dihedral perturbation augmentation (0 for False, 1 for True)",
    )
    grid.add_argument(
        "--pretrain-batch-sizes",
        type=int,
        nargs="+",
        default=[256],
        help="Batch sizes for JEPA pretraining",
    )
    grid.add_argument(
        "--finetune-batch-sizes",
        type=int,
        nargs="+",
        default=[64],
        help="Batch sizes for downstream fine‑tuning",
    )
    grid.add_argument(
        "--pretrain-epochs-options",
        type=int,
        nargs="+",
        default=[50],
        help="Number of epochs for JEPA pretraining",
    )
    grid.add_argument(
        "--finetune-epochs-options",
        type=int,
        nargs="+",
        default=[30],
        help="Number of epochs for downstream training",
    )
    grid.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-4],
        help="Learning rates to sweep over",
    )
    grid.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Random seeds for averaging results (overrides config)",
    )
    grid.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (cuda or cpu)",
    )
    grid.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Path to output CSV file for grid search results",
    )
    grid.add_argument(
        "--ckpt-dir",
        type=str,
        default="outputs/grid_ckpts",
        help="Directory in which to save intermediate checkpoints during the sweep",
    )
    grid.add_argument(
        "--ckpt-every",
        type=int,
        default=25,
        help="Checkpoint every N epochs during pretraining in the sweep",
    )
    grid.add_argument(
        "--use-scheduler",
        action="store_true",
        help="Enable learning‑rate warmup and cosine scheduler during grid search",
    )
    grid.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps for the scheduler during grid search",
    )
    grid.add_argument(
        "--best-config-out",
        type=str,
        default=None,
        help=(
            "Optional path to write the best hyper‑parameter configuration "
            "as a JSON file. This file can be parsed later to drive a "
            "production pretraining run."
        ),
    )
    grid.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging for the grid search",
    )
    grid.add_argument(
        "--wandb-project",
        type=str,
        default=CONFIG.get("wandb", {}).get("project", "m-jepa"),
        help="W&B project name for grid search runs",
    )
    grid.add_argument(
        "--wandb-tags",
        nargs="*",
        default=CONFIG.get("wandb", {}).get("tags", []),
        help="W&B tags for grid search runs",
    )
    grid.add_argument(
        "--force-tqdm",
        action="store_true",
        help="Force-enable tqdm progress bars even when not attached to a TTY",
    )
    grid.add_argument(
        "--sample-unlabeled",
        type=int,
        default=0,
        help="If >0, load at most N graphs from the unlabeled dataset.",
    )
    grid.add_argument(
        "--sample-labeled",
        type=int,
        default=0,
        help="If >0, load at most N graphs from the labeled dataset.",
    )
    grid.add_argument(
        "--n-rows-per-file",
        type=int,
        default=None,
        help="If set, limit rows read per file when loading datasets.",
    )
    grid.add_argument(
        "--max-pretrain-batches",
        type=int,
        default=0,
        help="If >0, stop each pretrain epoch after this many batches.",
    )
    grid.add_argument(
        "--max-finetune-batches",
        type=int,
        default=0,
        help="If >0, stop each finetune epoch after this many batches.",
    )
    grid.add_argument(
        "--time-budget-mins",
        type=int,
        default=0,
        help="Optional wallclock budget; stop early when exceeded.",
    )
    grid.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Ignore cached grid search outputs and recompute",
    )

    grid.set_defaults(func=cmd_grid_search)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.error("No subcommand provided")
    logger.info("Invoking subcommand %s", getattr(args, "func", None))
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
