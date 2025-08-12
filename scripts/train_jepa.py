"""End-to-end JEPA training and evaluation pipeline.

This script wires together the dataset loaders, encoder factory and training
utilities to provide a minimal yet real example of the JEPA workflow:

1. Load an **unlabelled** dataset for self‑supervised pretraining.
2. Pretrain an encoder using the JEPA objective.
3. Load a **labelled** dataset and fine‑tune a linear head.
4. Evaluate the head and report metrics.

Each major step is logged to Weights & Biases when enabled.  Distinct exit
codes are used so that GitHub Actions can determine which stage failed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Optional

import torch

# Allow running as a script without installing the package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import load_directory_dataset, build_encoder  # noqa: E402
from models.ema import EMA  # noqa: E402
from models.predictor import MLPPredictor  # noqa: E402
from training.unsupervised import train_jepa, train_contrastive  # noqa: E402
from training.supervised import train_linear_head  # noqa: E402
from utils.logging import maybe_init_wandb  # noqa: E402


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain JEPA then fine-tune")
    p.add_argument("--unlabeled-dir", required=True, help="Directory of unlabeled graphs")
    p.add_argument(
        "--labeled-dir", required=True, help="Directory of labelled graphs for downstream tasks"
    )
    p.add_argument(
        "--label-col", type=str, default="label", help="Column name for labels in the labelled set"
    )

    # Model hyperparameters
    p.add_argument("--gnn-type", type=str, default="mpnn", help="Encoder architecture")
    p.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size")
    p.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers")
    p.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay rate")

    # Training hyperparameters
    p.add_argument("--pretrain-epochs", type=int, default=1)
    p.add_argument("--finetune-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mask-ratio", type=float, default=0.15)
    p.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience for fine-tune")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs for DDP")

    # W&B options
    p.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="m-jepa")
    p.add_argument("--wandb-tags", nargs="*", default=None)

    # Vast.ai related options (currently informational only)
    p.add_argument("--vast-ai", action="store_true", help="Flag indicating execution on Vast.ai")
    p.add_argument("--vast-logdir", type=str, default=None, help="Optional log directory on Vast.ai")

    # Baseline comparison
    p.add_argument(
        "--contrastive",
        action="store_true",
        help="Also run contrastive pretraining baseline",
    )

    return p.parse_args()


# Distinct exit codes so GitHub Actions can surface which step failed
DATA_LOAD_ERROR = 1
PRETRAIN_ERROR = 2
FINETUNE_ERROR = 3
EVAL_ERROR = 4


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _init_encoder(
    *, gnn_type: str, input_dim: int, hidden_dim: int, num_layers: int, edge_dim: Optional[int]
):
    """Helper to construct an encoder and its EMA copy."""

    encoder = build_encoder(
        gnn_type=gnn_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_dim=edge_dim,
    )
    ema_encoder = build_encoder(
        gnn_type=gnn_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_dim=edge_dim,
    )
    return encoder, ema_encoder


class DataLoadError(RuntimeError):
    """Raised when dataset loading fails."""


class PretrainError(RuntimeError):
    """Raised when pretraining fails."""


class FinetuneError(RuntimeError):
    """Raised when fine-tuning fails."""


class EvalError(RuntimeError):
    """Raised when evaluation fails."""


def load_data(
    unlabeled_dir: str, labeled_dir: str, label_col: str, wb
):
    """Load datasets and infer feature dimensions."""

    wb.log({"phase": "data_load", "status": "start"})
    try:
        unlabeled = load_directory_dataset(unlabeled_dir)
        labeled = load_directory_dataset(labeled_dir, label_col=label_col)
        wb.log(
            {
                "phase": "data_load",
                "status": "success",
                "unlabeled_graphs": len(unlabeled),
                "labeled_graphs": len(labeled),
            }
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to load datasets")
        wb.log({"phase": "data_load", "status": "error"})
        raise DataLoadError(str(e))

    input_dim = unlabeled.graphs[0].x.shape[1]
    edge_dim = (
        None
        if unlabeled.graphs[0].edge_attr is None
        else unlabeled.graphs[0].edge_attr.shape[1]
    )
    return unlabeled, labeled, input_dim, edge_dim


def pretrain(
    unlabeled,
    *,
    gnn_type: str,
    input_dim: int,
    edge_dim: Optional[int],
    hidden_dim: int,
    num_layers: int,
    ema_decay: float,
    batch_size: int,
    mask_ratio: float,
    lr: float,
    epochs: int,
    device: str,
    devices: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_tags,
    contrastive: bool,
    wb,
):
    """Run JEPA pretraining and optional contrastive baseline."""

    wb.log({"phase": "pretrain", "status": "start"})
    try:
        encoder, ema_encoder = _init_encoder(
            gnn_type=gnn_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_dim,
        )
        ema = EMA(encoder, decay=ema_decay)
        predictor = MLPPredictor(embed_dim=hidden_dim, hidden_dim=hidden_dim * 2)

        losses = train_jepa(
            dataset=unlabeled,
            encoder=encoder,
            ema_encoder=ema_encoder,
            predictor=predictor,
            ema=ema,
            epochs=epochs,
            batch_size=batch_size,
            mask_ratio=mask_ratio,
            lr=lr,
            device=device,
            devices=devices,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_tags=wandb_tags,
        )
        final_loss = losses[-1] if losses else None
        wb.log({"phase": "pretrain", "status": "success", "final_loss": final_loss})
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Pretraining failed")
        wb.log({"phase": "pretrain", "status": "error"})
        raise PretrainError(str(e))

    contrastive_encoder = None
    if contrastive:
        contrastive_encoder = build_encoder(
            gnn_type=gnn_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_dim,
        )
        try:
            wb.log({"phase": "pretrain_contrastive", "status": "start"})
            c_losses = train_contrastive(
                dataset=unlabeled,
                encoder=contrastive_encoder,
                epochs=epochs,
                batch_size=batch_size,
                mask_ratio=mask_ratio,
                lr=lr,
                device=device,
                devices=devices,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_tags=wandb_tags,
            )
            c_final = c_losses[-1] if c_losses else None
            wb.log(
                {"phase": "pretrain_contrastive", "status": "success", "final_loss": c_final}
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Contrastive pretraining failed")
            wb.log({"phase": "pretrain_contrastive", "status": "error"})
            raise PretrainError(str(e))

    return encoder, contrastive_encoder


def finetune(
    labeled,
    *,
    encoder,
    task_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    patience: int,
    devices: int,
    contrastive_encoder=None,
    wb,
):
    """Fine-tune linear head on labelled data."""

    wb.log({"phase": "finetune", "status": "start"})
    try:
        metrics: Dict[str, float] = train_linear_head(
            dataset=labeled,
            encoder=encoder,
            task_type=task_type,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            patience=patience,
            devices=devices,
        )
        wb.log({"phase": "finetune", "status": "success"})
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Fine-tuning failed")
        wb.log({"phase": "finetune", "status": "error"})
        raise FinetuneError(str(e))

    contrastive_metrics: Dict[str, float] = {}
    if contrastive_encoder is not None:
        try:
            wb.log({"phase": "finetune_contrastive", "status": "start"})
            contrastive_metrics = train_linear_head(
                dataset=labeled,
                encoder=contrastive_encoder,
                task_type=task_type,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
                patience=patience,
                devices=devices,
            )
            wb.log({"phase": "finetune_contrastive", "status": "success"})
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Contrastive fine-tuning failed")
            wb.log({"phase": "finetune_contrastive", "status": "error"})
            raise FinetuneError(str(e))

    return metrics, contrastive_metrics


def evaluate(metrics: Dict[str, float], contrastive_metrics: Dict[str, float], wb):
    """Log evaluation metrics to W&B."""

    wb.log({"phase": "evaluation", "status": "start"})
    try:
        if metrics:
            wb.log({"phase": "evaluation", **{f"metric/{k}": v for k, v in metrics.items()}})
        if contrastive_metrics:
            wb.log(
                {
                    "phase": "evaluation",
                    **{f"contrastive/{k}": v for k, v in contrastive_metrics.items()},
                }
            )
        wb.log({"phase": "evaluation", "status": "success"})
    except Exception as e:  # pragma: no cover
        logger.exception("Evaluation logging failed")
        wb.log({"phase": "evaluation", "status": "error"})
        raise EvalError(str(e))


def main() -> None:
    args = parse_args()

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "unlabeled_dir": args.unlabeled_dir,
            "labeled_dir": args.labeled_dir,
            "gnn_type": args.gnn_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "mask_ratio": args.mask_ratio,
            "ema_decay": args.ema_decay,
            "vast_ai": args.vast_ai,
            "vast_logdir": args.vast_logdir,
        },
    )

    try:
        unlabeled, labeled, input_dim, edge_dim = load_data(
            args.unlabeled_dir, args.labeled_dir, args.label_col, wb
        )
        encoder, contrastive_encoder = pretrain(
            unlabeled,
            gnn_type=args.gnn_type,
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            ema_decay=args.ema_decay,
            batch_size=args.batch_size,
            mask_ratio=args.mask_ratio,
            lr=args.lr,
            epochs=args.pretrain_epochs,
            device=args.device,
            devices=args.devices,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
            contrastive=args.contrastive,
            wb=wb,
        )
        metrics, contrastive_metrics = finetune(
            labeled,
            encoder=encoder,
            task_type=args.task_type,
            epochs=args.finetune_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            patience=args.patience,
            devices=args.devices,
            contrastive_encoder=contrastive_encoder,
            wb=wb,
        )
        evaluate(metrics, contrastive_metrics, wb)
    except DataLoadError:
        sys.exit(DATA_LOAD_ERROR)
    except PretrainError:
        sys.exit(PRETRAIN_ERROR)
    except FinetuneError:
        sys.exit(FINETUNE_ERROR)
    except EvalError:
        sys.exit(EVAL_ERROR)
    finally:
        wb.finish()


if __name__ == "__main__":  # pragma: no cover
    main()

