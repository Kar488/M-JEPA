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
from training.unsupervised import train_jepa  # noqa: E402
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

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    try:
        unlabeled = load_directory_dataset(args.unlabeled_dir)
        labeled = load_directory_dataset(args.labeled_dir, label_col=args.label_col)
        wb.log(
            {
                "phase": "data_load",
                "unlabeled_graphs": len(unlabeled),
                "labeled_graphs": len(labeled),
            }
        )
    except Exception:  # pragma: no cover - defensive, we want a clear exit
        logger.exception("Failed to load datasets")
        wb.log({"phase": "data_load", "status": "error"})
        sys.exit(DATA_LOAD_ERROR)

    input_dim = unlabeled.graphs[0].x.shape[1]
    edge_dim = None if unlabeled.graphs[0].edge_attr is None else unlabeled.graphs[0].edge_attr.shape[1]

    encoder, ema_encoder = _init_encoder(
        gnn_type=args.gnn_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=edge_dim,
    )

    ema = EMA(encoder, decay=args.ema_decay)
    predictor = MLPPredictor(embed_dim=args.hidden_dim, hidden_dim=args.hidden_dim * 2)

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------
    try:
        wb.log({"phase": "pretrain", "status": "start"})
        losses = train_jepa(
            dataset=unlabeled,
            encoder=encoder,
            ema_encoder=ema_encoder,
            predictor=predictor,
            ema=ema,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            mask_ratio=args.mask_ratio,
            lr=args.lr,
            device=args.device,
            devices=args.devices,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
        )
        final_loss = losses[-1] if losses else None
        wb.log({"phase": "pretrain", "status": "success", "final_loss": final_loss})
    except Exception:  # pragma: no cover - defensive, we want a clear exit
        logger.exception("Pretraining failed")
        wb.log({"phase": "pretrain", "status": "error"})
        sys.exit(PRETRAIN_ERROR)

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------
    try:
        wb.log({"phase": "finetune", "status": "start"})
        metrics: Dict[str, float] = train_linear_head(
            dataset=labeled,
            encoder=encoder,
            task_type=args.task_type,
            epochs=args.finetune_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            patience=args.patience,
            devices=args.devices,
        )
        wb.log({"phase": "finetune", "status": "success"})
    except Exception:  # pragma: no cover - defensive
        logger.exception("Fine-tuning failed")
        wb.log({"phase": "finetune", "status": "error"})
        sys.exit(FINETUNE_ERROR)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    try:
        if metrics:
            wb.log({"phase": "evaluation", **{f"metric/{k}": v for k, v in metrics.items()}})
    except Exception:  # pragma: no cover
        logger.exception("Evaluation logging failed")
        wb.log({"phase": "evaluation", "status": "error"})
        sys.exit(EVAL_ERROR)

    wb.finish()


if __name__ == "__main__":  # pragma: no cover
    main()

