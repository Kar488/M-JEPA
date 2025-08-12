"""End-to-end JEPA training and evaluation pipeline.

This script wires together the dataset loaders, encoder factory and training
utilities to provide a minimal yet real example of the JEPA workflow:

1. Load an **unlabelled** dataset for self‑supervised pretraining.
2. Pretrain an encoder using the JEPA objective.
3. Load a **labelled** dataset and fine‑tune a linear head.
4. Evaluate the head and report metrics.

The command line interface exposes these stages as separate subcommands so
that deployment pipelines can invoke them independently::

    python scripts/train_jepa.py pretrain --unlabeled-dir <path> [opts]
    python scripts/train_jepa.py finetune --labeled-dir <path> --encoder encoder.pt [opts]
    python scripts/train_jepa.py evaluate --labeled-dir <path> --encoder encoder.pt [opts]

Each major step is logged to Weights & Biases when enabled. Distinct exit
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


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
    
    # Training hyperparameters
    p.add_argument("--gnn-type", type=str, default="mpnn", help="Encoder architecture")
    p.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size")
    p.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers")
    p.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay rate")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs for DDP")

    # W&B options
    p.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="m-jepa")
    p.add_argument("--wandb-tags", nargs="*", default=None)

    # Pipeline stage control
    p.add_argument("--stage", choices=["pretrain", "finetune", "eval"], default="eval",
        help="Run pipeline up to this stage",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser with subcommands."""

    parser = argparse.ArgumentParser(description="JEPA training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # Pretrain
    # ------------------------------------------------------------------
    pre = sub.add_parser("pretrain", help="Self-supervised pretraining")
    pre.add_argument("--unlabeled-dir", required=True, help="Directory of unlabeled graphs")
    pre.add_argument("--mask-ratio", type=float, default=0.15, help="Masking ratio for JEPA")
    pre.add_argument("--epochs", type=int, default=1, help="Number of pretraining epochs")
    pre.add_argument(
        "--output",
        type=str,
        default="encoder.pt",
        help=(
            "Where to save JEPA encoder weights; if --contrastive is set,"
            " contrastive weights are written to <output>_contrastive.pt"
        ),
    )
    pre.add_argument("--contrastive", action="store_true", help="Also run contrastive baseline")
    pre.add_argument("--aug-rotate", action="store_true", help="Randomly rotate coordinates")
    pre.add_argument("--aug-mask-angle", action="store_true", help="Mask bond angles")
    pre.add_argument("--aug-dihedral", action="store_true", help="Perturb dihedral angles")
    _add_common_args(pre)
    pre.set_defaults(func=cmd_pretrain)

    # ------------------------------------------------------------------
    # Finetune
    # ------------------------------------------------------------------
    ft = sub.add_parser("finetune", help="Fine-tune a linear head on labelled data")
    ft.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs")
    ft.add_argument("--label-col", type=str, default="label", help="Label column name")
    ft.add_argument("--encoder", required=True, help="Path to pretrained encoder checkpoint")
    ft.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    ft.add_argument("--epochs", type=int, default=1, help="Number of fine-tuning epochs")
    ft.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    _add_common_args(ft)
    ft.set_defaults(func=cmd_finetune)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    ev = sub.add_parser("evaluate", help="Evaluate a pretrained encoder with a linear probe")
    ev.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs")
    ev.add_argument("--label-col", type=str, default="label", help="Label column name")
    ev.add_argument("--encoder", required=True, help="Path to pretrained encoder checkpoint")
    ev.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    ev.add_argument("--epochs", type=int, default=1, help="Number of probe training epochs")
    ev.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    _add_common_args(ev)
    ev.set_defaults(func=cmd_evaluate)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    bench = sub.add_parser(
        "benchmark", help="Compare JEPA and contrastive encoders on labelled data"
    )
    bench.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs")
    bench.add_argument("--label-col", type=str, default="label", help="Label column name")
    bench.add_argument("--jepa-encoder", required=True, help="Path to JEPA encoder checkpoint")
    bench.add_argument(
        "--contrastive-encoder",
        required=False,
        help="Path to contrastive encoder checkpoint",
    )
    bench.add_argument(
        "--task-type", choices=["classification", "regression"], default="classification"
    )
    bench.add_argument("--epochs", type=int, default=1, help="Number of probe training epochs")
    bench.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    _add_common_args(bench)
    bench.set_defaults(func=cmd_benchmark)

    # ------------------------------------------------------------------
    # Tox21 case study
    # ------------------------------------------------------------------
    tox = sub.add_parser("tox21", help="Run Tox21 case-study ranking experiment")
    tox.add_argument("--csv", required=True, help="Path to Tox21 CSV with SMILES and labels")
    tox.add_argument("--task", required=True, help="Column name of toxicity task")
    tox.add_argument("--pretrain-epochs", type=int, default=5, help="JEPA pretrain epochs")
    tox.add_argument("--finetune-epochs", type=int, default=20, help="Regression head epochs")
    tox.add_argument(
        "--num-top-exclude", type=int, default=10, help="Top-k toxic molecules to exclude"
    )
    tox.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    _add_common_args(tox)
    tox.set_defaults(func=cmd_tox21)

    return parser

# Distinct exit codes so GitHub Actions can surface which step failed
DATA_LOAD_ERROR = 1
PRETRAIN_ERROR = 2
FINETUNE_ERROR = 3
EVAL_ERROR = 4
CASE_STUDY_ERROR = 5
BENCHMARK_ERROR = 6


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


class EvaluateError(RuntimeError):
    """Raised when evaluation fails."""


class CaseStudyError(RuntimeError):
    """Raised when the Tox21 case study fails."""


class BenchmarkError(RuntimeError):
    """Raised when benchmarking fails."""

def cmd_pretrain(args: argparse.Namespace) -> None:

    """Run the self-supervised pretraining stage.""" 

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "unlabeled_dir": args.unlabeled_dir,
            "gnn_type": args.gnn_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "mask_ratio": args.mask_ratio,
            "ema_decay": args.ema_decay,
        },
    )
        
    try:
        unlabeled = load_directory_dataset(args.unlabeled_dir)
        wb.log({"phase": "data_load", "unlabeled_graphs": len(unlabeled)})
    except Exception:
        logger.exception("Failed to load unlabeled dataset")
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

    contrastive_encoder = None
    if args.contrastive:
        contrastive_encoder = build_encoder(
            gnn_type=args.gnn_type,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            edge_dim=edge_dim,
        )

    try:
        wb.log({"phase": "pretrain", "status": "start"})   
        train_jepa(
            dataset=unlabeled,
            encoder=encoder,
            ema_encoder=ema_encoder,
            predictor=predictor,
            ema=ema,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mask_ratio=args.mask_ratio,
            random_rotate=args.aug_rotate,
            mask_angle=args.aug_mask_angle,
            perturb_dihedral=args.aug_dihedral,
            lr=args.lr,
            device=args.device,
            devices=args.devices,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
        )
        wb.log({"phase": "pretrain", "status": "success"})
    except Exception:
        logger.exception("Pretraining failed")
        wb.log({"phase": "pretrain", "status": "error"})
        sys.exit(PRETRAIN_ERROR)

    if args.contrastive and contrastive_encoder is not None:
        try:
            wb.log({"phase": "pretrain_contrastive", "status": "start"})
            train_contrastive(
                dataset=unlabeled,
                encoder=contrastive_encoder,
                epochs=args.epochs,
                batch_size=args.batch_size,
                mask_ratio=args.mask_ratio,
                random_rotate=args.aug_rotate,
                mask_angle=args.aug_mask_angle,
                perturb_dihedral=args.aug_dihedral,
                lr=args.lr,
                device=args.device,
                devices=args.devices,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_tags=args.wandb_tags,
            )
            wb.log({"phase": "pretrain_contrastive", "status": "success"})
        except Exception:
            logger.exception("Contrastive pretraining failed")
            wb.log({"phase": "pretrain_contrastive", "status": "error"})
            sys.exit(PRETRAIN_ERROR)

    torch.save({"encoder": encoder.state_dict()}, args.output)
    if args.contrastive and contrastive_encoder is not None:
        contrastive_path = f"{os.path.splitext(args.output)[0]}_contrastive.pt"
        torch.save({"encoder": contrastive_encoder.state_dict()}, contrastive_path)
        wb.log({"contrastive_checkpoint": contrastive_path})
    wb.finish()

def cmd_finetune(args: argparse.Namespace) -> None:
    """Run the fine-tuning stage on labelled data."""

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "labeled_dir": args.labeled_dir,
            "gnn_type": args.gnn_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "ema_decay": args.ema_decay,
            "task_type": args.task_type,
        },
    )

    try:
        labeled = load_directory_dataset(args.labeled_dir, label_col=args.label_col)
        wb.log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset")
        wb.log({"phase": "data_load", "status": "error"})
        sys.exit(DATA_LOAD_ERROR)

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = None if labeled.graphs[0].edge_attr is None else labeled.graphs[0].edge_attr.shape[1]
    encoder = build_encoder(
        gnn_type=args.gnn_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=edge_dim,
    )
    state = torch.load(args.encoder, map_location=args.device)
    if isinstance(state, dict) and "encoder" in state:
        encoder.load_state_dict(state["encoder"])
    else:
        encoder.load_state_dict(state)

    """Fine-tune linear head on labelled data."""

    
    try:
        wb.log({"phase": "finetune", "status": "start"})
        metrics: Dict[str, float] = train_linear_head(
            dataset=labeled,
            encoder=encoder,
            task_type=args.task_type,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            patience=args.patience,
            devices=args.devices,
        )
        wb.log({"phase": "finetune", "status": "success"})
        for k, v in metrics.items():
            if k != "head":
                wb.log({f"metric/{k}": v})
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Fine-tuning failed")
        wb.log({"phase": "finetune", "status": "error"})
        raise FinetuneError(FINETUNE_ERROR)

    wb.finish()


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a pretrained encoder on a labelled dataset."""

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "labeled_dir": args.labeled_dir,
            "gnn_type": args.gnn_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "ema_decay": args.ema_decay,
            "task_type": args.task_type,
        },
    )

    try:
        labeled = load_directory_dataset(args.labeled_dir, label_col=args.label_col)
        wb.log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset")
        wb.log({"phase": "data_load", "status": "error"})
        raise DataLoadError("failed to load labelled dataset")

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = None if labeled.graphs[0].edge_attr is None else labeled.graphs[0].edge_attr.shape[1]
    encoder = build_encoder(
        gnn_type=args.gnn_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=edge_dim,
    )
    state = torch.load(args.encoder, map_location=args.device)
    if isinstance(state, dict) and "encoder" in state:
        encoder.load_state_dict(state["encoder"])
    else:
        encoder.load_state_dict(state)

    try:
        wb.log({"phase": "evaluate", "status": "start"})
        metrics = train_linear_head(
            dataset=labeled,
            encoder=encoder,
            task_type=args.task_type,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            patience=args.patience,
            devices=args.devices,
        )
        wb.log({"phase": "evaluate", "status": "success"})
        metrics.pop("head", None)
        for k, v in metrics.items():
            wb.log({f"metric/{k}": v})
        logger.info("Evaluation metrics: %s", metrics)
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Evaluation failed")
        wb.log({"phase": "evaluate", "status": "error"})
        raise EvaluateError(str(e))
    finally:
        wb.finish()


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Compare JEPA and contrastive encoders on the same labelled dataset."""

    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={
            "labeled_dir": args.labeled_dir,
            "task_type": args.task_type,
            "epochs": args.epochs,
        },
    )

    try:
        labeled = load_directory_dataset(args.labeled_dir, label_col=args.label_col)
        wb.log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset")
        wb.log({"phase": "data_load", "status": "error"})
        raise DataLoadError("failed to load labelled dataset")

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = None if labeled.graphs[0].edge_attr is None else labeled.graphs[0].edge_attr.shape[1]

    results: Dict[str, Dict[str, float]] = {}
    try:
        wb.log({"phase": "benchmark", "status": "start"})
        # JEPA evaluation
        jepa_encoder = build_encoder(
            gnn_type=args.gnn_type,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            edge_dim=edge_dim,
        )
        state = torch.load(args.jepa_encoder, map_location=args.device)
        if isinstance(state, dict) and "encoder" in state:
            jepa_encoder.load_state_dict(state["encoder"])
        else:
            jepa_encoder.load_state_dict(state)
        metrics_j = train_linear_head(
            dataset=labeled,
            encoder=jepa_encoder,
            task_type=args.task_type,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            patience=args.patience,
            devices=args.devices,
        )
        metrics_j.pop("head", None)
        results["jepa"] = metrics_j

        # Contrastive baseline
        if args.contrastive_encoder:
            cont_encoder = build_encoder(
                gnn_type=args.gnn_type,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                edge_dim=edge_dim,
            )
            state_c = torch.load(args.contrastive_encoder, map_location=args.device)
            if isinstance(state_c, dict) and "encoder" in state_c:
                cont_encoder.load_state_dict(state_c["encoder"])
            else:
                cont_encoder.load_state_dict(state_c)
            metrics_c = train_linear_head(
                dataset=labeled,
                encoder=cont_encoder,
                task_type=args.task_type,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=args.device,
                patience=args.patience,
                devices=args.devices,
            )
            metrics_c.pop("head", None)
            results["contrastive"] = metrics_c

        wb.log({"phase": "benchmark", "status": "success"})
        for method, mets in results.items():
            for k, v in mets.items():
                wb.log({f"{method}/{k}": v})
        logger.info("Benchmark results: %s", results)
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Benchmarking failed")
        wb.log({"phase": "benchmark", "status": "error"})
        raise BenchmarkError(str(e))
    finally:
        wb.finish()


def cmd_tox21(args: argparse.Namespace) -> None:
    """Run the Tox21 case study and log results."""

    from experiments.case_study import run_tox21_case_study

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
            device=args.device,
        )
        wb.log({
            "phase": "tox21",
            "status": "success",
            "mean_true": mean_true,
            "mean_random_after": mean_random,
            "mean_jepa_after": mean_jepa,
        })
        for name, val in baseline_means.items():
            wb.log({f"baseline/{name}": val})
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Tox21 case study failed")
        wb.log({"phase": "tox21", "status": "error"})
        raise CaseStudyError(str(e))
    finally:
        wb.finish()


def main() -> None:  # pragma: no cover - CLI entry
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except DataLoadError:
        sys.exit(DATA_LOAD_ERROR)
    except PretrainError:
        sys.exit(PRETRAIN_ERROR)
    except FinetuneError:
        sys.exit(FINETUNE_ERROR)
    except EvaluateError:
        sys.exit(EVAL_ERROR)
    except CaseStudyError:
        sys.exit(CASE_STUDY_ERROR)
    except BenchmarkError:
        sys.exit(BENCHMARK_ERROR)

