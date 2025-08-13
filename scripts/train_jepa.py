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
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml

# Attempt to import reusable components from the package.
try:
    from main import load_directory_dataset
except Exception:
    load_directory_dataset = None  # type: ignore[assignment] 

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

try:
    from training.unsupervised import train_jepa, train_contrastive  # type: ignore[assignment]
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

    # Load unlabeled dataset
    try:
        unlabeled = load_directory_dataset(args.unlabeled_dir, add_3d=args.add_3d)  # type: ignore[arg-type]
        wb.log({"phase": "data_load", "unlabeled_graphs": len(unlabeled)})
    except Exception:
        logger.exception("Failed to load unlabeled dataset")
        wb.log({"phase": "data_load", "status": "error"})
        sys.exit(1)

    input_dim = unlabeled.graphs[0].x.shape[1]
    edge_dim = None if unlabeled.graphs[0].edge_attr is None else unlabeled.graphs[0].edge_attr.shape[1]
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

    # Pretrain JEPA
    try:
        wb.log({"phase": "pretrain", "status": "start"})
        train_jepa(
            dataset=unlabeled,
            encoder=encoder,
            ema_encoder=ema_encoder,
            predictor=predictor,
            ema=ema_helper,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mask_ratio=args.mask_ratio,
            contiguous=args.contiguous,
            lr=args.lr,
            device=device,
            reg_lambda=1e-4,
            random_rotate=args.aug_rotate,
            mask_angle=args.aug_mask_angle,
            perturb_dihedral=args.aug_dihedral,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
        )
        wb.log({"phase": "pretrain", "status": "success"})
    except Exception:
        logger.exception("JEPA pretraining failed")
        wb.log({"phase": "pretrain", "status": "error"})
        sys.exit(2)

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
                wandb_project=args.wandb_project,
                wandb_tags=args.wandb_tags,
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
    """Fine‑tune a linear head on labelled data across multiple seeds."""
    if load_directory_dataset is None or build_encoder is None or train_linear_head is None:
        logger.error("Fine‑tuning modules are unavailable.")
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
        labeled = load_directory_dataset(args.labeled_dir, label_col=args.label_col, add_3d=args.add_3d)  # type: ignore[arg-type]
        wb.log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset")
        wb.log({"phase": "data_load", "status": "error"})
        sys.exit(1)

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = None if labeled.graphs[0].edge_attr is None else labeled.graphs[0].edge_attr.shape[1]
    device = resolve_device(args.device)

    # Aggregate metrics across seeds
    metrics_runs: List[Dict[str, float]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        encoder = build_encoder(
            gnn_type=args.gnn_type,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            edge_dim=edge_dim,
        )
        # Load checkpoint
        state = torch.load(args.encoder, map_location=device)
        if isinstance(state, dict) and "encoder" in state:
            encoder.load_state_dict(state["encoder"])
        else:
            encoder.load_state_dict(state)

        try:
            wb.log({"phase": f"finetune_{seed}", "status": "start"})
            metrics = train_linear_head(
                dataset=labeled,
                encoder=encoder,
                task_type=args.task_type,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
                patience=args.patience,
                devices=args.devices,
            )
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
    # Reuse finetune implementation with a different default config section
    cmd_finetune(args)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Compare JEPA and contrastive encoders on the same labelled dataset.

    Runs training across seeds and reports which method yields better
    performance based on ROC‑AUC (classification) or RMSE (regression).
    """
    if load_directory_dataset is None or build_encoder is None or train_linear_head is None:
        logger.error("Benchmark modules are unavailable.")
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

    try:
        labeled = load_directory_dataset(args.labeled_dir, label_col=args.label_col, add_3d=args.add_3d)  # type: ignore[arg-type]
        wb.log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset for benchmarking")
        wb.log({"phase": "data_load", "status": "error"})
        sys.exit(1)

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = None if labeled.graphs[0].edge_attr is None else labeled.graphs[0].edge_attr.shape[1]
    device = resolve_device(args.device)

    # Prepare results dict
    all_results: Dict[str, Dict[str, float]] = {}

    def evaluate_encoder(ckpt_path: str, method_name: str) -> Dict[str, float]:
        """Evaluate a specific encoder checkpoint across seeds."""
        metrics_runs: List[Dict[str, float]] = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            enc = build_encoder(
                gnn_type=args.gnn_type,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                edge_dim=edge_dim,
            )
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "encoder" in state:
                enc.load_state_dict(state["encoder"])
            else:
                enc.load_state_dict(state)
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
        # Log to wandb
        for k, v in agg.items():
            wb.log({f"{method_name}/{k}": v})
        return agg

    wb.log({"phase": "benchmark", "status": "start"})
    # Evaluate JEPA
    agg_jepa = evaluate_encoder(args.jepa_encoder, "jepa")
    all_results["jepa"] = agg_jepa

    # Evaluate contrastive
    agg_cont: Dict[str, float] = {}
    if args.contrastive_encoder:
        agg_cont = evaluate_encoder(args.contrastive_encoder, "contrastive")
        all_results["contrastive"] = agg_cont

    # Decide which is better
    verdict = "jepa"
    if agg_cont:
        # Choose metric based on task
        if args.task_type == "classification":
            # Higher AUC/ACC is better
            key = "roc_auc_mean" if "roc_auc_mean" in agg_jepa else ("acc_mean" if "acc_mean" in agg_jepa else None)
            if key and agg_cont.get(key, float('-inf')) > agg_jepa.get(key, float('-inf')):
                verdict = "contrastive"
        else:
            # Lower RMSE/MAE is better
            key = "rmse_mean" if "rmse_mean" in agg_jepa else ("mae_mean" if "mae_mean" in agg_jepa else None)
            if key and agg_cont.get(key, float('inf')) < agg_jepa.get(key, float('inf')):
                verdict = "contrastive"

    wb.log({"phase": "benchmark", "status": "success", "best_method": verdict})
    logger.info(f"Benchmark completed. Best method: {verdict}")
    wb.finish()


def cmd_tox21(args: argparse.Namespace) -> None:
    """Run the Tox21 ranking case study."""
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
        logger.exception("Tox21 case study failed")
        wb.log({"phase": "tox21", "status": "error"})
        sys.exit(5)
    finally:
        wb.finish()


def cmd_grid_search(args: argparse.Namespace) -> None:
    """Run a hyper‑parameter sweep using the ``run_grid_search`` helper.

    This command loads the specified dataset using ``load_directory_dataset`` and
    then performs a grid search over various JEPA hyper‑parameters.  The search
    space is configurable via CLI flags.  Results are logged to Weights &
    Biases if enabled and optionally written to a CSV file.  When the grid
    search completes, the best configuration and its metric are reported.
    """
    # If the experiments module is unavailable, abort with a distinct exit code
    if run_grid_search is None:
        logger.error("Grid search functionality is unavailable. Install the experiments package or check the import.")
        sys.exit(7)

    # Convert numerical lists to tuples and boolean flags
    contiguities = tuple(bool(c) for c in args.contiguities)
    add_3d_opts = tuple(bool(a) for a in args.add_3d_options)
    seeds: tuple
    # Determine seeds: use CLI if provided, otherwise fall back to configuration defaults
    if args.seeds is not None and len(args.seeds) > 0:
        seeds = tuple(args.seeds)
    else:
        seeds = tuple(CONFIG.get("finetune", {}).get("seeds", [42, 123, 456]))

    # Create dataset loader closures for run_grid_search.  We support three
    # scenarios:
    #   1. A unified dataset (--dataset-dir) used for both pretraining and evaluation.
    #   2. Separate unlabeled and labeled datasets (--unlabeled-dir and/or --labeled-dir).
    # A function must accept ``add_3d`` and return a GraphDataset.  If only
    # unlabeled or labeled is provided, the missing one falls back to the
    # unified dataset if available.
    _dataset_fn = None
    _unlabeled_fn = None
    _eval_fn = None
    if args.dataset_dir:
        def _dataset_fn(add_3d: bool = False):  # type: ignore[override]
            return load_directory_dataset(
                args.dataset_dir,
                label_col=args.label_col,
                add_3d=add_3d,
            )
    # Define unlabeled and eval loaders if specified
    if args.unlabeled_dir:
        def _unlabeled_fn(add_3d: bool = False):  # type: ignore[override]
            return load_directory_dataset(
                args.unlabeled_dir,
                add_3d=add_3d,
            )
    if args.labeled_dir:
        def _eval_fn(add_3d: bool = False):  # type: ignore[override]
            return load_directory_dataset(
                args.labeled_dir,
                label_col=args.label_col,
                add_3d=add_3d,
            )

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
    wb.log({"phase": "grid_search", "status": "start"})

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
        )
        # Log each row to W&B for comprehensive visualisation.  We assign a
        # unique identifier to each configuration using its index.  This
        # produces a separate log entry per configuration, enabling plots and
        # tables in the W&B UI.
        best_conf = None
        if df is not None and not df.empty:
            for idx, row in df.iterrows():
                # Prepare a metrics dict excluding non‑numeric entries and
                # include the index as "config_id".  Flatten any lists or
                # arrays to scalars when possible.
                metrics_dict = {"config_id": int(idx)}
                for col, val in row.items():
                    if isinstance(val, (list, tuple)) and len(val) == 1:
                        val = val[0]
                    metrics_dict[col] = val
                wb.log(metrics_dict)
            best_conf = df.iloc[-1].to_dict()
            logger.info("Grid search completed. Best configuration: %s", best_conf)
        else:
            logger.info("Grid search returned no results.")
        wb.log({"phase": "grid_search", "status": "success", "best": best_conf})
    except Exception:
        logger.exception("Grid search failed")
        wb.log({"phase": "grid_search", "status": "error"})
        # exit with distinct code for grid search failures
        sys.exit(7)
    finally:
        wb.finish()


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser, section: str) -> None:
    """Add arguments common to multiple commands using defaults from the given config section."""
    # Model hyperparameters
    model_cfg = CONFIG.get("model", {})
    p.add_argument("--gnn-type", type=str, default=model_cfg.get("gnn_type", "mpnn"), help="GNN encoder type")
    p.add_argument("--hidden-dim", type=int, default=model_cfg.get("hidden_dim", 64), help="Hidden dimension size")
    p.add_argument("--num-layers", type=int, default=model_cfg.get("num_layers", 2), help="Number of GNN layers")
    p.add_argument("--ema-decay", type=float, default=model_cfg.get("ema_decay", 0.99), help="EMA decay rate")
    # Data augmentations and options
    p.add_argument("--add-3d", action="store_true", help="Augment with 3D coordinate featurisation")
    p.add_argument("--contiguous", action="store_true", help="Use contiguous subgraph masking (JEPA)")
    p.add_argument("--aug-rotate", action="store_true", help="Randomly rotate coordinates during pretraining")
    p.add_argument("--aug-mask-angle", action="store_true", help="Mask bond angles during pretraining")
    p.add_argument("--aug-dihedral", action="store_true", help="Perturb dihedral angles during pretraining")
    # Optimisation
    sec_cfg = CONFIG.get(section, {})
    p.add_argument("--epochs", type=int, default=sec_cfg.get("epochs", 1), help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=sec_cfg.get("batch_size", 32), help="Batch size")
    p.add_argument("--lr", type=float, default=sec_cfg.get("lr", 1e-3), help="Learning rate")
    # Seeds for downstream evaluation
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="Random seeds for averaging results")
    # Device & DDP
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs for DDP")
    # W&B
    wandb_cfg = CONFIG.get("wandb", {})
    p.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default=wandb_cfg.get("project", "m-jepa"), help="W&B project name")
    p.add_argument("--wandb-tags", nargs="*", default=wandb_cfg.get("tags", []), help="Tags for W&B run")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with subcommands."""
    parser = argparse.ArgumentParser(description="JEPA training and evaluation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # Pretrain subcommand
    pre = sub.add_parser("pretrain", help="Self‑supervised pretraining")
    pre.add_argument("--unlabeled-dir", required=True, help="Directory of unlabeled graphs (.parquet or .csv)")
    pre.add_argument("--output", type=str, default="encoder.pt", help="Where to save the JEPA encoder checkpoint")
    pre.add_argument("--contrastive", action="store_true", help="Also run a contrastive baseline")
    _add_common_args(pre, "pretrain")
    pre.set_defaults(func=cmd_pretrain)

    # Fine‑tune subcommand
    ft = sub.add_parser("finetune", help="Fine‑tune a linear head on labelled data")
    ft.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs (.parquet or .csv)")
    ft.add_argument("--label-col", type=str, default="label", help="Label column name in input files")
    ft.add_argument("--encoder", required=True, help="Path to a pretrained encoder checkpoint (.pt)")
    ft.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    ft.add_argument("--patience", type=int, default=CONFIG.get("finetune", {}).get("patience", 10), help="Early stopping patience")
    _add_common_args(ft, "finetune")
    ft.set_defaults(func=cmd_finetune)

    # Evaluate subcommand (alias for finetune)
    ev = sub.add_parser("evaluate", help="Evaluate a pretrained encoder via a linear probe")
    ev.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs")
    ev.add_argument("--label-col", type=str, default="label", help="Label column name")
    ev.add_argument("--encoder", required=True, help="Path to a pretrained encoder checkpoint (.pt)")
    ev.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    ev.add_argument("--patience", type=int, default=CONFIG.get("evaluate", {}).get("patience", 10), help="Early stopping patience")
    _add_common_args(ev, "evaluate")
    ev.set_defaults(func=cmd_evaluate)

    # Benchmark subcommand
    bench = sub.add_parser("benchmark", help="Compare JEPA and contrastive encoders on labelled data")
    bench.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs")
    bench.add_argument("--label-col", type=str, default="label", help="Label column name")
    bench.add_argument("--jepa-encoder", required=True, help="Path to a JEPA encoder checkpoint (.pt)")
    bench.add_argument("--contrastive-encoder", required=False, help="Path to a contrastive encoder checkpoint (.pt)")
    bench.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    bench.add_argument("--patience", type=int, default=CONFIG.get("benchmark", {}).get("patience", 10), help="Early stopping patience")
    _add_common_args(bench, "benchmark")
    bench.set_defaults(func=cmd_benchmark)

    # Tox21 case study
    tox = sub.add_parser("tox21", help="Run the Tox21 case study experiment")
    tox.add_argument("--csv", required=True, help="Path to the Tox21 CSV containing SMILES and labels")
    tox.add_argument("--task", required=True, help="Name of the toxicity column to predict")
    case_cfg = CONFIG.get("case_study", {})
    tox.add_argument("--pretrain-epochs", type=int, default=case_cfg.get("pretrain_epochs", 5), help="JEPA pretrain epochs for case study")
    tox.add_argument("--finetune-epochs", type=int, default=case_cfg.get("finetune_epochs", 20), help="Epochs to train regression head in case study")
    tox.add_argument("--num-top-exclude", type=int, default=case_cfg.get("num_top_exclude", 10), help="Top‑k toxic compounds to exclude when ranking")
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
    # Datasets for the sweep.  At least one of --dataset-dir or the pair
    # (--unlabeled-dir, --labeled-dir) must be provided.  If only
    # --dataset-dir is given it is used for both pretraining and evaluation.
    grid.add_argument(
        "--dataset-dir",
        required=False,
        default=None,
        help="Path to a graph dataset used for both pretraining and evaluation."
             " If omitted, you must specify --unlabeled-dir and/or --labeled-dir.",
    )
    grid.add_argument(
        "--unlabeled-dir",
        type=str,
        default=None,
        help="Directory of an unlabeled graph dataset for JEPA pretraining (e.g. ZINC/PubChem).",
    )
    grid.add_argument(
        "--labeled-dir",
        type=str,
        default=None,
        help="Directory of a labeled graph dataset for downstream evaluation (e.g. MoleculeNet).",
    )
    grid.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of the label column in the dataset (ignored for unlabeled data)",
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
    # Search space parameters
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
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging for the grid search",
    )
    # Additional options to mirror other subcommands
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
    grid.set_defaults(func=cmd_grid_search)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.error("No subcommand provided")
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)

