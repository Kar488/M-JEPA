# flake8: noqa
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import log_effective_gnn

# Ensure PyTorch uses filesystem-backed shared memory objects instead of file
# descriptors for inter-process tensor sharing.  The default "file_descriptor"
# strategy opens a unique FD for every shared tensor created by DataLoader
# workers, which eventually exhausts the per-process file descriptor quota
# during long fine-tuning runs.  Switching to "file_system" prevents the
# descriptor leak that triggers "Too many open files" crashes.
torch.multiprocessing.set_sharing_strategy("file_system")

from data.mdataset import GraphData
from utils.graph_ops import _encode_graph
from utils.pooling import global_mean_pool

logger = logging.getLogger(__name__)

_GNN_TYPES_REQUIRING_3D = {"schnet3d", "schnet"}


def _stage_outputs_dir() -> Optional[Path]:
    stage_dir = os.getenv("STAGE_OUTPUTS_DIR")
    if not stage_dir:
        return None
    try:
        path = Path(stage_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def _record_finetune_stage_outputs(payload: Dict[str, Any]) -> None:
    stage_dir = _stage_outputs_dir()
    if stage_dir is None:
        return
    try:
        out_path = stage_dir / "finetune.json"
        tmp_path = out_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(out_path)
    except Exception:
        logger.warning("Failed to write finetune stage outputs", exc_info=True)


def _sanitize_alias(raw: str) -> str:
    safe = [c if c.isalnum() or c in {"-", "_", "."} else "_" for c in raw]
    text = "".join(safe).strip("._")
    return text or "run"


def _maybe_enable_add_3d(args: argparse.Namespace) -> bool:
    """Ensure SchNet-based encoders always receive 3-D coordinates."""

    gnn_type = str(getattr(args, "gnn_type", "") or "").lower()
    requires_3d = gnn_type in _GNN_TYPES_REQUIRING_3D
    if requires_3d and not getattr(args, "add_3d", False):
        logger.info(
            "GNN '%s' requires 3D coordinates; enabling --add-3d automatically.",
            getattr(args, "gnn_type", gnn_type),
        )
        setattr(args, "add_3d", True)
    return requires_3d


def _ensure_dataset_has_pos(dataset) -> None:
    """Validate that a dataset provides ``pos`` coordinates when required."""

    graphs = getattr(dataset, "graphs", None)
    if not graphs:
        return

    for idx, graph in enumerate(graphs):
        pos = getattr(graph, "pos", None)
        if pos is None:
            num_nodes = 0
            if hasattr(graph, "num_nodes"):
                try:
                    num_nodes = int(graph.num_nodes())
                except Exception:
                    num_nodes = 0
            if not num_nodes:
                x_field = getattr(graph, "x", None)
                try:
                    num_nodes = int(len(x_field)) if x_field is not None else 0
                except Exception:
                    num_nodes = 0
            if num_nodes == 0:
                continue
            raise ValueError(
                "SchNet3D requires 3D coordinates `pos`; graph %d is missing them. "
                "Clear cached datasets or rebuild with --add-3d."
                % idx,
            )
        break


def _iter_trainable_params(model) -> List[nn.Parameter]:
    params = getattr(model, "parameters", None)
    if callable(params):
        try:
            return list(params())
        except Exception:
            return []

    for attr in ("encoder", "module", "backbone"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        sub_params = getattr(sub, "parameters", None)
        if callable(sub_params):
            try:
                return list(sub_params())
            except Exception:
                return []
    return []


def _configure_encoder_trainability(
    encoder: nn.Module,
    *,
    freeze_encoder: bool,
    unfreeze_top_layers: int,
    unfreeze_mode: str = "none",
) -> List[nn.Parameter]:
    """Apply fine-tuning freeze/unfreeze policy and return trainable params."""

    if not isinstance(encoder, nn.Module):
        return []

    params_fn = getattr(encoder, "parameters", None)
    if not callable(params_fn):
        return []

    try:
        all_params = list(params_fn())
    except Exception:
        return []

    mode = str(unfreeze_mode or "none").lower()
    if mode == "full":
        freeze_encoder = False
        unfreeze_top_layers = 0
    elif mode == "partial" and freeze_encoder:
        modules = list(encoder.children()) or [encoder]
        if unfreeze_top_layers <= 0:
            unfreeze_top_layers = max(1, min(len(modules), 1))
    if not freeze_encoder and unfreeze_top_layers <= 0:
        for param in all_params:
            param.requires_grad = True
        return all_params

    for param in all_params:
        param.requires_grad = False

    if not freeze_encoder and unfreeze_top_layers > 0:
        for param in all_params:
            param.requires_grad = True
        return all_params

    if unfreeze_top_layers < 0:
        for param in all_params:
            param.requires_grad = True
        return all_params

    if unfreeze_top_layers == 0:
        return []

    modules = list(encoder.children())
    if not modules:
        modules = [encoder]
    selected = modules[-unfreeze_top_layers:]
    seen: set[int] = set()
    trainable: List[nn.Parameter] = []
    for module in selected:
        params_fn = getattr(module, "parameters", None)
        if not callable(params_fn):
            continue
        try:
            module_params = list(params_fn())
        except Exception:
            continue
        for param in module_params:
            param.requires_grad = True
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                trainable.append(param)
    return trainable


def cmd_finetune(args: argparse.Namespace) -> None:
    """Fine‑tune a linear head on labelled data across multiple seeds resume & checkpoints."""
    logger.info("Starting finetune with args: %s", args)

    from utils.checkpoint import compute_state_dict_hash, load_checkpoint, save_checkpoint

    try:
        from ..utils.checkpoint import (
            load_state_dict_forgiving as _load_state_dict_forgiving,  # type: ignore[import-not-found]
        )
        from ..utils.checkpoint import (
            safe_load_checkpoint as _safe_load_checkpoint,  # type: ignore[import-not-found]
        )
    except ImportError:
        # Fallback: absolute imports when run from repo root with PYTHONPATH set
        from utils.checkpoint import (
            load_state_dict_forgiving as _load_state_dict_forgiving,  # type: ignore[import-not-found]
        )
        from utils.checkpoint import (
            safe_load_checkpoint as _safe_load_checkpoint,  # type: ignore[import-not-found]
        )

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
        logger.error("Fine-tuning modules are unavailable.")
        sys.exit(3)

    requires_3d = _maybe_enable_add_3d(args)

    max_finetune_batches = int(getattr(args, "max_finetune_batches", 0) or 0)
    setattr(args, "max_finetune_batches", max_finetune_batches)
    pretrain_cap = int(getattr(args, "max_pretrain_batches", 0) or 0)
    if pretrain_cap > 0 and max_finetune_batches == 0:
        logger.warning(
            "Ignoring --max-pretrain-batches=%d during fine-tuning; use --max-finetune-batches to cap downstream epochs.",
            pretrain_cap,
        )
    if max_finetune_batches > 0:
        logger.info(
            "Fine-tune batches per epoch capped at %d via --max-finetune-batches.",
            max_finetune_batches,
        )

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
            "encoder_lr": getattr(args, "encoder_lr", None),
            "head_lr": getattr(args, "head_lr", None),
            "ema_decay": args.ema_decay,
            "seeds": seeds,
            "add_3d": bool(getattr(args, "add_3d", False)),
            "freeze_encoder": bool(getattr(args, "freeze_encoder", True)),
            "unfreeze_top_layers": int(getattr(args, "unfreeze_top_layers", 0) or 0),
            "max_finetune_batches": max_finetune_batches,
        },
    )
    log_effective_gnn(args, logger, wb)

    # Load labelled dataset
    try:
        labeled = load_directory_dataset(
            args.labeled_dir,
            label_col=args.label_col,
            add_3d=args.add_3d,
            num_workers=getattr(args, "num_workers", -1),
            cache_dir=getattr(args, "cache_dir", None),
        )  # type: ignore[arg-type]

        if requires_3d:
            _ensure_dataset_has_pos(labeled)

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
    seed_best_paths: Dict[int, str] = {}
    seed_train_steps: Dict[int, float] = {}
    seed_best_metric: Dict[int, float] = {}
    seed_best_step: Dict[int, float] = {}
    seed_best_mode: Dict[int, str] = {}
    encoder_unfreeze_mode: Optional[str] = None
    encoder_was_trainable = False
    cumulative_encoder_batches = 0.0

    # --- choose metric & direction (maximize cls metrics, minimize losses/errors) ---
    metric_choice = getattr(args, "metric", None)
    if not metric_choice:
        metric_choice = "val_auc" if args.task_type == "classification" else "val_loss"
        setattr(args, "metric", metric_choice)
        logger.debug("Defaulting fine-tune metric to %s for task=%s", metric_choice, args.task_type)

    metric_name = str(metric_choice or "").lower()
    maximize_metrics = {
        "acc",
        "accuracy",
        "auc",
        "auroc",
        "roc_auc",
        "val_auc",
        "f1",
        "f1_macro",
        "f1_micro",
        "r2",
        "val_r2",
    }
    preferred_mode = "max" if metric_name in maximize_metrics else "min"

    def _lookup_metric(m: dict, name: str):
        """Return float metric value; try common aliases if the exact key is missing."""
        v = m.get(name)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
        aliases = {
            "val_rmse": ["rmse_mean", "rmse"],
            "val_mae": ["mae_mean", "mae"],
            "val_auc": ["auc", "auroc", "roc_auc"],
            "acc": ["accuracy", "val_acc"],
            "accuracy": ["acc", "val_acc"],
            "r2": ["val_r2"],
        }
        for k in aliases.get(name, []):
            v = m.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    def _is_better(curr: float, best: Optional[float], mode: str) -> bool:
        if best is None:
            return True
        return curr > best if mode == "max" else curr < best

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

        # Unit tests frequently substitute light-weight encoder stubs without the
        # ``parameters`` iterator that ``nn.Module`` provides.  Guard against that
        # so downstream code can continue treating the encoder as frozen.
        params_attr = getattr(encoder, "parameters", None)
        if params_attr is None or not callable(params_attr):
            setattr(encoder, "parameters", lambda: iter(()))  # type: ignore[attr-defined]

        # Load pretrained encoder weights (from pretrain output)
        if getattr(args, "encoder", None):
            state, _ = _safe_load_checkpoint(
                primary=args.encoder,
                ckpt_dir=None,
                default_name="encoder.pt",
                map_location=device,
                allow_missing=False,
            )
            # state may be {"encoder": ...} or a raw state_dict
            enc_weights = state.get("encoder", state) if isinstance(state, dict) else {}
            if enc_weights:
                logger.info("[finetune] loaded encoder from %s", args.encoder)
                _load_state_dict_forgiving(encoder, enc_weights)
            else:
                logger.warning(
                    "[finetune] no encoder weights found in %s; using random init",
                    args.encoder,
                )

        extra_encoder_ckpt = getattr(args, "load_encoder_checkpoint", None)
        if extra_encoder_ckpt:
            state, _ = _safe_load_checkpoint(
                primary=extra_encoder_ckpt,
                ckpt_dir=None,
                default_name="encoder.pt",
                map_location=device,
                allow_missing=True,
            )
            enc_weights = state.get("encoder", state) if isinstance(state, dict) else {}
            if enc_weights:
                logger.info(
                    "[finetune] loaded additional encoder weights from %s",
                    extra_encoder_ckpt,
                )
                _load_state_dict_forgiving(encoder, enc_weights)
            else:
                logger.warning(
                    "[finetune] encoder checkpoint %s lacked weights; ignoring",
                    extra_encoder_ckpt,
                )

        # If resuming a fine-tune checkpoint, it may contain a fresher encoder
        if "encoder" in resume_state:
            logger.info("Overriding encoder from resume checkpoint")
            _load_state_dict_forgiving(encoder, resume_state["encoder"])

        # Build linear head for fine-tuning
        raw_mode = str(getattr(args, "unfreeze_mode", "none") or "none").lower()
        if raw_mode not in {"none", "partial", "full"}:
            raw_mode = "none"
        freeze_override = getattr(args, "freeze_encoder", None)
        if freeze_override is True:
            freeze_flag = True
            effective_mode = "none"
        elif freeze_override is False:
            freeze_flag = False
            effective_mode = "full"
        else:
            effective_mode = raw_mode
            freeze_flag = effective_mode != "full"

        unfreeze_top = int(getattr(args, "unfreeze_top_layers", 0) or 0)
        if effective_mode == "partial" and unfreeze_top <= 0:
            modules = list(encoder.children()) or [encoder]
            unfreeze_top = max(1, min(len(modules), 1))
            logger.info(
                "Partial unfreeze selected without explicit layer count; defaulting to top %d module(s).",
                unfreeze_top,
            )

        trainable_encoder_params = _configure_encoder_trainability(
            encoder,
            freeze_encoder=freeze_flag,
            unfreeze_top_layers=unfreeze_top,
            unfreeze_mode=effective_mode,
        )
        trainable_param_count = sum(int(p.numel()) for p in trainable_encoder_params)
        encoder_was_trainable = encoder_was_trainable or (trainable_param_count > 0)
        if trainable_param_count > 0:
            logger.info(
                "Encoder fine-tuning enabled (%d trainable parameters, mode=%s, freeze=%s, unfreeze_top_layers=%d)",
                trainable_param_count,
                effective_mode,
                freeze_flag,
                unfreeze_top,
            )
        else:
            logger.info(
                "Encoder frozen during fine-tuning (mode=%s, freeze=%s, unfreeze_top_layers=%d)",
                effective_mode,
                freeze_flag,
                unfreeze_top,
            )
        if effective_mode in {"partial", "full"} and trainable_param_count == 0:
            raise RuntimeError(
                "Encoder unfreeze mode '%s' requested but no trainable parameters were found. Check encoder construction."
                % effective_mode
            )
        if wb is not None:
            try:
                wb.log(
                    {
                        "encoder/trainable_params": float(trainable_param_count),
                        "encoder/freeze_flag": bool(freeze_flag),
                        "encoder/unfreeze_mode": effective_mode,
                    }
                )
            except Exception:
                pass
        encoder_unfreeze_mode = effective_mode
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

        if "head" in resume_state and hasattr(head, "load_state_dict"):
            _load_state_dict_forgiving(head, resume_state["head"])

        _maybe_to(head, device)

        # Optimizer & scheduler
        encoder_params = [
            p for p in _iter_trainable_params(encoder) if getattr(p, "requires_grad", False)
        ]
        head_params = [p for p in head.parameters() if p.requires_grad]
        optimizer_groups = []
        head_lr = getattr(args, "head_lr", None) or args.lr
        encoder_lr = getattr(args, "encoder_lr", None)
        if encoder_params:
            if encoder_lr is None:
                encoder_lr = head_lr * 0.1
                logger.info(
                    "Encoder LR defaulting to %.2e (head_lr * 0.1)", float(encoder_lr)
                )
            optimizer_groups.append({"params": encoder_params, "lr": encoder_lr})
        if head_params:
            optimizer_groups.append({"params": head_params, "lr": head_lr})

        optimizer = (
            torch.optim.AdamW(
                optimizer_groups,
                lr=head_lr,
                weight_decay=1e-4,
            )
            if optimizer_groups
            else None
        )
        if encoder_params:
            logger.info(
                "Optimizer encoder group: %d tensors lr=%.2e",
                len(encoder_params),
                float(encoder_lr or 0.0),
            )
        if head_params:
            logger.info(
                "Optimizer head group: %d tensors lr=%.2e",
                len(head_params),
                float(head_lr),
            )
        if wb is not None:
            try:
                lr_payload = {"optimizer/lr_head": float(head_lr)}
                if encoder_params:
                    lr_payload["optimizer/lr_encoder"] = float(encoder_lr or 0.0)
                wb.log(lr_payload)
            except Exception:
                pass

        encoder_lr_display: Optional[float] = None
        if encoder_params:
            try:
                encoder_lr_display = float(encoder_lr or 0.0)
            except Exception:
                encoder_lr_display = None
        logger.info(
            "[finetune] learning rates: lr_head=%.2e lr_encoder=%s unfreeze_mode=%s",
            float(head_lr),
            f"{encoder_lr_display:.2e}" if encoder_lr_display is not None else "<frozen>",
            effective_mode,
        )

        cache_embeddings = not bool(encoder_params)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        logger.info(
            "CosineAnnealingLR configured with T_max=%d epochs for fine-tuning.",
            args.epochs,
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
            best_metric: Optional[float] = None
            best_metric_mode = preferred_mode

            wrote_best = False
            last_epoch = start_epoch - 1
            warned_small_loader = False
            for epoch in range(start_epoch, args.epochs):
                metrics = train_linear_head(
                    dataset=labeled,
                    encoder=encoder,
                    # head_type=getattr(args, "head", "linear"),  # <- change innvalid param for supervised?
                    task_type=args.task_type,
                    epochs=1,
                    max_batches=max_finetune_batches,
                    time_budget_mins=getattr(
                        args, "time_budget_mins", 0
                    ),  # ensure it does not crash for unit tests
                    lr=args.lr,
                    batch_size=args.batch_size,
                    device=device,
                    patience=args.patience,
                    devices=args.devices,
                    head=head,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    # dataloader & AMP knobs
                    num_workers=getattr(args, "num_workers", -1),
                    pin_memory=getattr(args, "pin_memory", True),
                    persistent_workers=getattr(args, "persistent_workers", True),
                    prefetch_factor=getattr(args, "prefetch_factor", 4),
                    bf16=getattr(args, "bf16", False),
                    encoder_lr=getattr(args, "encoder_lr", None),
                    head_lr=getattr(args, "head_lr", None),
                    freeze_encoder=False,
                    early_stop_metric=getattr(args, "metric", "val_loss"),
                    cache_graph_embeddings=cache_embeddings,
                )

                train_batches = float(metrics.get("train/batches", 0.0) or 0.0)
                epoch_batches = float(metrics.get("train/epoch_batches", train_batches) or train_batches)
                loader_batches = float(metrics.get("train/loader_batches", 0.0) or 0.0)
                if epoch == start_epoch and loader_batches > 0:
                    logger.info(
                        "[finetune] seed=%d train loader reports %d batches per epoch (max_finetune_batches=%d)",
                        seed,
                        int(loader_batches),
                        max_finetune_batches,
                    )
                batch_size_hint = int(getattr(args, "batch_size", 0) or 0)
                if (
                    not warned_small_loader
                    and loader_batches > 0
                    and loader_batches < 6
                    and batch_size_hint >= 256
                ):
                    logger.warning(
                        "[finetune] seed=%d loader has %d batches with batch_size=%d; consider using 128 or 64 for ≥6 steps per epoch.",
                        seed,
                        int(loader_batches),
                        batch_size_hint,
                    )
                    warned_small_loader = True
                if train_batches > 0:
                    logger.info(
                        "[finetune] seed=%d epoch=%d encoder batches=%d",
                        seed,
                        epoch,
                        int(train_batches),
                    )
                if max_finetune_batches == 0 and epoch_batches < 10:
                    logger.warning(
                        "Fine-tune epoch produced only %d batches (seed=%d epoch=%d); representation updates may be limited.",
                        int(epoch_batches),
                        seed,
                        epoch,
                    )
                seed_train_steps[seed] = seed_train_steps.get(seed, 0.0) + epoch_batches
                cumulative_encoder_batches += epoch_batches
                if wb is not None:
                    try:
                        wb_payload = {
                            f"finetune_{seed}/train_batches": train_batches,
                            "encoder/train_batches": train_batches,
                            "encoder/train_batches_epoch": epoch_batches,
                            "encoder/train_batches_total": cumulative_encoder_batches,
                        }
                        if loader_batches > 0:
                            wb_payload[f"finetune_{seed}/loader_batches"] = loader_batches
                        wb.log(wb_payload)
                    except Exception:
                        pass

                trained_head = metrics.pop("head", None)
                if trained_head is not None:
                    head = trained_head

                current = _lookup_metric(metrics, metric_name)
                current_mode = preferred_mode

                if current is None and metric_name != "val_loss":
                    fallback_val = _lookup_metric(metrics, "val_loss")
                    if fallback_val is not None:
                        logger.debug(
                            "Metric '%s' missing in results; falling back to val_loss for checkpointing.",
                            metric_name,
                        )
                        current = fallback_val
                        current_mode = "min"

                if current is not None:
                    if best_metric_mode != current_mode:
                        best_metric = None
                        best_metric_mode = current_mode

                    if _is_better(current, best_metric, current_mode):
                        best_metric = current
                        best_path = os.path.join(seed_dir, "ft_best.pt")
                        best_payload = {"epoch": epoch, "best_metric": best_metric}
                        enc_state = _maybe_state_dict(encoder)
                        head_state = _maybe_state_dict(head)
                        if enc_state is not None:
                            best_payload["encoder"] = enc_state
                        if head_state is not None:
                            best_payload["head"] = head_state
                        if optimizer is not None and hasattr(optimizer, "state_dict"):
                            best_payload["optimizer"] = optimizer.state_dict()
                        if scheduler is not None and hasattr(scheduler, "state_dict"):
                            best_payload["scheduler"] = scheduler.state_dict()
                        save_checkpoint(best_path, **best_payload)
                        wrote_best = True
                        seed_best_paths[seed] = best_path
                        seed_best_metric[seed] = float(current)
                        seed_best_step[seed] = float(
                            seed_train_steps.get(seed, 0.0)
                        )
                        seed_best_mode[seed] = str(current_mode)
                        # optional: stable link at the finetune root

                        try:
                            from utils.checkpoint import safe_link_or_copy

                            link = os.path.join(args.ckpt_dir, "head.pt")
                            mode = safe_link_or_copy(best_path, link)
                            logger.info("Updated head.pt (%s) -> %s", mode, best_path)
                        except Exception:
                            logger.warning(
                                "Could not create head.pt symlink", exc_info=True
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

            # Fallback: if no best was recorded, promote last snapshot to best + head.pt
            if not wrote_best:
                logger.info("Attempting to write best")
                try:
                    # find latest epoch file we just wrote
                    snaps = [
                        p
                        for p in os.listdir(seed_dir)
                        if p.startswith("ft_epoch_") and p.endswith(".pt")
                    ]
                    if snaps:
                        snaps.sort(key=lambda s: int(s.split("_")[-1].split(".")[0]))
                        last = os.path.join(seed_dir, snaps[-1])
                        best_path = os.path.join(seed_dir, "ft_best.pt")

                        shutil.copy2(last, best_path)
                        from utils.checkpoint import safe_link_or_copy

                        mode = safe_link_or_copy(
                            best_path, os.path.join(args.ckpt_dir, "head.pt")
                        )
                        logger.info(
                            "Fallback best: head.pt (%s) -> %s", mode, best_path
                        )

                        seed_best_paths[seed] = best_path

                        logger.warning(
                            "No metric '%s' found; promoted %s to ft_best.pt",
                            metric_name,
                            snaps[-1],
                        )
                except Exception:
                    logger.warning("Failed to create fallback ft_best.pt", exc_info=True)  # type: ignore

            wb.log({"phase": f"finetune_{seed}", "status": "success"})
            metrics_runs.append({k: v for k, v in metrics.items() if k != "head"})
        except Exception:
            logger.exception(f"Fine‑tuning failed on seed {seed}")
            wb.log({"phase": f"finetune_{seed}", "status": "error"})
            sys.exit(3)

    total_train_batches = float(sum(seed_train_steps.values()))
    if encoder_was_trainable and total_train_batches <= 0:
        logger.warning(
            "Encoder marked trainable but no optimisation batches were recorded; check dataset splits."
        )
    elif encoder_was_trainable and total_train_batches < 50:
        logger.warning(
            "Encoder fine-tune ran only %d batches across all seeds; consider increasing dataset size, batch size, or epochs.",
            int(total_train_batches),
        )

    export_path: Optional[str] = None
    export_name: Optional[str] = None
    primary_seed = seeds[0] if seeds else None

    summary_seed: Optional[int] = None
    summary_metric_value: Optional[float] = None
    summary_mode = preferred_mode
    summary_score: Optional[float] = None
    for seed, value in seed_best_metric.items():
        if value is None:
            continue
        mode = seed_best_mode.get(seed, preferred_mode)
        try:
            val_float = float(value)
        except Exception:
            continue
        score = val_float if mode == "max" else -val_float
        if summary_score is None or score > summary_score:
            summary_score = score
            summary_seed = seed
            summary_metric_value = val_float
            summary_mode = mode
    if summary_seed is None:
        summary_seed = primary_seed
        summary_metric_value = (
            seed_best_metric.get(summary_seed)
            if summary_seed is not None
            else None
        )
        summary_mode = seed_best_mode.get(summary_seed, preferred_mode)
    summary_step = (
        seed_best_step.get(summary_seed)
        if summary_seed is not None
        else None
    )
    if summary_step is None and summary_seed is not None:
        summary_step = seed_train_steps.get(summary_seed)

    def _fmt_metric(val: Optional[float]) -> str:
        if val is None:
            return "<nan>"
        try:
            return f"{float(val):.4f}"
        except Exception:
            return "<nan>"

    logger.info(
        "[finetune] summary: total_encoder_steps=%.1f best_seed=%s best_step=%.1f metric=%s mode=%s value=%s",
        total_train_batches,
        summary_seed if summary_seed is not None else "<none>",
        float(summary_step or 0.0),
        metric_name,
        summary_mode,
        _fmt_metric(summary_metric_value),
    )
    export_hash: Optional[str] = None
    if encoder_was_trainable and seed_best_paths:
        best_candidate = seed_best_paths.get(primary_seed) if primary_seed is not None else None
        if not best_candidate:
            best_candidate = next(iter(seed_best_paths.values()), None)
        if best_candidate:
            try:
                best_state = load_checkpoint(best_candidate)
                enc_state = (
                    best_state.get("encoder")
                    if isinstance(best_state, dict)
                    else best_state
                )
                if enc_state:
                    alias_source = (
                        os.getenv("EXP_ID")
                        or os.getenv("RUN_ID")
                        or (f"seed{primary_seed}" if primary_seed is not None else "finetune")
                    )
                    # The exported checkpoint is tagged as ``encoder_ft:<alias>`` so
                    # downstream evaluations can distinguish fine-tuned encoders
                    # from the original pretraining lineage.
                    alias = _sanitize_alias(str(alias_source))
                    export_name = f"encoder_ft:{alias}"
                    export_path = os.path.join(args.ckpt_dir, f"encoder_ft_{alias}.pt")
                    encoder_hash = None
                    try:
                        encoder_hash = compute_state_dict_hash(enc_state)
                    except Exception:
                        logger.exception("Failed to compute hash for fine-tuned encoder export")
                    save_checkpoint(export_path, encoder=enc_state)
                    try:
                        from utils.checkpoint import safe_link_or_copy

                        link_target = os.path.join(args.ckpt_dir, "encoder_ft.pt")
                        safe_link_or_copy(export_path, link_target)
                    except Exception:
                        logger.debug("Could not create encoder_ft.pt symlink", exc_info=True)
                    logger.info(
                        "Exported fine-tuned encoder to %s (artifact %s)",
                        export_path,
                        export_name,
                    )
                    if wb is not None:
                        try:
                            wb.log(
                                {
                                    "encoder/export_path": export_path,
                                    "encoder/export_artifact": export_name,
                                }
                            )
                        except Exception:
                            pass
                    if encoder_hash:
                        export_hash = encoder_hash
                        logger.info("[encoder_hash]=%s source=finetune_export path=%s", encoder_hash, export_path)
            except Exception:
                logger.exception(
                    "Failed to export fine-tuned encoder from %s", best_candidate
                )

    agg = aggregate_metrics(metrics_runs)
    for k, v in agg.items():
        wb.log({f"metric/{k}": v})
    wb.finish()

    stage_payload: Dict[str, Any] = {
        "encoder_unfreeze_mode": encoder_unfreeze_mode or "none",
        "encoder_trainable": bool(encoder_was_trainable),
        "encoder_train_steps": total_train_batches,
        "seeds": {
            str(seed): {
                "best_checkpoint": seed_best_paths.get(seed),
                "train_batches": float(seed_train_steps.get(seed, 0.0)),
                "best_metric": float(seed_best_metric.get(seed))
                if seed in seed_best_metric
                else None,
                "best_step": float(seed_best_step.get(seed))
                if seed in seed_best_step
                else None,
                "best_mode": seed_best_mode.get(seed),
            }
            for seed in seeds
        },
    }
    if export_path:
        stage_payload["encoder_finetuned"] = {
            "checkpoint": export_path,
            "artifact": export_name,
            "source_seed": primary_seed,
        }
        if export_hash:
            stage_payload["encoder_finetuned"]["hash"] = export_hash
    _record_finetune_stage_outputs(stage_payload)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a pretrained encoder by training a linear probe across seeds."""
    logger.info("Starting evaluate with args: %s", args)
    # Reuse finetune implementation with a different default config section
    cmd_finetune(args)


def evaluate_finetuned_head(
    ckpt_path: str, dataset, args: argparse.Namespace, device
) -> Dict[str, float]:
    """Evaluate a fine‑tuned encoder+head on a labelled dataset.

    This is used for the benchmark step when we want to avoid training a new
    head on the test split. The checkpoint is expected to contain both an
    ``encoder`` and ``head`` state dict. Metrics are computed on the entire
    dataset.
    """

    from utils.checkpoint import load_checkpoint
    from utils.metrics import compute_classification_metrics, compute_regression_metrics

    try:
        from ..models.factory import build_encoder  # type: ignore[import-not-found]
        from ..utils.checkpoint import (
            load_state_dict_forgiving as _load_state_dict_forgiving,  # type: ignore[import-not-found]
        )
        from ..utils.pooling import global_mean_pool  # type: ignore[import-not-found]
    except ImportError:
        # Fallback: absolute imports when run from repo root with PYTHONPATH set
        from models.factory import build_encoder  # type: ignore[import-not-found]
        from utils.checkpoint import (
            load_state_dict_forgiving as _load_state_dict_forgiving,  # type: ignore[import-not-found]
        )
        from utils.pooling import global_mean_pool  # type: ignore[import-not-found]

    # module logger (safe even when run outside the injector)
    import logging

    logger = logging.getLogger(__name__)

    # local helper so there are zero naming conflicts with injected globals
    def _to_dev(x, dev):
        try:
            return x.to(dev)
        except Exception:
            return x

    state = load_checkpoint(ckpt_path)
    if "encoder" not in state or "head" not in state:
        logger.warning("Checkpoint missing encoder or head: %s", ckpt_path)
        return {}

    # 1) Try to get the exact encoder config from the finetune ckpt
    enc_cfg = {}
    if isinstance(state, dict):
        enc_cfg = {
            k: v for k, v in (state.get("encoder_cfg") or {}).items() if v is not None
        }

    logger.info("Eval encoder cfg: %s", enc_cfg)

    # 2) If missing, look for a sidecar pretrain encoder next to the head
    #    (we often symlink encoder.pt into the finetune dir)
    import collections
    import os

    import torch

    sidecar = os.path.join(os.path.dirname(ckpt_path or ""), "encoder.pt")
    side_state = None
    if not enc_cfg and os.path.isfile(sidecar):
        try:
            side_state = torch.load(sidecar, map_location="cpu")
        except Exception:
            side_state = None

    # 3) If still missing, *attempt* to infer hidden_dim from state shapes (best-effort)
    def _infer_hidden_dim(sd):
        if not isinstance(sd, dict):
            return None
        c = collections.Counter()
        for k, v in sd.items():
            shp = getattr(v, "shape", None)
            if isinstance(shp, tuple) and len(shp) == 2:
                in_f = shp[1]
                if 64 <= in_f <= 2048 and in_f % 32 == 0:
                    c[in_f] += 1
        return c.most_common(1)[0][0] if c else None

    if not enc_cfg:
        hid = _infer_hidden_dim((state or {}).get("encoder", {})) or _infer_hidden_dim(
            side_state or {}
        )
        if hid:
            enc_cfg["hidden_dim"] = hid

    # 4) Fall back to CLI args for anything still missing
    for k in ("gnn_type", "hidden_dim", "num_layers", "add_3d"):
        if k not in enc_cfg and hasattr(args, k):
            enc_cfg[k] = getattr(args, k)

    # 4.1) Derive edge_dim from the dataset (needed for edge_mpnn)
    try:
        g0 = dataset.graphs[0]
        _edge_dim = (
            None
            if getattr(g0, "edge_attr", None) is None
            else int(g0.edge_attr.shape[1])
        )
    except Exception:
        _edge_dim = None
    if _edge_dim is None or _edge_dim <= 0:
        _edge_dim = 1
    enc_cfg.setdefault("edge_dim", _edge_dim)

    # 4.5) Ensure required input_dim is present (infer from dataset if needed)
    in_dim = None
    for attr in ("input_dim", "node_feat_dim", "n_node_features"):
        val = getattr(dataset, attr, None)
        if isinstance(val, int) and val > 0:
            in_dim = val
            break
    if in_dim is None:
        try:
            bx, _, _, _ = dataset.get_batch([0])
            in_dim = int(bx.shape[-1])
        except Exception:
            in_dim = None
    if in_dim is not None:
        enc_cfg["input_dim"] = in_dim

    # 5) Finally build the encoder with the best config we have (filter unknown keys)

    import inspect

    enc_cfg = enc_cfg or {}
    try:
        sig_params = set(inspect.signature(build_encoder).parameters.keys())
    except (ValueError, TypeError):
        sig_params = {
            "gnn_type",
            "hidden_dim",
            "num_layers",
            "input_dim",
        }  # conservative fallback
    # normalize & filter
    norm = dict(enc_cfg)
    gt = norm.get("gnn_type")
    if isinstance(gt, str):
        norm["gnn_type"] = gt.lower()

    if norm.get("gnn_type") == "edge_mpnn" and norm.get("edge_dim") is None:
        # Be permissive: default to a single constant edge feature instead of crashing
        norm["edge_dim"] = 1
        if "logger" in globals():
            logger.warning(
                "edge_dim missing for edge_mpnn; defaulting to 1 (no edge features found)"
            )

    filtered = {k: v for k, v in norm.items() if (v is not None and k in sig_params)}
    extra = [k for k in norm if k not in sig_params]
    if extra and "logger" in globals():
        logger.debug("build_encoder: ignoring unsupported keys: %s", extra)
    enc = build_encoder(**filtered)

    # If edge features are missing/empty, pad them so forward() won’t blow up
    def _ensure_edge_attr(g, need_dim: int, device=None):
        """
        Ensure g.edge_attr exists and has shape (E, need_dim).
        Works whether g.x / g.edge_attr are numpy arrays or torch tensors.
        """
        import numpy as np

        try:
            import torch as _t

            _HAS_TORCH = True
        except Exception:
            _HAS_TORCH = False

        # 1) How many edges?
        E = 0
        ei = getattr(g, "edge_index", None)
        if ei is not None:
            try:
                E = int(ei.shape[1])
            except Exception:
                E = int(np.array(ei).shape[1])
        else:
            adj = getattr(g, "adj", None)
            if adj is not None:
                if _HAS_TORCH and isinstance(adj, _t.Tensor):
                    E = int((adj > 0).sum().item())
                else:
                    A = np.asarray(adj)
                    E = int((A > 0).sum())

        # 2) Build a zeros matrix of the *same type family* as g.x (numpy or torch)
        def _zeros_like_x(n_rows, n_cols):
            x = getattr(g, "x", None)
            if _HAS_TORCH and isinstance(x, _t.Tensor):
                dt = x.dtype
                dev = x.device if hasattr(x, "device") else device
                if dev is not None:
                    return _t.zeros((n_rows, n_cols), dtype=dt, device=dev)
                return _t.zeros((n_rows, n_cols), dtype=dt)
            # fallback: numpy
            dt = getattr(x, "dtype", np.float32)
            return np.zeros((n_rows, n_cols), dtype=dt)

        # 3) Create or fix edge_attr
        e = getattr(g, "edge_attr", None)
        e_w = getattr(e, "shape", (0, 0))[1] if e is not None else 0

        if e is None or e_w == 0:
            g.edge_attr = _zeros_like_x(E, need_dim)
            return g

        # 4) Pad / truncate to need_dim (handle both numpy and torch)
        if _HAS_TORCH and isinstance(e, _t.Tensor):
            if e.shape[1] < need_dim:
                pad = _zeros_like_x(e.shape[0], need_dim - e.shape[1])
                g.edge_attr = _t.cat([e, pad], dim=1)
            elif e.shape[1] > need_dim:
                g.edge_attr = e[:, :need_dim]
        else:
            e_np = np.asarray(e)
            if e_np.shape[1] < need_dim:
                pad = _zeros_like_x(e_np.shape[0], need_dim - e_np.shape[1])
                g.edge_attr = np.concatenate([e_np, pad], axis=1)
            elif e_np.shape[1] > need_dim:
                g.edge_attr = e_np[:, :need_dim]

        return g

    # load weights (prefer finetune's encoder substate; else sidecar)
    enc_sub = (state or {}).get("encoder", {})
    if not enc_sub and isinstance(side_state, dict):
        enc_sub = side_state
    _load_state_dict_forgiving(enc, enc_sub)

    # ---- build & load HEAD from checkpoint (infer shape from saved weights) ----
    import torch.nn as nn

    head_state = (state or {}).get("head", {})
    in_dim, out_dim = None, 1

    if isinstance(head_state, dict):
        for k, v in head_state.items():
            if k.endswith("weight") and getattr(v, "ndim", 0) == 2:
                out_dim, in_dim = v.shape  # [out, in]
                break

    # best-effort fallback for in_dim if weight shape wasn’t found
    if in_dim is None:
        in_dim = (
            enc_cfg.get("hidden_dim")
            or getattr(enc, "hidden_dim", None)
            or getattr(enc, "out_dim", None)
        )
    if in_dim is None:
        logger.error("Cannot infer head input dim; encoder_cfg=%s", enc_cfg)
        raise RuntimeError("Cannot infer head input dim")

    head = nn.Linear(int(in_dim), int(out_dim))
    if isinstance(head_state, dict) and head_state:
        _load_state_dict_forgiving(head, head_state)

    # move to device & eval
    enc = _to_dev(enc, device)
    head = _to_dev(head, device)
    enc.eval()
    head.eval()

    # use probabilities for classification; raw scores for regression
    task_is_cls = getattr(args, "task_type", "regression") == "classification"
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for start in range(0, len(dataset), args.batch_size):
        batch_indices = list(range(start, min(start + args.batch_size, len(dataset))))
        batch_x, batch_adj, batch_ptr, batch_labels = dataset.get_batch(batch_indices)

        batch_pos_np = None
        if hasattr(dataset, "graphs"):
            pos_blocks = []
            all_have_pos = True
            for idx in batch_indices:
                g_single = dataset.graphs[idx]
                pos_arr = getattr(g_single, "pos", None)
                if pos_arr is None:
                    all_have_pos = False
                    break
                pos_blocks.append(np.asarray(pos_arr, dtype=np.float32))
            if all_have_pos and pos_blocks:
                batch_pos_np = np.concatenate(pos_blocks, axis=0)

        batch_x = batch_x.to(device, non_blocking=True)
        batch_adj = batch_adj.to(device, non_blocking=True)
        # batch_ptr may be None; when present, pooling indices should be long
        batch_ptr = (
            batch_ptr.to(device, non_blocking=True).long()
            if batch_ptr is not None
            else None
        )

        with torch.no_grad():
            edge_idx = batch_adj.nonzero().T.detach().cpu().numpy()
            if edge_idx.size == 0:
                edge_idx = np.zeros((2, 0), dtype=np.int64)
            g = GraphData(
                x=batch_x.detach().cpu().numpy(),
                edge_index=edge_idx,
                pos=batch_pos_np,
            )

            if batch_ptr is not None:
                g.graph_ptr = batch_ptr.detach().cpu()

            g = _ensure_edge_attr(g, int(enc_cfg["edge_dim"]), device=device)
            node_emb = _encode_graph(enc, g)  # [N, D]
            # Guard against NaNs/Infs before pooling or batching
            node_emb = torch.nan_to_num(node_emb, nan=0.0, posinf=0.0, neginf=0.0)
            graph_emb = (
                global_mean_pool(node_emb, batch_ptr)
                if batch_ptr is not None
                else node_emb.mean(dim=0, keepdim=True)
            )
            logits = head(graph_emb).squeeze(1)
            # Clamp any stray non-finite values in the head output
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            preds_t = torch.sigmoid(logits) if task_is_cls else logits
            # And sanitize the post-activation tensor too (esp. regression path)
            preds_t = torch.nan_to_num(preds_t, nan=0.0, posinf=1e6, neginf=-1e6)
            all_preds.append(preds_t.detach().cpu().numpy())
            # targets: one value per graph in the batch (already aligned with graph_emb)
            all_targets.append(batch_labels.detach().cpu().numpy())

    # concat & filter non-finite rows (both y_true and y_pred)
    y_pred = np.concatenate(all_preds).astype(np.float32)
    y_true = np.concatenate(all_targets).astype(np.float32)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if task_is_cls:
        y_true = y_true.astype(np.int64, copy=False)
        # y_pred are probabilities in [0,1] for classification
        return compute_classification_metrics(y_true, y_pred)
    return compute_regression_metrics(y_true, y_pred)
