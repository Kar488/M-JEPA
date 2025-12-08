from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from utils.wandb_filters import silence_pydantic_field_warnings
except Exception:  # pragma: no cover - helper optional in minimal installs
    def silence_pydantic_field_warnings() -> None:  # type: ignore
        return

from . import log_effective_gnn

stage_config: Dict[str, Any] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_metric_direction(name: Optional[str], fallback: Optional[bool] = None) -> Optional[bool]:
    if isinstance(fallback, bool):
        return fallback
    if not name:
        return fallback
    lowered = name.lower()
    if any(token in lowered for token in ("loss", "rmse", "mae", "error", "mse", "duration")):
        return False
    if any(token in lowered for token in ("acc", "auc", "f1", "precision", "recall", "roc")):
        return True
    return fallback


def _normalize_stage_outputs_dir() -> Optional[Path]:
    stage_dir = os.getenv("STAGE_OUTPUTS_DIR")
    if not stage_dir:
        return None
    try:
        path = Path(stage_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def _record_stage_outputs(payload: Dict[str, Any]) -> None:
    stage_dir = _normalize_stage_outputs_dir()
    if stage_dir is None:
        return
    try:
        out = stage_dir / "pretrain.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except Exception:
        # Stage outputs are best-effort; never crash the training command.
        logger.warning("Failed to write pretrain stage outputs", exc_info=True)


def _extract_metric_from_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    # Support a variety of attribute spellings to ease integration with CI wrappers.
    metric: Dict[str, Any] = {}

    if hasattr(args, "best_metric") and isinstance(getattr(args, "best_metric"), dict):
        metric.update(getattr(args, "best_metric"))

    name_candidates = [
        getattr(args, "validation_metric_name", None),
        getattr(args, "val_metric_name", None),
        getattr(args, "best_metric_name", None),
        getattr(args, "metric_name", None),
        getattr(args, "metric", None),
    ]
    value_candidates = [
        getattr(args, "validation_metric", None),
        getattr(args, "val_metric", None),
        getattr(args, "best_metric_value", None),
        getattr(args, "metric_value", None),
    ]
    hib_candidates = [
        getattr(args, "validation_metric_higher_is_better", None),
        getattr(args, "val_metric_higher_is_better", None),
        getattr(args, "higher_is_better", None),
        getattr(args, "metric_higher_is_better", None),
    ]

    env_value = os.getenv("PRETRAIN_VAL_METRIC_VALUE") or os.getenv("VAL_METRIC_VALUE")
    env_name = os.getenv("PRETRAIN_VAL_METRIC_NAME") or os.getenv("VAL_METRIC_NAME")
    env_dir = os.getenv("PRETRAIN_VAL_METRIC_DIRECTION") or os.getenv("VAL_METRIC_DIRECTION")

    def _parse_direction(raw: Any) -> Optional[bool]:
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return bool(raw)
        text = str(raw).strip().lower()
        if text in {"true", "t", "1", "yes", "y", "on", "max", "maximize", "higher", "up"}:
            return True
        if text in {"false", "f", "0", "no", "n", "off", "min", "minimize", "lower", "down"}:
            return False
        return None

    name = metric.get("name") or next((n for n in name_candidates if n), None) or env_name
    value = metric.get("value")
    if value is None:
        for cand in value_candidates:
            if cand is not None:
                value = cand
                break
    if value is None and env_value is not None:
        value = env_value

    higher_is_better = metric.get("higher_is_better")
    if higher_is_better is None:
        for cand in hib_candidates:
            parsed = _parse_direction(cand)
            if parsed is not None:
                higher_is_better = parsed
                break
    if higher_is_better is None:
        higher_is_better = _parse_direction(env_dir)

    val = _safe_float(value)
    if val is None:
        return None

    higher_is_better = _infer_metric_direction(name, higher_is_better)
    return {
        "name": name,
        "value": val,
        "higher_is_better": higher_is_better if isinstance(higher_is_better, bool) else None,
    }


def _extract_metric_from_manifest(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return None
    for key in ("validation", "val", "best", "metric"):
        candidate = metrics.get(key)
        if isinstance(candidate, dict):
            if "value" in candidate:
                result = {
                    "name": candidate.get("name"),
                    "value": _safe_float(candidate.get("value")),
                    "higher_is_better": candidate.get("higher_is_better"),
                }
                if result["value"] is not None:
                    if result["higher_is_better"] is None:
                        result["higher_is_better"] = _infer_metric_direction(result.get("name"))
                    return result
    return None


def _load_existing_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logger.warning("Failed to parse existing encoder manifest at %s", path, exc_info=True)
        return None


def _metric_is_better(new: Dict[str, Any], old: Optional[Dict[str, Any]]) -> bool:
    if new is None:
        return old is None
    if old is None:
        return True
    new_val = _safe_float(new.get("value"))
    old_val = _safe_float(old.get("value"))
    if new_val is None:
        return False
    if old_val is None:
        return True
    higher = new.get("higher_is_better")
    if higher is None:
        old_direction = None
        if isinstance(old, dict):
            old_direction = _infer_metric_direction(
                old.get("name"), old.get("higher_is_better")
            )
        higher = _infer_metric_direction(new.get("name"), old_direction)
    if higher is None:
        higher = True
    if higher:
        return new_val > old_val
    return new_val < old_val


def _collect_run_metadata(wb) -> Dict[str, Any]:
    run = getattr(wb, "run", None)
    if run is None:
        run = wb
    if run is None:
        return {}
    meta = {}
    for attr in ("id", "name", "project", "entity", "group", "job_type"):
        value = getattr(run, attr, None)
        if value:
            meta[attr] = value
    url = getattr(run, "url", None)
    if url:
        meta["url"] = url
    return meta


def cmd_pretrain(args: argparse.Namespace) -> None:
    """Self‑supervised pretraining of a JEPA encoder and optional contrastive baseline."""
    logger.info("Starting pretrain with args: %s", args)
    if load_directory_dataset is None or build_encoder is None or train_jepa is None:
        logger.error("Pretraining modules are unavailable.")
        sys.exit(2)

    # W&B run
    wb = maybe_init_wandb(
        args.use_wandb,
        project=getattr(args, "wandb_project", os.getenv("WANDB_PROJECT", "m-jepa")),
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
            "temperature": args.temperature,
            "ema_decay": args.ema_decay,
            "contrastive": args.contrastive,
        },
    )
    log_effective_gnn(args, logger, wb)

    lr_value = getattr(args, "lr", None)
    mask_ratio = getattr(args, "mask_ratio", None)
    sample_unlabeled = getattr(args, "sample_unlabeled", None)
    logger.info(
        "[pretrain] effective_hparams gnn_type=%s hidden_dim=%s num_layers=%s lr=%s mask_ratio=%s sample_unlabeled=%s",
        getattr(args, "gnn_type", None),
        getattr(args, "hidden_dim", None),
        getattr(args, "num_layers", None),
        lr_value,
        mask_ratio,
        sample_unlabeled if sample_unlabeled not in (None, 0) else "all",
    )

    try:
        lr_float = float(lr_value)
    except (TypeError, ValueError):
        lr_float = None

    if lr_float is None or lr_float <= 0:
        logger.error("[pretrain] invalid non-positive learning rate: %s", lr_value)
        sys.exit(2)

    gnn_type = str(getattr(args, "gnn_type", ""))
    allow_gcn_fallback = _env_flag("ALLOW_GCN_FALLBACK", default=_env_flag("MJEPA_ALLOW_DATA_FALLBACKS", True))
    if gnn_type.lower().strip() == "gcn":
        if allow_gcn_fallback:
            logger.warning(
                "[pretrain] allowing gnn_type=gcn because ALLOW_GCN_FALLBACK=%s",
                os.getenv("ALLOW_GCN_FALLBACK", os.getenv("MJEPA_ALLOW_DATA_FALLBACKS", "1")),
            )
        else:
            logger.error(
                "[pretrain] refusing to launch with fallback gnn_type=gcn; check Phase-2 winner propagation"
            )
            sys.exit(2)

    def _wb_run_ok(wb):
        return (wb is not None) and (getattr(wb, "run", None) is not None)

    def _wb_log(wb, payload: dict):
        if _wb_run_ok(wb):
            run = wb.run
            # prefer run.log when present to avoid the preinit wrapper
            (getattr(run, "log", wb.log))(payload)

    def _wb_summary(wb, payload: dict):
        if _wb_run_ok(wb):
            wb.summary.update(payload)

    def _wb_finish(wb):
        if _wb_run_ok(wb):
            try:
                wb.finish()
            except Exception:
                pass

    import random

    from utils.checkpoint import (
        compute_state_dict_hash,
        load_checkpoint,
        save_checkpoint,
    )

    # Resume state
    args.ckpt_dir = getattr(args, "ckpt_dir", "ckpts/pretrain")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_every = max(1, int(getattr(args, "save_every", 1)))
    start_epoch = 0

    if getattr(args, "resume_ckpt", None):
        _wb_log(wb, {"phase": "pretrain", "status": "resume", "ckpt": args.resume_ckpt})
        ckpt_state = load_checkpoint(args.resume_ckpt)
    else:
        ckpt_state = {}

    try:
        # Load unlabeled dataset
        try:
            seeds: tuple
            # Determine seeds: use CLI if provided, otherwise fall back to configuration defaults
            if args.seeds is not None and len(args.seeds) > 0:
                seeds = tuple(args.seeds)
            else:
                seeds = tuple(CONFIG.get("pretrain", {}).get("seeds", [0]))

            seed = int(seeds[0]) if seeds else 0

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Sample a subset of the unlabeled dataset if requested.  Use getattr to
            # avoid AttributeError when the caller hasn’t set sample_unlabeled.
            sample_ul = getattr(args, "sample_unlabeled", 0) or None
            rows_per_file = getattr(args, "n_rows_per_file", None)
            logger.info(
                "Loading unlabeled (cap=%s, rows_per_file=%s, workers=%s)…",
                sample_ul,
                rows_per_file,
                getattr(args, "num_workers", -1),
            )
            t0 = time.time()

            unlabeled = load_directory_dataset(
                args.unlabeled_dir,
                add_3d=args.add_3d,
                num_workers=getattr(args, "num_workers", -1),
                cache_dir=getattr(args, "cache_dir", None),
                n_rows_per_file=rows_per_file,
                max_graphs=sample_ul,
            )  # type: ignore[arg-type]
            logger.info(
                "Loaded unlabeled dataset in %.2fs (%s graphs)",
                time.time() - t0,
                len(unlabeled),
            )

            _wb_log(wb, {"phase": "data_load", "unlabeled_graphs": len(unlabeled)})
        except Exception:
            logger.exception("Failed to load unlabeled dataset")
            _wb_log(wb, {"phase": "data_load", "status": "error"})
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
                _load_state_dict_forgiving(encoder, ckpt_state["encoder"])
            if "ema_encoder" in ckpt_state:
                _load_state_dict_forgiving(ema_encoder, ckpt_state["ema_encoder"])
            if "predictor" in ckpt_state:
                _load_state_dict_forgiving(predictor, ckpt_state["predictor"])
            if "ema" in ckpt_state and hasattr(ema_helper, "load_state_dict"):
                _load_state_dict_forgiving(ema_helper, ckpt_state["ema"])
            start_epoch = ckpt_state.get("epoch", 0) + 1

        # Augmentation kwargs for JEPA pretraining
        kwargs: Dict[str, bool] = {}
        if getattr(args, "aug_rotate", False):
            kwargs["random_rotate"] = True
        if getattr(args, "aug_mask_angle", False):
            kwargs["mask_angle"] = True
        if getattr(args, "aug_dihedral", False):
            kwargs["perturb_dihedral"] = True
        if getattr(args, "aug_bond_deletion", False):
            kwargs["bond_deletion"] = True
        if getattr(args, "aug_atom_masking", False):
            kwargs["atom_masking"] = True
        if getattr(args, "aug_subgraph_removal", False):
            kwargs["subgraph_removal"] = True

        # Pretrain JEPA
        pretrain_losses: List[float] = []
        try:
            _wb_log(wb, {"phase": "pretrain", "status": "start"})
            for epoch in range(start_epoch, args.epochs):
                ep_loss = train_jepa(
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
                    contiguous=getattr(args, "contiguity", False),
                    lr=args.lr,
                    device=device,
                    devices=getattr(args, "devices", 1),
                    reg_lambda=1e-4,
                    use_wandb=args.use_wandb,
                    wandb_project=args.wandb_project,
                    wandb_tags=args.wandb_tags,
                    disable_tqdm=(not getattr(args, "force_tqdm", False))
                    and (not sys.stdout.isatty()),
                    # dataloader & AMP knobs
                    num_workers=getattr(args, "num_workers", -1),
                    pin_memory=getattr(args, "pin_memory", True),
                    persistent_workers=getattr(args, "persistent_workers", True),
                    prefetch_factor=getattr(args, "prefetch_factor", 4),
                    bf16=getattr(args, "bf16", False),
                    compile_models=not getattr(args, "no_compile", False),
                    # forward augmentation flags only when enabled
                    **kwargs,
                )
                pretrain_losses.extend(ep_loss)
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
            _wb_log(wb, {"phase": "pretrain", "status": "success"})
        except Exception:
            logger.exception("JEPA pretraining failed")
            _wb_log(wb, {"phase": "pretrain", "status": "error"})
            sys.exit(2)

        aug_cfg = AugmentationConfig(
            rotate=getattr(args, "aug_rotate", False),
            mask_angle=getattr(args, "aug_mask_angle", False),
            dihedral=getattr(args, "aug_dihedral", False),
            bond_deletion=getattr(args, "aug_bond_deletion", False),
            atom_masking=getattr(args, "aug_atom_masking", False),
            subgraph_removal=getattr(args, "aug_subgraph_removal", False),
        )

        # Optionally run contrastive baseline
        cont_losses: List[float] = []
        cont_path: Optional[str] = None
        if args.contrastive:
            cont_encoder = build_encoder(
                gnn_type=args.gnn_type,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                edge_dim=edge_dim,
            )
            try:
                _wb_log(wb, {"phase": "pretrain_contrastive", "status": "start"})
                cont_losses = train_contrastive(  # type: ignore[call-arg]
                    dataset=unlabeled,
                    encoder=cont_encoder,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    mask_ratio=args.mask_ratio,
                    lr=args.lr,
                    temperature=args.temperature,
                    device=device,
                    devices=getattr(args, "devices", 1),
                    use_wandb=args.use_wandb,
                    random_rotate=aug_cfg.rotate,
                    mask_angle=aug_cfg.mask_angle,
                    perturb_dihedral=aug_cfg.dihedral,
                    bond_deletion=aug_cfg.bond_deletion,
                    atom_masking=aug_cfg.atom_masking,
                    subgraph_removal=aug_cfg.subgraph_removal,
                    wandb_project=args.wandb_project,
                    wandb_tags=args.wandb_tags,
                    disable_tqdm=(not getattr(args, "force_tqdm", False))
                    and (not sys.stdout.isatty()),
                    # dataloader & AMP knobs
                    num_workers=getattr(args, "num_workers", -1),
                    pin_memory=getattr(args, "pin_memory", True),
                    persistent_workers=getattr(args, "persistent_workers", True),
                    prefetch_factor=getattr(args, "prefetch_factor", 4),
                    bf16=getattr(args, "bf16", False),
                )

                _wb_log(wb, {"phase": "pretrain_contrastive", "status": "success"})
            except Exception:
                logger.exception("Contrastive pretraining failed")
                _wb_log(wb, {"phase": "pretrain_contrastive", "status": "error"})
                sys.exit(2)

        # Save checkpoints
        ckpt_base = args.output
        os.makedirs(os.path.dirname(ckpt_base) or ".", exist_ok=True)
        enc_state = encoder.state_dict()
        encoder_hash = None
        try:
            encoder_hash = compute_state_dict_hash(enc_state)
        except Exception:
            logger.exception("Failed to compute encoder hash during export")
        encoder_cfg = {
            "gnn_type": args.gnn_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "add_3d": bool(getattr(args, "add_3d", False)),
            "edge_dim": int(edge_dim) if edge_dim is not None else None,
            "input_dim": int(input_dim),
        }
        save_checkpoint(ckpt_base, encoder=enc_state, encoder_cfg=encoder_cfg)
        _wb_log(wb, {"jepa_checkpoint": ckpt_base})
        if args.contrastive:
            cont_path = f"{os.path.splitext(ckpt_base)[0]}_contrastive.pt"
            save_checkpoint(
                cont_path,
                encoder=cont_encoder.state_dict(),
                encoder_cfg=encoder_cfg,
            )
            _wb_log(wb, {"contrastive_checkpoint": cont_path})

        # keep a stable pointer the FT step can always find
        try:
            link = os.path.join(args.ckpt_dir, "encoder.pt")
            if os.path.realpath(link) != os.path.realpath(ckpt_base):
                if os.path.islink(link) or os.path.exists(link):
                    os.remove(link)
                os.symlink(ckpt_base, link)
        except Exception:
            logger.warning("Could not create encoder.pt symlink", exc_info=True)

        # Plot training losses to W&B and filesystem
        plot_dir = getattr(args, "plot_dir", None) or os.path.join(
            args.ckpt_dir, "plots"
        )
        csv_path = os.path.join(args.ckpt_dir, "pretrain_losses.csv")
        try:
            curves = {"jepa": pretrain_losses}
            if cont_losses:
                curves["contrastive"] = cont_losses
            fig = plot_training_curves(curves, wb=wb)

            # Respect --plot-dir if provided; otherwise default to <ckpt_dir>/plots
            os.makedirs(plot_dir, exist_ok=True)
            out_png = os.path.join(plot_dir, "pretrain_loss.png")
            fig.savefig(out_png, dpi=200)

            # Log to W&B only if a run exists (avoid preinit errors)
            try:
                silence_pydantic_field_warnings()
                import wandb as _wandb

                if (wb is not None) and (getattr(wb, "run", None) is not None):
                    wb.log({"pretrain/loss_plot": _wandb.Image(out_png)})
            except Exception:
                pass

            # Also write a CSV of epoch losses next to the checkpoint
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,loss\n")
                for i, v in enumerate(pretrain_losses, 1):
                    f.write(f"{i},{float(v)}\n")

            # Free the figure
            try:
                import matplotlib.pyplot as _plt

                _plt.close(fig)
            except Exception:
                pass
        except Exception:
            logger.exception("Failed to plot training curves")

        # Assemble encoder manifest and stage outputs
        try:
            manifest_metric = _extract_metric_from_args(args)
            experiment_root = Path(
                os.getenv("EXPERIMENT_DIR")
                or os.getenv("EXP_ROOT")
                or args.ckpt_dir
            )
            artifacts_dir = Path(os.getenv("ARTIFACTS_DIR") or (experiment_root / "artifacts"))
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = artifacts_dir / "encoder_manifest.json"

            existing_manifest = _load_existing_manifest(manifest_path)
            existing_metric = _extract_metric_from_manifest(existing_manifest or {})

            should_write = True
            if existing_manifest is not None:
                if manifest_metric is None:
                    should_write = False
                else:
                    should_write = _metric_is_better(manifest_metric, existing_metric)

            manifest_payload: Dict[str, Any]
            if should_write:
                timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                hyperparameters: Dict[str, Any] = {
                    "gnn_type": args.gnn_type,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "mask_ratio": args.mask_ratio,
                    "epochs": args.epochs,
                    "save_every": save_every,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "temperature": args.temperature,
                    "ema_decay": args.ema_decay,
                    "contrastive": bool(getattr(args, "contrastive", False)),
                    "contiguity": bool(getattr(args, "contiguity", False)),
                    "add_3d": bool(getattr(args, "add_3d", False)),
                    "aug_rotate": bool(getattr(args, "aug_rotate", False)),
                    "aug_mask_angle": bool(getattr(args, "aug_mask_angle", False)),
                    "aug_dihedral": bool(getattr(args, "aug_dihedral", False)),
                    "aug_bond_deletion": bool(getattr(args, "aug_bond_deletion", False)),
                    "aug_atom_masking": bool(getattr(args, "aug_atom_masking", False)),
                    "aug_subgraph_removal": bool(getattr(args, "aug_subgraph_removal", False)),
                    "devices": getattr(args, "devices", 1),
                    "bf16": bool(getattr(args, "bf16", False)),
                    "pin_memory": bool(getattr(args, "pin_memory", False)),
                    "persistent_workers": bool(getattr(args, "persistent_workers", False)),
                    "prefetch_factor": getattr(args, "prefetch_factor", None),
                    "edge_dim": int(edge_dim) if edge_dim is not None else None,
                    "input_dim": int(input_dim),
                }
                if seeds:
                    try:
                        hyperparameters["seeds"] = [int(s) for s in seeds]
                    except Exception:
                        hyperparameters["seeds"] = list(seeds)
                if sample_ul is not None:
                    hyperparameters["sample_unlabeled"] = sample_ul
                if rows_per_file is not None:
                    hyperparameters["n_rows_per_file"] = rows_per_file

                manifest_paths: Dict[str, Any] = {
                    "encoder": os.path.abspath(ckpt_base),
                    "encoder_symlink": os.path.abspath(link),
                    "checkpoint_dir": os.path.abspath(args.ckpt_dir),
                }
                if cont_path:
                    manifest_paths["contrastive"] = os.path.abspath(cont_path)
                if plot_dir:
                    manifest_paths["plot_dir"] = os.path.abspath(plot_dir)
                if 'csv_path' in locals():
                    manifest_paths["loss_csv"] = os.path.abspath(csv_path)

                pretrain_exp_id = os.getenv("PRETRAIN_EXP_ID") or os.getenv("RUN_ID")
                if not pretrain_exp_id:
                    try:
                        pretrain_exp_id = experiment_root.name
                    except Exception:
                        pretrain_exp_id = None
                experiment_root_str = os.path.abspath(str(experiment_root))
            else:
                manifest_payload = existing_manifest or {}
                if manifest_metric is None:
                    logger.info(
                        "Skipping manifest update; no validation metric provided and existing manifest present",
                    )
                else:
                    logger.info(
                        "Skipping manifest update; validation metric did not improve (current=%s, previous=%s)",
                        manifest_metric,
                        existing_metric,
                    )

            hashes_block: Dict[str, Any] = {}
            if encoder_hash:
                hashes_block["encoder"] = encoder_hash

            if should_write:
                manifest_payload = {
                    "created_at": timestamp,
                    "pretrain_exp_id": pretrain_exp_id,
                    "experiment_root": experiment_root_str,
                    "paths": manifest_paths,
                    "hyperparameters": hyperparameters,
                    "run": _collect_run_metadata(wb),
                    "metrics": {},
                }
                manifest_payload["featurizer"] = {
                    "add_3d": bool(getattr(args, "add_3d", False)),
                    "edge_dim": int(edge_dim) if edge_dim is not None else None,
                    "input_dim": int(input_dim),
                    "contiguity": bool(getattr(args, "contiguity", False)),
                }
            elif manifest_payload:
                manifest_payload.setdefault("metrics", {})

            if should_write and hashes_block:
                manifest_payload["hashes"] = hashes_block
                if manifest_metric is not None:
                    manifest_payload["metrics"]["validation"] = manifest_metric

                with manifest_path.open("w", encoding="utf-8") as fh:
                    json.dump(manifest_payload, fh, indent=2, sort_keys=True)
                    fh.write("\n")
                logger.info("Wrote encoder manifest to %s", manifest_path)
            elif should_write:
                manifest_payload = existing_manifest or {}
                logger.info(
                    "Skipping manifest update; validation metric did not improve (current=%s, previous=%s)",
                    manifest_metric,
                    existing_metric,
                )

            if manifest_path.exists():
                manifest_str = str(manifest_path)
                _wb_summary(wb, {"encoder_manifest": manifest_str})
                _wb_log(wb, {"encoder_manifest": manifest_str})

            stage_payload = {
                "encoder_checkpoint": os.path.abspath(ckpt_base),
                "manifest_path": str(manifest_path),
                "manifest_updated": bool(should_write),
            }
            if encoder_hash:
                stage_payload["encoder_hash"] = encoder_hash
            if cont_path:
                stage_payload["contrastive_checkpoint"] = os.path.abspath(cont_path)
            if manifest_metric is not None:
                stage_payload["validation_metric"] = manifest_metric
            if stage_config:
                try:
                    stage_payload["stage_config"] = dict(stage_config)
                except Exception:
                    stage_payload["stage_config"] = stage_config
            _record_stage_outputs(stage_payload)
        except Exception:
            logger.exception("Failed to update encoder manifest")

    except Exception as e:
        _wb_log(wb, {"phase": "pretrain", "status": "error", "msg": str(e)})
        raise
    finally:
        _wb_finish(wb)

