from __future__ import annotations

import argparse
import os
import sys
import time
import random
from typing import Dict, List

import numpy as np
import torch


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

    from utils.checkpoint import load_checkpoint, save_checkpoint

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
                getattr(args, "num_workers", 0),
            )
            t0 = time.time()

            unlabeled = load_directory_dataset(
                args.unlabeled_dir,
                add_3d=args.add_3d,
                num_workers=getattr(args, "num_workers", 0),
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
        if args.aug_rotate:
            kwargs["random_rotate"] = True
        if args.aug_mask_angle:
            kwargs["mask_angle"] = True
        if args.aug_dihedral:
            kwargs["perturb_dihedral"] = True

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
                    reg_lambda=1e-4,
                    use_wandb=args.use_wandb,
                    wandb_project=args.wandb_project,
                    wandb_tags=args.wandb_tags,
                    disable_tqdm=(not getattr(args, "force_tqdm", False))
                    and (not sys.stdout.isatty()),
                    # dataloader & AMP knobs
                    num_workers=getattr(args, "num_workers", 0),
                    pin_memory=getattr(args, "pin_memory", True),
                    persistent_workers=getattr(args, "persistent_workers", True),
                    prefetch_factor=getattr(args, "prefetch_factor", 4),
                    bf16=getattr(args, "bf16", False),
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
            rotate=args.aug_rotate,
            mask_angle=args.aug_mask_angle,
            dihedral=args.aug_dihedral,
        )

        # Optionally run contrastive baseline
        cont_losses: List[float] = []
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
                    use_wandb=args.use_wandb,
                    random_rotate=aug_cfg.rotate,
                    mask_angle=aug_cfg.mask_angle,
                    perturb_dihedral=aug_cfg.dihedral,
                    wandb_project=args.wandb_project,
                    wandb_tags=args.wandb_tags,
                    disable_tqdm=(not getattr(args, "force_tqdm", False))
                    and (not sys.stdout.isatty()),
                    # dataloader & AMP knobs
                    num_workers=getattr(args, "num_workers", 0),
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
        torch.save({"encoder": encoder.state_dict()}, ckpt_base)
        _wb_log(wb, {"jepa_checkpoint": ckpt_base})
        if args.contrastive:
            cont_path = f"{os.path.splitext(ckpt_base)[0]}_contrastive.pt"
            torch.save({"encoder": cont_encoder.state_dict()}, cont_path)
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
        try:
            curves = {"jepa": pretrain_losses}
            if cont_losses:
                curves["contrastive"] = cont_losses
            fig = plot_training_curves(curves, wb=wb)

            # Respect --plot-dir if provided; otherwise default to <ckpt_dir>/plots
            plot_dir = getattr(args, "plot_dir", None) or os.path.join(
                args.ckpt_dir, "plots"
            )
            os.makedirs(plot_dir, exist_ok=True)
            out_png = os.path.join(plot_dir, "pretrain_loss.png")
            fig.savefig(out_png, dpi=200)

            # Log to W&B only if a run exists (avoid preinit errors)
            try:
                import wandb as _wandb

                if (wb is not None) and (getattr(wb, "run", None) is not None):
                    wb.log({"pretrain/loss_plot": _wandb.Image(out_png)})
            except Exception:
                pass

            # Also write a CSV of epoch losses next to the checkpoint
            csv_path = os.path.join(args.ckpt_dir, "pretrain_losses.csv")
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

    except Exception as e:
        _wb_log(wb, {"phase": "pretrain", "status": "error", "msg": str(e)})
        raise
    finally:
        _wb_finish(wb)

