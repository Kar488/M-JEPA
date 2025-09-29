from __future__ import annotations

import argparse
import os
import sys
import time


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
            "temperatures": args.temperatures,
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
            temperatures=tuple(args.temperatures),
            device=args.device,
            use_wandb=args.use_wandb,
            ckpt_dir=args.ckpt_dir,
            ckpt_every=args.ckpt_every,
            use_scheduler=args.use_scheduler,
            warmup_steps=args.warmup_steps,
            out_csv=args.out_csv,
            # If your run_grid_search signature supports these, they’ll be used;
            # otherwise they’ll be ignored (or remove them here).
            target_pretrain_samples=getattr(args, "target_pretrain_samples", 0),
            max_pretrain_batches=getattr(args, "max_pretrain_batches", 0),
            max_finetune_batches=getattr(args, "max_finetune_batches", 0),
            time_budget_mins=getattr(args, "time_budget_mins", 0),
            disable_tqdm=(not getattr(args, "force_tqdm", False))
            and (not sys.stdout.isatty()),
            # dataloader & AMP knobs
            num_workers=getattr(args, "num_workers", -1),
            pin_memory=getattr(args, "pin_memory", True),
            persistent_workers=getattr(args, "persistent_workers", True),
            prefetch_factor=getattr(args, "prefetch_factor", 4),
            bf16=getattr(args, "bf16", False),
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

