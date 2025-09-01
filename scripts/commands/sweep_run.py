from __future__ import annotations
import argparse
def cmd_sweep_run(args: argparse.Namespace) -> None:
    """
    Run one hyperparameter config (JEPA or contrastive) for W&B sweeps.
    Mirrors one row of grid-search, but logs directly to W&B.
    """
    from experiments.grid_search import Config, _run_one_config_method  

    from wandb_safety import wb_get_or_init as _wb_get_or_init
    from wandb_safety import wb_summary_update as _wb_summary_update
    from wandb_safety import wb_finish_safely as _wb_finish_safely

    to_bool = lambda v: bool(int(v)) if isinstance(v, (str,int)) else bool(v)
    add_3d               = to_bool(getattr(args, "add_3d", 0))
    aug_contiguity       = to_bool(getattr(args, "contiguity", 0))

    import json

    def _b(x):
        # robust 0/1/True/False/"0"/"1" → bool
        if isinstance(x, bool): return x
        try: return bool(int(str(x)))
        except Exception: return bool(x)

    G = lambda k, d=None: getattr(args, k, d)

    aug_kwargs = {
        "random_rotate":     _b(G("aug_rotate", 0)),
        "mask_angle":        _b(G("aug_mask_angle", 0)),
        "perturb_dihedral":  _b(G("aug_dihedral", 0)),
        "bond_deletion":     _b(G("aug_bond_deletion", 0)),
        "atom_masking":      _b(G("aug_atom_masking", 0)),
        "subgraph_removal":  _b(G("aug_subgraph_removal", 0)),
    }
    print("[sweep-run] args: " + json.dumps(vars(args), sort_keys=True, default=str))

    # Build config object
    cfg = Config(
        mask_ratio=args.mask_ratio,
        contiguous=aug_contiguity,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        ema_decay=args.ema_decay,
        add_3d=add_3d,
        pretrain_bs=args.pretrain_batch_size,
        finetune_bs=args.finetune_batch_size,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        lr=args.learning_rate,
        temperature=args.temperature,
        augmentations=AugmentationConfig(**aug_kwargs),
    )
    
    import os, re

    def _resolve_env_path(p: str) -> str:
        # turn ${env:VAR} -> ${VAR} then expand env/user/abspath
        p = re.sub(r"\$\{env:([^}]+)\}", r"${\1}", p)
        return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))

    _labeled_dir   = _resolve_env_path(args.labeled_dir)
    _unlabeled_dir = _resolve_env_path(args.unlabeled_dir)
    print(f"[sweep-run] resolved labeled_dir={_labeled_dir}")
    print(f"[sweep-run] resolved unlabeled_dir={_unlabeled_dir}")

    per_trial_cfg = {k: v for k, v in vars(args).items() if k != "func"}

    use_wandb = bool(int(getattr(args, "use_wandb", 1)))   # default: on
    project   = getattr(args, "wandb_project", None) or os.getenv("WANDB_PROJECT", "mjepa")
    tags      = getattr(args, "wandb_tags", None)

    _wb_get_or_init(args)

    # One-config run
    row = _run_one_config_method(
        cfg=cfg,
        method=args.training_method,  # "jepa" or "contrastive"

        unlabeled_dataset_fn=lambda add3d: load_directory_dataset(
            dirpath=_unlabeled_dir,
            label_col=None,                  # unlabeled set
            add_3d=add3d,
            max_graphs=args.sample_unlabeled,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
        ),
        eval_dataset_fn=lambda add3d: load_directory_dataset(
            dirpath=_labeled_dir,
            label_col=args.label_col,        # labeled set
            add_3d=add3d,
            max_graphs=args.sample_labeled,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
        ),

        task_type=args.task_type,
        seeds=[args.seed],
        device="cuda",
        use_wandb=True,
        ckpt_dir="outputs/sweep_ckpts",
        ckpt_every=50,
        use_scheduler=True,
        warmup_steps=1000,
        baseline_unlabeled_file=None,
        baseline_eval_file=None,
        baseline_smiles_col="smiles",
        baseline_label_col=args.label_col,
        use_scaffold=False,
        prebuilt_loaders=None,
        prebuilt_datasets=None,
        target_pretrain_samples=0,
        max_pretrain_batches=args.max_pretrain_batches,
        max_finetune_batches=args.max_finetune_batches,
        time_left=lambda: float("inf"),
        num_workers=int(getattr(args, "num_workers", 4)),
        pin_memory=bool(int(getattr(args, "pin_memory", 1))),
        persistent_workers=bool(int(getattr(args, "persistent_workers", 0))),
        prefetch_factor=int(getattr(args, "prefetch_factor", 2)),
        bf16=bool(int(getattr(args, "bf16", 0))),
    )

    _wb_summary_update(row)
    _wb_finish_safely()

