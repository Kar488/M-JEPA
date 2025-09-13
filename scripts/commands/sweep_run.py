from __future__ import annotations
import argparse
import hashlib, json

def cmd_sweep_run(args: argparse.Namespace) -> None:
    """
    Run one hyperparameter config (JEPA or contrastive) for W&B sweeps.
    Mirrors one row of grid-search, but logs directly to W&B.
    """
    from experiments.grid_search import Config, _run_one_config_method  

    from wandb_safety import wb_get_or_init as _wb_get_or_init
    from wandb_safety import wb_summary_update as _wb_summary_update
    from wandb_safety import wb_finish_safely as _wb_finish_safely

    # dataset + device helpers live in train_jepa.py; import them here so the
    # lambdas below resolve correctly in this module’s namespace.
    try:
        from scripts.train_jepa import load_directory_dataset, resolve_device
    except Exception:
        # fallback when executed with repo root on sys.path but not as a package
        try:
            from train_jepa import load_directory_dataset, resolve_device
        except Exception as e:
            raise ImportError("Could not import load_directory_dataset/resolve_device") from e

    # --- 0) Initialize W&B run FIRST (no config!), then read sampled config ---
    import os
    try:
        import wandb
    except Exception:
        wandb = None
    run = None
    if wandb is not None:
        # do NOT pass config here; agent provides sampled config
        wb = getattr(wandb, "run", None) or _wb_get_or_init(args)
    try:
        cfg = wandb.config.as_dict()
    except Exception:
        cfg = dict(getattr(wandb, "config", {}) or {})

    def _as_bool(v):
        if isinstance(v, bool): return v
        try: return bool(int(str(v)))
        except Exception: return bool(v)

    def _apply(src_key, dest_attr, cast=lambda x: x):
        if src_key in cfg:
            try:
                setattr(args, dest_attr, cast(cfg[src_key]))
            except Exception:
                pass

    # core model + training knobs commonly swept
    _apply("gnn_type",             "gnn_type",             lambda s: str(s).lower())
    _apply("gnn",                 "gnn_type",             lambda s: str(s).lower())
    _apply("hidden_dim",           "hidden_dim",           int)
    _apply("num_layers",           "num_layers",           int)
    _apply("add_3d",               "add_3d",               _as_bool)
    _apply("contiguous",           "contiguity",           _as_bool)
    _apply("contiguity",           "contiguity",           _as_bool)
    _apply("mask_ratio",           "mask_ratio",           float)
    _apply("ema_decay",            "ema_decay",            float)
    _apply("temperature",          "temperature",          float)
    _apply("pretrain_epochs",      "pretrain_epochs",      int)
    _apply("finetune_epochs",      "finetune_epochs",      int)
    _apply("pretrain_batch_size",  "pretrain_batch_size",  int)
    _apply("finetune_batch_size",  "finetune_batch_size",  int)
    _apply("pretrain_bs",          "pretrain_batch_size",  int)
    _apply("finetune_bs",           "finetune_batch_size", int)
    _apply("max_pretrain_batches", "max_pretrain_batches", int)
    _apply("max_finetune_batches", "max_finetune_batches", int)
    _apply("sample_unlabeled",     "sample_unlabeled",     int)
    _apply("sample_labeled",       "sample_labeled",       int)
    # learning rate may be named "learning_rate" or "lr"
    if "learning_rate" in cfg:
        args.learning_rate = float(cfg["learning_rate"])
    elif "lr" in cfg:
        args.learning_rate = float(cfg["lr"])

    # booleans used below
    to_bool = _as_bool
    add_3d         = to_bool(getattr(args, "add_3d", 0))
    aug_contiguity = to_bool(getattr(args, "contiguity", 0))
    gnn = str(getattr(args, "gnn_type", "")).lower()

    import os
    if os.environ.get("SWEEP_DUMP", "0") == "1":
        print(f"[sweep-run] gnn_type={gnn} hidden_dim={args.hidden_dim} num_layers={args.num_layers} "
              f"add_3d={to_bool(getattr(args,'add_3d',0))} contiguity={to_bool(getattr(args,'contiguity',0))} "
               f"lr={getattr(args,'learning_rate',None)}",
              flush=True)
        return

    if gnn in ("schnet3d", "schnet"):
        if not add_3d:
            print("[guard] gnn_type=schnet3d requires --add-3d=1; enabling it.", flush=True)
            add_3d = True
    else:
        if add_3d:
            print(f"[guard] gnn_type={gnn} is a 2D backbone; forcing --add-3d=0.", flush=True)
            add_3d = False

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
    #print("[sweep-run] args: " + json.dumps(vars(args), sort_keys=True, default=str))

    # Build config object
    cfg = Config(
        mask_ratio=args.mask_ratio,
        contiguous=aug_contiguity,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=gnn,
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

    # NOTE: we already initialized earlier; do NOT re-init with args/config here
    wb = wandb.run if wandb is not None else None
    # re-open if an inner path finished it (without config!)
    if wandb is not None and wandb.run is None:
        _wb_get_or_init(project=getattr(args, "wandb_project", None),
                        job_type="sweep-run",
                        mode=os.getenv("WANDB_MODE"))
    
    mr = getattr(args, "mask_ratio", None)
    if mr is not None: mr = round(float(mr), 6)

    # Normalise a config subset that should match across methods for a "pair"
    pid_cfg = {
        "gnn_type":    gnn,
        "hidden_dim":  int(args.hidden_dim),
        "num_layers":  int(args.num_layers),
        # use the actual contiguity knob name; no stray 'contiguous'
        "contiguity":  int(getattr(args, "contiguity", 0)),
    }

    pair_id = hashlib.sha1(json.dumps(pid_cfg, sort_keys=True).encode()).hexdigest()[:8]
    print(f"[sweep-run] pair_id={pair_id} | gnn_type={args.gnn_type} | "
          f"hidden_dim={args.hidden_dim} | num_layers={args.num_layers} | "
          f"contiguity={int(getattr(args,'contiguity',0))} | add_3d={int(bool(getattr(args,'add_3d',0)))}",
          flush=True)

    if wb is not None:
        wb.config.update({
        "training_method": args.training_method,
        "pair_id": pair_id,
        "seed": getattr(args, "seed", None),   # keep seed visible separately
        "gnn_type": gnn,
        "add_3d": int(add_3d),   # <-- ensure Phase-2 sees the gated value
        }, allow_val_change=True)
    else:
        print("⚠️ W&B is disabled or failed to init, skipping config update")

    #print(f"[sweep-run] training_method={args.training_method} pid_cfg={pid_cfg} pair_id={pair_id}", flush=True)

    import time as _t
    _deadline = None
    if getattr(args, "time_budget_mins", 0):
        _deadline = _t.perf_counter() + float(args.time_budget_mins) * 60.0

    def _time_left() -> float:
        if _deadline is None:
            return float("inf")
        # minutes remaining
        return max(0.0, (_deadline - _t.perf_counter()) / 60.0)
    
    import os, pathlib
    if args.cache_dir:
        pathlib.Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

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
        device=resolve_device("cuda" if getattr(args, "devices", 1) > 0 else "cpu"),
        use_wandb=bool(int(getattr(args, "use_wandb", 1))),
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
        target_pretrain_samples=int(getattr(args, "target_pretrain_samples", 0)),
        max_pretrain_batches=int(getattr(args, "max_pretrain_batches", 0)),
        max_finetune_batches=int(getattr(args, "max_finetune_batches", 0)),
        time_left=_time_left,
        num_workers=int(getattr(args, "num_workers", 4)),
        pin_memory=bool(int(getattr(args, "pin_memory", 0))),
        persistent_workers=bool(int(getattr(args, "persistent_workers", 0))),
        prefetch_factor=int(getattr(args, "prefetch_factor", 2)),
        bf16=bool(int(getattr(args, "bf16", 0))),
    )

    # --- normalize result into a payload dict for W&B summary update ---
    if isinstance(row, dict):
        payload = row
    elif isinstance(row, (float, int)):
        payload = {"val_rmse": float(row)} 
    else:
        # tuple like (metrics, artifacts), or None; grab the first dict if present
        if isinstance(row, tuple) and row and isinstance(row[0], dict):
            payload = row[0]
        else:
            payload = {}

    if args.task_type == "regression":
        if "val_rmse" not in payload:
            for k in ("metric", "rmse", "rmse_mean", "probe_rmse_mean"):
                v = payload.get(k)
                if v is not None:
                    try:
                        rmse = float(v)
                        print(f"[sweep-run] publishing val_rmse={rmse:.6f} (from key={k})", flush=True)
                        payload["val_rmse"] = rmse
                    except Exception:
                        pass
                    break
    else:
        # classification: ensure val_auc lands in summary
        if "val_auc" not in payload:
            for k in ("auc", "roc_auc", "pr_auc"):
                v = payload.get(k)
                if v is not None:
                    try:
                        auc = float(v)
                        print(f"[sweep-run] publishing val_auc={auc:.6f} (from key={k})", flush=True)
                        payload["val_auc"] = auc
                    except Exception:
                        pass
                    break
               
    _wb_get_or_init(args)           # re-open if an inner path finished it
    _wb_summary_update(payload)
    _wb_finish_safely()

