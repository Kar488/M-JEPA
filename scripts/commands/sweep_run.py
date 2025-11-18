from __future__ import annotations
import argparse
import hashlib, json
import os
import pathlib
import time
from typing import Optional, Dict, Any

from . import dataset_cache

try:
    from utils.wandb_filters import silence_pydantic_field_warnings
except Exception:  # pragma: no cover - helper only available in repo context
    def silence_pydantic_field_warnings() -> None:  # type: ignore
        return


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(str(raw))
    except Exception:
        return None


def _infer_best_step(payload: Dict[str, Any]) -> int:
    for key in ("best_step", "best_epoch", "epoch", "epochs", "step", "global_step"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return 0


def _local_result_file(exp_id: Optional[str], config_idx: Optional[int], seed: Optional[int]) -> Optional[pathlib.Path]:
    if exp_id is None or config_idx is None or seed is None:
        return None
    base = pathlib.Path("/data/mjepa/experiments") / exp_id / "recheck_results"
    return base / f"cfg{config_idx}_seed{seed}.json"

def cmd_sweep_run(args: argparse.Namespace) -> None:
    """
    Run one hyperparameter config (JEPA or contrastive) for W&B sweeps.
    Mirrors one row of grid-search, but logs directly to W&B.
    """
    try:
        from experiments.grid_search import (
            AugmentationConfig,
            Config,
            _run_one_config_method,
        )
    except ImportError:
        from experiments.grid_search import Config, _run_one_config_method  # type: ignore

        from types import SimpleNamespace

        def AugmentationConfig(**kwargs):  # type: ignore
            return SimpleNamespace(**kwargs)

    from wandb_safety import wb_summary_update as _wb_summary_update
    from wandb_safety import wb_finish_safely as _wb_finish_safely

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        try:
            return bool(int(str(v)))
        except Exception:
            return bool(v)

    # dataset + device helpers live in train_jepa.py; import them here so the
    # lambdas below resolve correctly in this module’s namespace.
    try:
        from scripts.train_jepa import load_directory_dataset, resolve_device
    except Exception:
        # fallback when executed with repo root on sys.path but not as a package
        try:
            from train_jepa import load_directory_dataset, resolve_device
        except Exception as e:
            raise ImportError(
                "Could not import load_directory_dataset/resolve_device"
            ) from e
    # --- 0) Initialize W&B run FIRST (no config!), then read sampled config ---
    try:
        silence_pydantic_field_warnings()
        import wandb
    except Exception:
        wandb = None
    sweep_cfg = {}
    if wandb is not None and _as_bool(getattr(args, "use_wandb", 1)):
        wb = getattr(wandb, "run", None)
        for cfg_src in (getattr(wb, "config", None), getattr(wandb, "config", None)):
            if cfg_src is None:
                continue
            try:
                if hasattr(cfg_src, "as_dict"):
                    sweep_cfg = cfg_src.as_dict()
                else:
                    sweep_cfg = dict(cfg_src)  # type: ignore[arg-type]
                break
            except Exception:
                continue

    # Flatten nested configs so "model.gnn_type" etc. are visible to _apply
    def _flatten(d, parent_key: str = "", sep: str = "."):
        out = {}
        for k, v in (d or {}).items():
            nk = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                out.update(_flatten(v, nk, sep))
            else:
                out[nk] = v
                out[k] = v  # also expose short key (last segment)
        return out
    sweep_cfg = _flatten(sweep_cfg or {})
    try:
        sample_keys = sorted(list(sweep_cfg.keys()))[:10]
    except Exception:
        sample_keys = []
    print(f"[sweep-run] cfg keys sample: {sample_keys}", flush=True)

    def _apply(src_key, dest_attr, cast=lambda x: x):
        if src_key in sweep_cfg:
            try:
                setattr(args, dest_attr, cast(sweep_cfg[src_key]))
            except Exception:
                pass

    def _apply_any(src_keys, dest_attr, cast=lambda x: x):
        for k in src_keys:
            if k in sweep_cfg:
                try:
                    setattr(args, dest_attr, cast(sweep_cfg[k]))
                    return
                except Exception:
                    pass

    def _apply_bool(src_keys, dest_attr):
        for k in src_keys:
            if k not in sweep_cfg:
                continue
            raw = sweep_cfg[k]
            if isinstance(raw, dict):
                if "value" in raw:
                    raw = raw["value"]
                elif len(raw) == 1:
                    # tolerate single-key dicts like {"min": 0}
                    raw = next(iter(raw.values()))
            try:
                setattr(args, dest_attr, _as_bool(raw))
                return
            except Exception:
                continue

    # core model + training knobs commonly swept
    _apply_any(
        ["gnn_type", "model.gnn_type", "model/gnn_type", "backbone", "model.backbone"],
        "gnn_type",
        lambda s: str(s).lower(),
    )
    _apply_any(
        ["contiguity", "model.contiguity", "contiguous", "model.contiguous"],
        "contiguity",
        _as_bool,
    )
    _apply_any(
        ["pretrain_bs", "pretrain_batch_size", "train.pretrain_bs"],
        "pretrain_batch_size",
        int,
    )
    _apply_any(
        ["finetune_bs", "finetune_batch_size", "train.finetune_bs"],
        "finetune_batch_size",
        int,
    )

    _apply_bool(
        [
            "cache_datasets",
            "cache-datasets",
            "cache_datasets.value",
            "cache-datasets.value",
            "parameters.cache_datasets",
            "parameters.cache-datasets",
            "parameters.cache_datasets.value",
            "parameters.cache-datasets.value",
        ],
        "cache_datasets",
    )

    _apply_any(["hidden_dim", "model.hidden_dim", "width"], "hidden_dim", int)
    _apply_any(["num_layers", "model.num_layers", "layers"], "num_layers", int)
    _apply_any(["mask_ratio", "model.mask_ratio"], "mask_ratio", float)

    _apply("ema_decay", "ema_decay", float)
    _apply("temperature", "temperature", float)
    _apply("pretrain_epochs", "pretrain_epochs", int)
    _apply("finetune_epochs", "finetune_epochs", int)
    _apply("pretrain_batch_size", "pretrain_batch_size", int)
    _apply("finetune_batch_size", "finetune_batch_size", int)
    _apply("max_pretrain_batches", "max_pretrain_batches", int)
    _apply("max_finetune_batches", "max_finetune_batches", int)
    _apply("sample_unlabeled", "sample_unlabeled", int)
    _apply("sample_labeled", "sample_labeled", int)
    
    # learning rate may be named "learning_rate" or "lr"
    if "learning_rate" in sweep_cfg:
        args.learning_rate = float(sweep_cfg["learning_rate"])
    elif "lr" in sweep_cfg:
        args.learning_rate = float(sweep_cfg["lr"])

    # booleans used below
    to_bool = _as_bool
    aug_contiguity = to_bool(getattr(args, "contiguity", 0))
    gnn = str(getattr(args, "gnn_type", "")).lower()
    add_3d = gnn in ("schnet3d", "schnet")

    import os
    if os.environ.get("SWEEP_DUMP", "0") == "1":
        print(
            f"[sweep-run] gnn_type={gnn} hidden_dim={args.hidden_dim} num_layers={args.num_layers} "
            f"add_3d={int(add_3d)} contiguity={to_bool(getattr(args,'contiguity',0))} "
            f"lr={getattr(args,'learning_rate',None)}",
            flush=True,
        )
        return

    import json

    def _b(x):
        # robust 0/1/True/False/"0"/"1" → bool
        if isinstance(x, bool):
            return x
        try:
            return bool(int(str(x)))
        except Exception:
            return bool(x)

    G = lambda k, d=None: getattr(args, k, d)

    aug_kwargs = {
        "random_rotate": _b(G("aug_rotate", 0)),
        "mask_angle": _b(G("aug_mask_angle", 0)),
        "perturb_dihedral": _b(G("aug_dihedral", 0)),
        "bond_deletion": _b(G("aug_bond_deletion", 0)),
        "atom_masking": _b(G("aug_atom_masking", 0)),
        "subgraph_removal": _b(G("aug_subgraph_removal", 0)),
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
    
    import os

    _labeled_dir = dataset_cache.resolve_env_path(args.labeled_dir)
    _unlabeled_dir = dataset_cache.resolve_env_path(args.unlabeled_dir)

    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir:
        args.cache_dir = dataset_cache.resolve_env_path(str(cache_dir))
        print(f"[sweep-run] resolved cache_dir={args.cache_dir}", flush=True)
    else:
        print("[sweep-run] cache_dir not provided; using default cache roots", flush=True)
        
    mr = getattr(args, "mask_ratio", None)
    if mr is not None:
        mr = round(float(mr), 6)

    # Normalise a config subset that should match across methods for a "pair"
    pid_cfg = {
        "gnn_type": gnn,
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        # use the actual contiguity knob name; no stray 'contiguous'
        "contiguity": int(getattr(args, "contiguity", 0)),
    }

    pair_id = hashlib.sha1(json.dumps(pid_cfg, sort_keys=True).encode()).hexdigest()[:8]
    print(
        f"[sweep-run] pair_id={pair_id} | gnn_type={args.gnn_type} | "
        f"hidden_dim={args.hidden_dim} | num_layers={args.num_layers} | "
        f"contiguity={int(getattr(args,'contiguity',0))} | add_3d={int(add_3d)}",
        flush=True,
    )

    def _update_run_config(run, payload):
        if run is None or not payload:
            return False
        config_obj = getattr(run, "config", None)
        if config_obj is None:
            return False
        try:
            config_obj.update(payload, allow_val_change=True)
            return True
        except Exception:
            try:
                for key, value in payload.items():
                    config_obj[key] = value
            except Exception:
                return False
            return True

    using_wandb = bool(int(getattr(args, "use_wandb", 1)))
    config_updated = False
    config_payload = {}
    if using_wandb:
        upd = {"pair_id": pair_id}
        if "training_method" not in sweep_cfg:
            upd["training_method"] = args.training_method
        if "seed" not in sweep_cfg:
            upd["seed"] = getattr(args, "seed", None)
        if "gnn_type" not in sweep_cfg:
            upd["gnn_type"] = gnn
        if "add_3d" not in sweep_cfg:
            upd["add_3d"] = int(add_3d)  # ensure Phase-2 sees the gated value
        config_payload = {k: v for k, v in upd.items() if v is not None}

    import time as _t
    _deadline = None
    if getattr(args, "time_budget_mins", 0):
        _deadline = _t.perf_counter() + float(args.time_budget_mins) * 60.0

    def _time_left() -> float:
        if _deadline is None:
            return float("inf")
        # minutes remaining
        return max(0.0, (_deadline - _t.perf_counter()) / 60.0)
    
    if args.cache_dir:
        pathlib.Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    dataset_cache_enabled = bool(int(getattr(args, "cache_datasets", 0)))
    dataset_cache_dir = dataset_cache.prepare_cache_root(
        args.cache_dir, enabled=dataset_cache_enabled
    )
    if dataset_cache_enabled and dataset_cache_dir:
        print(
            f"[sweep-run] dataset caching enabled (root={dataset_cache_dir})",
            flush=True,
        )
    else:
        print("[sweep-run] dataset caching disabled", flush=True)

    def _load_or_build_dataset(kind: str, payload: dict, builder):
        log = lambda msg: print(f"[sweep-run] {msg}", flush=True)

        return dataset_cache.load_or_build_dataset(
            kind,
            payload,
            builder,
            dataset_cache_dir,
            log=log,
        )

    def _build_unlabeled_dataset(add3d: bool):
        return load_directory_dataset(
            dirpath=_unlabeled_dir,
            label_col=None,  # unlabeled set
            add_3d=add3d,
            max_graphs=args.sample_unlabeled,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
        )

    def _build_labeled_dataset(add3d: bool):
        return load_directory_dataset(
            dirpath=_labeled_dir,
            label_col=args.label_col,  # labeled set
            add_3d=add3d,
            max_graphs=args.sample_labeled,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
        )

    unlabeled_ds = _load_or_build_dataset(
        "unlabeled",
        {
            "path": _unlabeled_dir,
            "add_3d": bool(add_3d),
            "sample": int(getattr(args, "sample_unlabeled", 0)),
        },
        lambda: _build_unlabeled_dataset(add_3d),
    )
    labeled_ds = _load_or_build_dataset(
        "labeled",
        {
            "path": _labeled_dir,
            "add_3d": bool(add_3d),
            "sample": int(getattr(args, "sample_labeled", 0)),
            "label_col": args.label_col,
        },
        lambda: _build_labeled_dataset(add_3d),
    )

    prebuilt_datasets = (unlabeled_ds, labeled_ds, labeled_ds)

    # One-config run
    row = _run_one_config_method(
        cfg=cfg,
        method=args.training_method,  # "jepa" or "contrastive"
        unlabeled_dataset_fn=None,
        eval_dataset_fn=None,

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
        prebuilt_datasets=prebuilt_datasets,
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
        payload = dict(row)
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
                        print(
                            f"[sweep-run] publishing val_rmse={rmse:.6f} (from key={k})",
                            flush=True,
                        )
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
                        print(
                            f"[sweep-run] publishing val_auc={auc:.6f} (from key={k})",
                            flush=True,
                        )
                        payload["val_auc"] = auc
                    except Exception:
                        pass
                    break
               
    metadata = {
        "pair_id": pair_id,
        "training_method": getattr(args, "training_method", None),
        "seed": getattr(args, "seed", None),
        "gnn_type": gnn,
        "add_3d": int(add_3d),
        "contiguity": int(getattr(args, "contiguity", 0)),
    }
    for key, value in metadata.items():
        if key not in payload and value is not None:
            payload[key] = value

    config_idx = _env_int("RECHECK_CONFIG_INDEX")
    seed_idx = _env_int("RECHECK_SEED")
    exp_id = os.getenv("RECHECK_EXP_ID")
    run_name = os.getenv("WANDB_NAME")

    raw_val = payload.get("val_rmse")
    try:
        val_rmse = float(raw_val) if raw_val is not None else None
    except Exception:
        val_rmse = None
    if val_rmse is not None:
        payload["val_rmse"] = val_rmse

    best_step = _infer_best_step(payload)
    payload["best_step"] = best_step

    try:
        silence_pydantic_field_warnings()
        import wandb as _wandb_mod  # type: ignore
    except Exception:
        _wandb_mod = None

    _wandb_wait_timeout = float(os.getenv("WANDB_SWEEP_INIT_TIMEOUT", 20.0))

    def _wait_for_wandb_run(timeout_s: float = _wandb_wait_timeout) -> Optional[Any]:
        if _wandb_mod is None:
            return None
        run_obj = getattr(_wandb_mod, "run", None)
        if run_obj is not None or not using_wandb:
            return run_obj
        poll_interval = 0.5
        deadline = time.perf_counter() + float(timeout_s)
        while run_obj is None and time.perf_counter() < deadline:
            time.sleep(poll_interval)
            run_obj = getattr(_wandb_mod, "run", None)
        return run_obj

    run = _wait_for_wandb_run()
    if run is None and using_wandb:
        print(
            "[sweep-run] wandb.run not initialised yet; delaying summary sync",
            flush=True,
        )
    if run is not None:
        if val_rmse is not None:
            run.summary["val_rmse"] = float(val_rmse)
        run.summary["best_step"] = int(best_step)
        if pair_id is not None:
            try:
                run.summary["pair_id"] = pair_id
            except Exception:
                pass
        if run_name and getattr(run, "name", None) != run_name:
            try:
                run.name = run_name
                run.save()
            except Exception:
                pass
    if using_wandb and config_payload:
        if run is None:
            run = _wait_for_wandb_run()
        if run is not None:
            config_updated = _update_run_config(run, config_payload)
        if not config_updated:
            print("⚠️ W&B is disabled or failed to init, skipping config update", flush=True)

    should_publish_summary = True
    if using_wandb:
        if run is None:
            run = _wait_for_wandb_run()
        if run is None:
            should_publish_summary = False
            print(
                "⚠️ W&B run never became active; canonical summary metrics were not synced",
                flush=True,
            )
    if should_publish_summary:
        _wb_summary_update(payload)

    result_path = _local_result_file(exp_id, config_idx, seed_idx)
    if result_path is not None:
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = result_path.with_suffix(".tmp")
            artifact = {
                "val_rmse": val_rmse,
                "best_step": int(best_step),
                "run_name": run_name,
                "config_idx": config_idx,
                "seed": seed_idx,
                "updated_at": time.time(),
            }
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump({k: v for k, v in artifact.items() if v is not None}, handle, indent=2)
            os.replace(tmp_path, result_path)
        except Exception as exc:
            print(f"[sweep-run] unable to write fallback metrics to {result_path}: {exc}", flush=True)

    _wb_finish_safely()

