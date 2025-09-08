from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from data.mdataset import GraphData
from utils.graph_ops import _encode_graph
from utils.pooling import global_mean_pool


def cmd_finetune(args: argparse.Namespace) -> None:
    """Fine‑tune a linear head on labelled data across multiple seeds resume & checkpoints."""
    logger.info("Starting finetune with args: %s", args)

    from utils.checkpoint import load_checkpoint, save_checkpoint
    try:
        from ..utils.checkpoint  import safe_load_checkpoint as _safe_load_checkpoint        # type: ignore[import-not-found]
        from ..utils.checkpoint  import load_state_dict_forgiving as _load_state_dict_forgiving      # type: ignore[import-not-found]
    except ImportError:
        # Fallback: absolute imports when run from repo root with PYTHONPATH set
        from utils.checkpoint import safe_load_checkpoint  as _safe_load_checkpoint        # type: ignore[import-not-found]
        from utils.checkpoint import load_state_dict_forgiving as _load_state_dict_forgiving        # type: ignore[import-not-found]

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
        labeled = load_directory_dataset(
            args.labeled_dir,
            label_col=args.label_col,
            add_3d=args.add_3d,
            num_workers=getattr(args, "num_workers", 0),
            cache_dir=getattr(args, "cache_dir", None),
        )  # type: ignore[arg-type]

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

    # --- choose metric & direction (maximize cls metrics, minimize losses/errors) ---
    metric_name = (getattr(args, "metric", "val_loss") or "").lower()
    higher_is_better = metric_name in {
        "acc", "accuracy",
        "auc", "auroc", "val_auc",
        "f1", "f1_macro", "f1_micro",
        "r2", "val_r2",
    }

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
            "val_mae":  ["mae_mean", "mae"],
            "val_auc":  ["auc", "auroc"],
            "acc":      ["accuracy", "val_acc"],
            "accuracy": ["acc", "val_acc"],
            "r2":       ["val_r2"],
        }
        for k in aliases.get(name, []):
            v = m.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    def _is_better(curr, best):
        return (curr > best) if higher_is_better else (curr < best)

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
                logger.warning("[finetune] no encoder weights found in %s; using random init", args.encoder)

        # If resuming a fine-tune checkpoint, it may contain a fresher encoder
        if "encoder" in resume_state:
            logger.info("Overriding encoder from resume checkpoint")
            _load_state_dict_forgiving(encoder, resume_state["encoder"])

        # Build linear head for fine-tuning
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
        params = _iter_params(encoder) + _iter_params(head)
        optimizer = (
            torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4) if params else None
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
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
            best_metric = -float("inf") if higher_is_better else float("inf")

            wrote_best = False
            last_epoch = start_epoch - 1
            for epoch in range(start_epoch, args.epochs):
                metrics = train_linear_head(
                    dataset=labeled,
                    encoder=encoder,
                    #head_type=getattr(args, "head", "linear"),  # <- change innvalid param for supervised?
                    task_type=args.task_type,
                    epochs=1,
                    max_batches=getattr(
                        args, "max_pretrain_batches", 0
                    ),  # ensure it does not crash for unit tests
                    time_budget_mins=getattr(
                        args, "time_budget_mins", 0
                    ),  # ensure it does not crash for unit tests
                    lr=args.lr,
                    batch_size=args.batch_size,
                    device=device,
                    patience=args.patience,
                    devices=args.devices,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    # dataloader & AMP knobs
                    num_workers=getattr(args, "num_workers", 0),
                    pin_memory=getattr(args, "pin_memory", True),
                    persistent_workers=getattr(args, "persistent_workers", True),
                    prefetch_factor=getattr(args, "prefetch_factor", 4),
                    bf16=getattr(args, "bf16", False),
                )
                
                current = _lookup_metric(metrics, metric_name)

                if current is not None and _is_better(current, best_metric):
                    best_metric = current
                    best_path = os.path.join(seed_dir, "ft_best.pt")
                    save_checkpoint(
                        best_path,
                        epoch=epoch,
                        encoder=encoder.state_dict(),
                        head=head.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        best_metric=best_metric,
                    )
                    wrote_best = True
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
                # advance LR schedule after the epoch
                try:
                    scheduler.step()
                except Exception:
                    pass
            # Fallback: if no best was recorded, promote last snapshot to best + head.pt
            if not wrote_best:
                logger.info("Attempting to write best")
                try:
                    # find latest epoch file we just wrote
                    snaps = [p for p in os.listdir(seed_dir) if p.startswith("ft_epoch_") and p.endswith(".pt")]
                    if snaps:
                        snaps.sort(key=lambda s: int(s.split("_")[-1].split(".")[0]))
                        last = os.path.join(seed_dir, snaps[-1])
                        best_path = os.path.join(seed_dir, "ft_best.pt")
                       
                        shutil.copy2(last, best_path)
                        from utils.checkpoint import safe_link_or_copy
                        mode = safe_link_or_copy(best_path, os.path.join(args.ckpt_dir, "head.pt"))
                        logger.info("Fallback best: head.pt (%s) -> %s", mode, best_path)
                        
                        logger.warning("No metric '%s' found; promoted %s to ft_best.pt", metric_name, snaps[-1])
                except Exception:
                    logger.warning("Failed to create fallback ft_best.pt", exc_info=True)


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
        from ..models.factory import build_encoder        # type: ignore[import-not-found]
        from ..utils.pooling import global_mean_pool      # type: ignore[import-not-found]
        from ..utils.checkpoint  import load_state_dict_forgiving as _load_state_dict_forgiving      # type: ignore[import-not-found]
    except ImportError:
        # Fallback: absolute imports when run from repo root with PYTHONPATH set
        from models.factory import build_encoder          # type: ignore[import-not-found]
        from utils.pooling import global_mean_pool        # type: ignore[import-not-found]
        from utils.checkpoint import load_state_dict_forgiving as _load_state_dict_forgiving        # type: ignore[import-not-found]

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

    enc = build_encoder(
        gnn_type=args.gnn_type,
        input_dim=dataset.graphs[0].x.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=None
        if dataset.graphs[0].edge_attr is None
        else dataset.graphs[0].edge_attr.shape[1],
    )
    _load_state_dict_forgiving(enc, state["encoder"])
    head = nn.Linear(enc.hidden_dim, 1)
    _load_state_dict_forgiving(head, state["head"])
    enc  = _to_dev(enc, device)
    head = _to_dev(head, device)
    enc.eval()
    head.eval()

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    for start in range(0, len(dataset), args.batch_size):
        batch_indices = list(range(start, min(start + args.batch_size, len(dataset))))
        batch_x, batch_adj, batch_ptr, batch_labels = dataset.get_batch(batch_indices)
       
        batch_x   = batch_x.to(device, non_blocking=True)
        batch_adj = batch_adj.to(device, non_blocking=True)
        batch_ptr = batch_ptr.to(device, non_blocking=True)
        with torch.no_grad():
            edge_idx = batch_adj.nonzero().T.detach().cpu().numpy()
            if edge_idx.size == 0:
                import numpy as _np
                edge_idx = _np.zeros((2, 0), dtype=_np.int64)
            g = GraphData(
                x=batch_x.detach().cpu().numpy(),
                edge_index=edge_idx,
            )
            node_emb = _encode_graph(enc, g)
            graph_emb = global_mean_pool(node_emb, batch_ptr.to(device)) if batch_ptr is not None else node_emb.mean(dim=0, keepdim=True)
            preds = head(graph_emb).squeeze(1)
            
        all_preds.append(preds.detach().to(torch.float32).cpu().numpy())
        all_targets.append(batch_labels.to(torch.float32).cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    if args.task_type == "classification":
        return compute_classification_metrics(y_true, y_pred)
    return compute_regression_metrics(y_true, y_pred)


