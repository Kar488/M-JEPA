from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

from models.encoder import GNNEncoder  # used for fallback run
from training.unsupervised import train_contrastive  # fallback


def _try_import_molclr():
    try:
        # adjust if your entrypoint differs; this is just an example
        from MolCLR.molclr import pretrain as molclr_pretrain

        return molclr_pretrain
    except Exception:
        return None


def _try_import_geomgcl():
    try:
        from GeomGCL.train import pretrain as geomgcl_pretrain

        return geomgcl_pretrain
    except Exception:
        return None


def _try_import_himol():
    try:
        from HiMol.train import pretrain as himol_pretrain

        return himol_pretrain
    except Exception:
        return None


def pretrain_baseline(
    method: str,
    *,
    dataset,
    input_dim: int,
    device: str = "cuda",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Uniform call to external baselines; falls back to our contrastive trainer if missing."""
    method = method.lower()
    cfg = dict(cfg or {})
    results: Dict[str, Any] = {}

    if method == "molclr":
        fn = _try_import_molclr()
        if fn is not None:
            return fn(dataset=dataset, device=device, **cfg)  # use their API
        warnings.warn(
            "MolCLR not found; falling back to internal contrastive pretrain."
        )

    if method == "geomgcl":
        fn = _try_import_geomgcl()
        if fn is not None:
            return fn(dataset=dataset, device=device, **cfg)
        warnings.warn(
            "GeomGCL not found; falling back to internal contrastive pretrain."
        )

    if method == "himol":
        fn = _try_import_himol()
        if fn is not None:
            return fn(dataset=dataset, device=device, **cfg)
        warnings.warn("HiMol not found; falling back to internal contrastive pretrain.")

    # ---- Fallback: our simple contrastive baseline ----
    encoder = GNNEncoder(
        input_dim=input_dim,
        hidden_dim=cfg.get("hidden_dim", 256),
        num_layers=cfg.get("num_layers", 3),
        gnn_type=cfg.get("gnn_type", "mpnn"),
    )
    losses = train_contrastive(
        dataset=dataset,
        encoder=encoder,
        projection_dim=cfg.get("projection_dim", 64),
        epochs=cfg.get("epochs", 50),
        batch_size=cfg.get("batch_size", 256),
        mask_ratio=cfg.get("mask_ratio", 0.2),
        lr=cfg.get("lr", 1e-4),
        device=device,
        temperature=cfg.get("temperature", 0.1),
        use_wandb=cfg.get("use_wandb", False),
        wandb_project=cfg.get("wandb_project", "m-jepa"),
        wandb_tags=cfg.get("wandb_tags"),
        ckpt_path=cfg.get("ckpt_path"),
        ckpt_every=cfg.get("ckpt_every", 10),
        use_scheduler=cfg.get("use_scheduler", True),
        warmup_steps=cfg.get("warmup_steps", 500),
    )
    results["loss_curve"] = losses
    results["encoder"] = encoder
    return results
