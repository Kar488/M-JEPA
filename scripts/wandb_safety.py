# scripts/mjepa/wandb_safety.py
from __future__ import annotations
from typing import Optional, Dict, Any
import os, contextlib

def wb_get_or_init(args) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Ensure there is an active wandb run for sweep-run. Returns a run or None."""
    import wandb
    # already initialised by agent?
    run = getattr(wandb, "run", None)
    if run is not None:
        return run

    # respect disable flags
    if os.getenv("WANDB_MODE") == "disabled" or os.getenv("WANDB_DISABLED") in {"true", "1"}:
        return None

    # prefer shared helper if present
    try:
        from utils.logging import maybe_init_wandb
    except Exception:
        maybe_init_wandb = None

    if maybe_init_wandb is not None:
        run = maybe_init_wandb(
            enable=True,
            project=os.getenv("WANDB_PROJECT", "mjepa"),
            config=vars(args) if args is not None else None,
            #group=os.getenv("WANDB_RUN_GROUP"),
            #job_type="sweep-run",
            tags=["sweep-run"],
            
        )
        if run is not None:
            return run

    # hard fallback
    try:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "mjepa"),
            group=os.getenv("WANDB_RUN_GROUP"),
            job_type="sweep-run",
            config=vars(args) if args is not None else None,
            reinit=True,
        )
        return run
    except Exception:
        return getattr(wandb, "run", None)

def wb_summary_update(payload: Dict[str, Any]) -> None:
    """Safe summary update: only writes if a run exists; never throws."""
    import wandb
    run = getattr(wandb, "run", None)
    if run is None:
        return
    with contextlib.suppress(Exception):
        # val_rmse aliasing
        if payload.get("val_rmse") is None:
            for k in ("rmse_mean", "probe_rmse_mean"):
                if k in payload and payload[k] is not None:
                    run.summary["val_rmse"] = float(payload[k])
                    break
        # val_mae aliasing
        if "val_mae" not in payload and "mae_mean" in payload and payload["mae_mean"] is not None:
            run.summary["val_mae"] = float(payload["mae_mean"])
        run.summary.update(payload)

def wb_finish_safely() -> None:
    with contextlib.suppress(Exception):
        import wandb; wandb.finish()
