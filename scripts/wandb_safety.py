# scripts/mjepa/wandb_safety.py
from __future__ import annotations
from typing import Optional, Dict, Any, TYPE_CHECKING
import os, contextlib, json, time, pathlib

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run  # for the Optional["Run"] annotation

#TODO clean later
DEBUG = os.getenv("WBS_DEBUG", "1") == "1"
def _dbg(*a):
    if DEBUG: print("[wandb_safety]", *a)

def wb_get_or_init(args) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Ensure there is an active wandb run for sweep-run. Returns a run or None."""
    import wandb
    # already initialised by agent?
    run = getattr(wandb, "run", None)
    if run is not None:
        return run

    # respect disable flags
    if os.getenv("WANDB_MODE") == "disabled" or os.getenv("WANDB_DISABLED") in {"true", "1"}:
        _dbg("WANDB disabled by env; skipping init")
        return None

    # prefer shared helper if present
    try:
        from utils.logging import maybe_init_wandb
    except Exception:
        maybe_init_wandb = None

    if maybe_init_wandb is not None:
        run = maybe_init_wandb(
            enable=True,
            project=os.getenv("WANDB_PROJECT", "m-jepa"),
            config=vars(args) if args is not None else None,
            group=os.getenv("WANDB_RUN_GROUP"),
            job_type="sweep-run",
            tags=["sweep-run"],
            api_key=os.getenv("WANDB_API_KEY"),   
        )
        return getattr(wandb, "run", None)
        
    # hard fallback
    if run is None:
        _dbg("fallback wandb.init(...)")
        try:
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "m-jepa"),
                group=os.getenv("WANDB_RUN_GROUP"),
                job_type="sweep-run",
                config=vars(args) if args is not None else None,
                reinit=True,
            )
            _dbg("wandb.init -> run id:", getattr(run, "id", None))
            return run
        except Exception:
            _dbg("wandb.init raised; using wandb.run if present")
            return getattr(wandb, "run", None)

METRIC_CANDIDATES = ("val_rmse", "rmse_mean", "rmse", "probe_rmse_mean", "metric")


def wb_summary_update(payload: Dict[str, Any]) -> None:
    """Safe summary update: only writes if a run exists; never throws."""
    import wandb
    run = getattr(wandb, "run", None)
    if run is None:
        _dbg("wb_summary_update: no active run; skipping. keys=", list(payload.keys()))
        return
    # persist last payload to help debugging
    try:
        log_dir = os.getenv("LOG_DIR") or "./logs"
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(log_dir, "wb_last_payload.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        _dbg("could not write wb_last_payload.json:", e)

    try:
        # resolve RMSE candidate
        v_key = None
        v = payload.get("val_rmse")
        if v is None:
            for k in METRIC_CANDIDATES:
                if k == "val_rmse":
                    continue
                if payload.get(k) is not None:
                    try:
                        v = float(payload[k])
                    except Exception:
                        continue
                    v_key = k
                    break
        else:
            try:
                v = float(v)
            except Exception:
                v = None

        if v is not None:
            _dbg(f"logging val_rmse={v} (from key={v_key or 'val_rmse'})")
            wandb.log({"val_rmse": float(v)})
            run.summary["val_rmse"] = float(v)
        elif any(k in payload for k in METRIC_CANDIDATES):
            _dbg("no RMSE candidate in payload; keys=", list(payload.keys()))

        # val_mae aliasing
        if "val_mae" not in payload and payload.get("mae_mean") is not None:
            run.summary["val_mae"] = float(payload["mae_mean"])


        # classification: alias AUC candidates
        if "val_auc" not in run.summary:
            for k in ("val_auc", "auc", "roc_auc", "pr_auc"):
                if payload.get(k) is not None:
                    try:
                        run.summary["val_auc"] = float(payload[k])
                    except Exception:
                        pass
                    break

        try:
            run.summary.update(payload)
        except Exception as e:
            _dbg("wb_summary_update update() exception:", e)
            for key, value in payload.items():
                try:
                    run.summary[key] = value
                except Exception as inner:
                    _dbg(f"failed to set summary[{key!r}]:", inner)
    except Exception as e:
        _dbg("wb_summary_update exception:", e)

def wb_finish_safely() -> None:
    with contextlib.suppress(Exception):
        import wandb; wandb.finish()
