# scripts/mjepa/wandb_safety.py
from __future__ import annotations
from typing import Optional, Dict, Any, TYPE_CHECKING
import os, json, time, pathlib

try:
    from utils.wandb_filters import silence_pydantic_field_warnings
except Exception:  # pragma: no cover - helper available in packaged installs
    def silence_pydantic_field_warnings() -> None:  # type: ignore
        return

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run  # for the Optional["Run"] annotation

#TODO clean later
DEBUG = os.getenv("WBS_DEBUG", "1") == "1"
def _dbg(*a):
    if DEBUG: print("[wandb_safety]", *a)


def _persist_payload(payload: Dict[str, Any]) -> None:
    """Best-effort write of the latest summary payload for offline debugging."""
    try:
        log_dir = os.getenv("LOG_DIR") or "./logs"
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(log_dir, "wb_last_payload.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as exc:
        _dbg("could not write wb_last_payload.json:", exc)

def wb_get_or_init(args) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Ensure there is an active wandb run for sweep-run. Returns a run or None."""
    silence_pydantic_field_warnings()
    import wandb
    # already initialised by agent?
    run = getattr(wandb, "run", None)
    if run is not None:
        return run

    wandb_disabled = os.getenv("WANDB_MODE") == "disabled" or os.getenv("WANDB_DISABLED") in {"true", "1", "True"}

    # prefer shared helper if present
    try:
        from utils.logging import maybe_init_wandb
    except Exception:
        maybe_init_wandb = None

    if maybe_init_wandb is not None:
        maybe_init_wandb(
            enable=not wandb_disabled,
            project=os.getenv("WANDB_PROJECT", "m-jepa"),
            config=vars(args) if args is not None else None,
            group=os.getenv("WANDB_RUN_GROUP"),
            job_type="sweep-run",
            tags=["sweep-run"],
            api_key=os.getenv("WANDB_API_KEY"),
        )
        run = getattr(wandb, "run", None)
        if run is not None:
            return run

    # respect disable flags once helper had a chance to opt-in
    if wandb_disabled:
        _dbg("WANDB disabled by env; skipping init")
        return None

    # hard fallback
    _dbg("fallback wandb.init(...)")
    try:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "m-jepa"),
            entity=os.getenv("WANDB_ENTITY"),
            name=os.getenv("WANDB_NAME"),
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
    silence_pydantic_field_warnings()
    try:
        import wandb  # type: ignore
    except Exception:
        wandb = None  # type: ignore
    run = getattr(wandb, "run", None) if wandb is not None else None
    wandb_disabled = os.getenv("WANDB_MODE") == "disabled" or os.getenv("WANDB_DISABLED") in {"1", "true", "True"}
    # persist last payload to help debugging/offline export even when wandb is disabled
    _persist_payload(payload)
    if run is None:
        if not wandb_disabled:
            _dbg("wb_summary_update: no active run; skipping. keys=", list(payload.keys()))
        return

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

        best_step_val = payload.get("best_step")
        try:
            best_step = int(best_step_val) if best_step_val is not None else None
        except Exception:
            best_step = None

        log_payload = {}
        if v is not None:
            log_payload["val_rmse"] = float(v)
        if best_step is not None:
            log_payload["best_step"] = int(best_step)
        if log_payload:
            _dbg(
                "logging metrics:",
                {k: log_payload[k] for k in sorted(log_payload)},
            )
            try:
                wandb.log(log_payload)
            except Exception as exc:
                _dbg("wandb.log failed:", exc)
        elif any(k in payload for k in METRIC_CANDIDATES):
            _dbg("no RMSE candidate in payload; keys=", list(payload.keys()))

        if v is not None:
            run.summary["val_rmse"] = float(v)
        if best_step is not None:
            run.summary["best_step"] = int(best_step)

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

def wb_finish_safely(timeout: float = 30.0) -> None:
    try:
        silence_pydantic_field_warnings()
        import wandb
    except Exception:
        return

    finish = getattr(wandb, "finish", None)
    if not callable(finish):
        return

    start = time.time()
    try:
        try:
            finish(quiet=True)
        except TypeError:
            # Older mocks (and some unit tests) expose finish() without a
            # ``quiet`` keyword. Retry without the argument so the call still
            # happens and errors are surfaced for callers.
            finish()
    except Exception:
        pass

    try:
        while getattr(wandb, "run", None) is not None and time.time() - start < timeout:
            time.sleep(0.5)
    except Exception:
        pass
