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
    _dbg(
        "wb_get_or_init: disabled=", wandb_disabled, "use_wandb_arg=", getattr(args, "use_wandb", None)
    )

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

RMSE_CANDIDATES = ("val_rmse", "rmse_mean", "rmse", "probe_rmse_mean", "metric")
AUC_CANDIDATES = ("val_auc", "auc", "roc_auc", "pr_auc")


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _wait_for_run(module, timeout_s: float) -> Optional["Run"]:
    run = getattr(module, "run", None)
    if run is not None:
        return run
    poll_interval = 0.5
    deadline = time.time() + float(timeout_s)
    while run is None and time.time() < deadline:
        time.sleep(poll_interval)
        run = getattr(module, "run", None)
    return run


def _coerce_from_candidates(payload: Dict[str, Any], candidates, coerce_fn):
    for key in candidates:
        value = coerce_fn(payload.get(key))
        if value is not None:
            return value
        mean_value = coerce_fn(payload.get(f"{key}_mean"))
        if mean_value is not None:
            return mean_value
    return None


def wb_summary_update(payload: Dict[str, Any]) -> None:
    """Safe summary update: waits for an active run and logs canonical metrics."""
    silence_pydantic_field_warnings()
    wandb_disabled = os.getenv("WANDB_MODE") == "disabled" or os.getenv("WANDB_DISABLED") in {
        "1",
        "true",
        "True",
    }
    # persist last payload to help debugging/offline export even when wandb is disabled
    _persist_payload(payload)

    try:
        import wandb  # type: ignore
    except Exception:
        wandb = None  # type: ignore

    if wandb is None:
        if not wandb_disabled:
            _dbg("wb_summary_update: wandb import failed; skipping summary sync")
        else:
            _dbg("wb_summary_update: wandb disabled and import failed; exiting early")
        return

    timeout = float(os.getenv("WANDB_SWEEP_INIT_TIMEOUT", 20.0))
    run = getattr(wandb, "run", None)
    api_run = None
    run_id = os.getenv("WANDB_RUN_ID")
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT") or "m-jepa"
    target = "/".join([p for p in (entity, project, run_id) if p]) if run_id else None

    _dbg(
        "wb_summary_update: target run:",
        target if target else "(none)",
        "run_present=",
        run is not None,
        "wandb_disabled=",
        wandb_disabled,
    )

    if run is None and target:
        try:
            api_run = wandb.Api().run(target)
            _dbg(
                "wb_summary_update: recovered run via Api.run",
                getattr(api_run, "id", None),
            )
        except Exception as exc:  # pragma: no cover - API availability varies in CI
            if not wandb_disabled:
                _dbg("wb_summary_update: wandb.Api lookup failed:", exc)
    elif run is None and not target:
        _dbg(
            "wb_summary_update: no wandb.run and no target for Api lookup; cannot fetch run"
        )

    if run is None and api_run is None and not wandb_disabled:
        run = _wait_for_run(wandb, timeout)

    run_like = run or api_run
    if run_like is None:
        if not wandb_disabled:
            _dbg("wb_summary_update: no active run; skipping. keys=", list(payload.keys()))
        else:
            _dbg(
                "wb_summary_update: wandb disabled and no run available; skipping. keys=",
                list(payload.keys()),
            )
        return

    try:
        _dbg(
            "wb_summary_update: using run",
            getattr(run_like, "id", None),
            getattr(run_like, "entity", None),
            getattr(run_like, "project", None),
        )
    except Exception as exc:
        _dbg("wb_summary_update: unable to introspect run info:", exc)

    try:
        val_rmse = _coerce_from_candidates(payload, RMSE_CANDIDATES, _coerce_float)
        val_auc = _coerce_from_candidates(payload, AUC_CANDIDATES, _coerce_float)
        best_step = _coerce_from_candidates(payload, ("best_step", "epoch", "step"), _coerce_int)

        log_payload = {}
        if val_rmse is not None:
            log_payload["val_rmse"] = val_rmse
        if val_auc is not None:
            log_payload["val_auc"] = val_auc
        if best_step is not None:
            log_payload["best_step"] = best_step
        if log_payload and run is not None:
            _dbg("logging metrics:", {k: log_payload[k] for k in sorted(log_payload)})
            try:
                wandb.log(log_payload)
                _dbg("wandb.log succeeded")
            except Exception as exc:
                _dbg("wandb.log failed:", exc)

        summary_payload = dict(payload)
        if val_rmse is not None:
            summary_payload["val_rmse"] = val_rmse
        if val_auc is not None:
            summary_payload["val_auc"] = val_auc
        if best_step is not None:
            summary_payload["best_step"] = best_step
        if "val_mae" not in summary_payload and payload.get("mae_mean") is not None:
            mae_val = _coerce_float(payload.get("mae_mean"))
            if mae_val is not None:
                summary_payload["val_mae"] = mae_val

        try:
            _dbg("wb_summary_update: calling summary.update with keys", list(summary_payload.keys()))
            run_like.summary.update(summary_payload)
            _dbg("wb_summary_update: summary.update succeeded")
        except Exception as exc:
            _dbg("wb_summary_update summary.update failed:", exc)
            for key, value in summary_payload.items():
                try:
                    run_like.summary[key] = value
                    _dbg(f"wb_summary_update: wrote summary[{key!r}] via fallback")
                except Exception as inner:
                    _dbg(f"failed to set summary[{key!r}]:", inner)
    except Exception as exc:
        _dbg("wb_summary_update exception:", exc)

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
        finish()
    except Exception:
        pass

    try:
        while getattr(wandb, "run", None) is not None and time.time() - start < timeout:
            time.sleep(0.5)
    except Exception:
        pass
