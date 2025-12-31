"""Tiny helpers for saying what our experiments did.

This module provides a minimal wrapper around the optional :mod:`wandb`
package so that training scripts can log metrics even when the library
isn't installed.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Dict, Optional, Sequence

try:
    from .wandb_filters import silence_pydantic_field_warnings
except Exception:  # pragma: no cover - fallback for minimal wheels
    def silence_pydantic_field_warnings() -> None:  # type: ignore
        return


class DummyWandb:
    """A toy logger that does nothing.

    This stand-in object mimics the interface of :mod:`wandb` when the real
    package is unavailable.
    """

    def __init__(self):
        """Create a placeholder logger.

        The object stores a flag showing that logging is not active.
        """
        self._ok = False

    def log(self, *a, **k):
        """Pretend to record some numbers.

        All arguments are ignored, but kept for API compatibility.
        """
        pass

    def Image(self, *_a, **_k):
        """Stand in for :func:`wandb.Image`.

        Returns ``None`` so that calls like ``wb.Image(fig)`` succeed even when
        the real wandb package is absent.
        """
        return None
    
    def finish(self):
        """Say we're done logging.

        This method exists so calls to ``wandb.finish`` won't fail.
        """
        pass


def _normalise_resume_flag(value: Any) -> Any:
    """Coerce resume flags into an explicit mode.

    W&B defaults to resuming from ``latest-run`` when ``resume`` is unset and a
    prior run directory exists.  That behaviour caused cross-stage metric
    mixing, so we opt into a strict default of ``"never"`` unless the caller
    explicitly asks to resume.
    """

    if value is None:
        return "never"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "never", "off", "none"}:
            return "never"
        return value
    if value is False:
        return "never"
    return value


def _safe_label(token: Optional[str]) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.:-]+", "-", token or "")
    cleaned = cleaned.strip("-")
    return cleaned or "run"


def _derive_run_id(env: Dict[str, str], name: Optional[str], job_type: Optional[str]) -> str:
    prefix = job_type or name or env.get("MJEPACI_STAGE") or env.get("WANDB_NAME") or "run"
    return f"{_safe_label(prefix)}-{uuid.uuid4().hex[:8]}"




def maybe_init_wandb(
    enable: bool,
    project: str = "m-jepa",
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    *,
    entity: Optional[str] = None,         # optional
    group: Optional[str] = None,          # optional
    job_type: Optional[str] = None,       # optional
    settings: Optional["wandb.Settings"] = None,  # optional passthrough
    initialise_run: bool = True,
    **extra: Any,                         # future-proof for more kwargs
):
    """Start a tracker if we ask nicely.

    If ``enable`` is ``True`` and the :mod:`wandb` package can be imported, a
    real wandb run is started. Otherwise a :class:`DummyWandb` is returned so
    logging calls remain safe.

    Parameters
    ----------
    enable : bool
        Whether to attempt initialising wandb.
    project : str, optional
        Name of the wandb project. Defaults to ``"m-jepa"``.
    config : Optional[Dict[str, Any]], optional
        Configuration values to record. Defaults to ``None``.
    tags : Optional[Sequence[str]], optional
        Labels for the run. Defaults to ``None``.
    api_key : Optional[str], optional
        API key for logging in to wandb. If provided, ``wandb.login`` is called
        before initialising the run. Defaults to ``None``.

    Parameters
    ----------
    initialise_run : bool, optional
        When ``False``, import :mod:`wandb` and perform login handling without
        creating or updating a run. Defaults to ``True`` to maintain the
        original behaviour for training scripts.

    Returns
    -------
    Any
        The real :mod:`wandb` module if initialised, otherwise a dummy logger.
    """
    if not enable:
        return DummyWandb()
    try:
        silence_pydantic_field_warnings()
        import wandb

        # ---- env-driven config (lets CI control id/name/resume etc.) ----
        env = os.environ
        # Allow login from either explicit arg or env
        key = api_key or os.getenv("WANDB_API_KEY")
        if key:
            try:
                wandb.login(key=key)
            except Exception as e:
                logging.warning("Failed to login to wandb: %s", e)

        run = getattr(wandb, "run", None)
        if not initialise_run:
            return wandb
        resume_flag = _normalise_resume_flag(extra.pop("resume", env.get("WANDB_RESUME")))

        name_kw = extra.pop("name", None)
        group_kw = extra.pop("group", None)
        job_kw = extra.pop("job_type", None)
        id_kw = extra.pop("id", None)
        settings_kw = settings or extra.pop("settings", None)

        name_val = env.get("WANDB_NAME") or name_kw
        job_type_val = env.get("WANDB_JOB_TYPE") or job_kw or job_type
        group_val = env.get("WANDB_RUN_GROUP") or group_kw or group or env.get("EXP_ID") or env.get("RUN_ID")
        allow_resume = True
        if isinstance(resume_flag, str) and resume_flag.lower() in {"never", "false", "0", "off", "none"}:
            allow_resume = False
        if resume_flag is False:
            allow_resume = False

        existing_id = env.get("WANDB_RUN_ID")
        run_id = (existing_id if (existing_id and allow_resume) else None) or id_kw or _derive_run_id(env, name_val, job_type_val)

        # Keep the generated id visible to subprocesses that rely on env state
        os.environ["WANDB_RUN_ID"] = run_id

        # When a previous run is still attached to this process and the caller
        # did *not* ask to resume, finish it to avoid cross-stage reuse.
        reuse_existing = allow_resume
        if run is not None and not reuse_existing:
            try:
                run.finish()
            except Exception:
                pass
            run = None

        if run is None:
            entity_kw = entity if entity is not None else env.get("WANDB_ENTITY")
            settings_final = settings_kw or getattr(wandb, "Settings", lambda **_: None)()  # type: ignore[misc]
            kw = dict(
                id=run_id,
                resume=resume_flag,
                name=name_val,
                group=group_val,
                job_type=job_type_val,
                dir=env.get("WANDB_DIR"),
                mode=env.get("WANDB_MODE"),
                project=env.get("WANDB_PROJECT", project),
                entity=entity_kw,
                config=config or {},
                tags=list(tags) if tags else None,
                settings=settings_final,
                reinit=True,
            )
            kw.update(extra)
            kw = {k: v for k, v in kw.items() if v is not None}
            wandb.init(**kw)
        else:
            # update name/config on resume (optional)
            new_name = os.getenv("WANDB_NAME")
            if new_name and getattr(run, "name", None) != new_name:
                run.name = new_name
                try:
                    run.save()
                except Exception:
                    pass
            if config and getattr(wandb, "config", None) is not None:
                try:
                    wandb.config.update(config, allow_val_change=True)
                except Exception:
                    pass
        return wandb

    except Exception as exc:
        logging.warning("Failed to initialise wandb: %s", exc)
        return DummyWandb()
