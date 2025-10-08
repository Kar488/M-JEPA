"""Tiny helpers for saying what our experiments did.

This module provides a minimal wrapper around the optional :mod:`wandb`
package so that training scripts can log metrics even when the library
isn't installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence
import os

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




def maybe_init_wandb(
    enable: bool,
    project: str = "m-jepa",
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    *,
    group: Optional[str] = None,          # optional
    job_type: Optional[str] = None,       # optional
    settings: Optional["wandb.Settings"] = None,  # optional passthrough
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
        if run is None:
            kw = dict(
                id   = env.get("WANDB_RUN_ID"),         # same id reused across stages
                resume   = env.get("WANDB_RESUME", "allow"),# allow/auto/never/… (allow is safe)
                name     = env.get("WANDB_NAME"),           # "grid", "pretrain", "finetune", …
                group    = env.get("WANDB_RUN_GROUP",group),       # optional
                job_type = env.get("WANDB_JOB_TYPE",job_type),        # optional
                dir  = env.get("WANDB_DIR"),             # e.g., /data/mjepa/wandb
                mode     = env.get("WANDB_MODE"),            # online/offline/disabled
                project  = env.get("WANDB_PROJECT", project),
                entity   = env.get("WANDB_ENTITY"),
                config=config or {},
                tags=list(tags) if tags else None,
                reinit=True,
            )
            kw = {k: v for k, v in kw.items() if v is not None}
            wandb.init(**kw)
        else:
            # update name/config on resume (optional)
            new_name = os.getenv("WANDB_NAME")
            if new_name and getattr(run, "name", None) != new_name:
                run.name = new_name
                run.save()
            if config and getattr(wandb, "config", None) is not None:
                try: wandb.config.update(config, allow_val_change=True)
                except Exception: pass
        return wandb

    except Exception as exc:
        logging.warning("Failed to initialise wandb: %s", exc)
        return DummyWandb()
