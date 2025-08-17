"""Tiny helpers for saying what our experiments did.

This module provides a minimal wrapper around the optional :mod:`wandb`
package so that training scripts can log metrics even when the library
isn't installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence


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
        import wandb

        if api_key:
            # If login fails, we still fall through to init and handle errors uniformly
            try:
                wandb.login(key=api_key)
            except Exception as e:
                logging.warning("Failed to login to wandb: %s", e)

        run = getattr(wandb, "run", None)  # <- handle fakes without a .run attribute
        
        if run is None:
            # Single consolidated run: init once
            wandb.init(
                project=project,
                config=config or {},
                tags=list(tags) if tags else None,
                reinit=False,
                resume="never",
            )
        else:
            # Optional: merge later config safely if .config exists
            if config and getattr(wandb, "config", None) is not None:
                try:
                    wandb.config.update(config, allow_val_change=True)
                except Exception:
                    # Be tolerant; config merging is best-effort
                    pass
        
        return wandb
    except Exception as exc:
        logging.warning("Failed to initialise wandb: %s", exc)
        return DummyWandb()
