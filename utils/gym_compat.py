"""Compatibility helpers for transitioning from Gym to Gymnasium.

The BuildAMol dependency (used for optional 3-D graph renderings) still
imports ``gym`` internally.  Gym is unmaintained and emits a loud warning when
imported under NumPy 2+, so we opportunistically alias ``gymnasium`` to
``gym`` when possible.  Downstream imports observe a ``gym`` module that is
backed by Gymnasium, which is API-compatible for our simple viewer usage.
"""
from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType

_logger = logging.getLogger(__name__)


def ensure_gymnasium_alias() -> bool:
    """Alias Gymnasium as ``gym`` if Gym is missing.

    Returns ``True`` when the alias was installed.  When Gym is already
    imported or Gymnasium cannot be resolved we leave ``sys.modules`` untouched
    and return ``False``.
    """

    if "gym" in sys.modules:
        return False

    try:
        module = importlib.import_module("gymnasium")
    except ModuleNotFoundError:
        return False
    except Exception as exc:  # pragma: no cover - defensive logging path
        _logger.debug("failed to import gymnasium: %s", exc)
        return False

    if not isinstance(module, ModuleType):  # pragma: no cover - defensive
        return False

    sys.modules.setdefault("gym", module)
    return True


__all__ = ["ensure_gymnasium_alias"]
