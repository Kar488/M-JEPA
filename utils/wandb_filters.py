"""Helpers to tame noisy warnings from optional W&B dependencies."""

from __future__ import annotations

import warnings


def silence_pydantic_field_warnings() -> None:
    """Mute noisy Pydantic field attribute warnings when possible.

    Recent releases of the W&B client depend on :mod:`pydantic` 2.x which emits
    ``UnsupportedFieldAttributeWarning`` whenever third-party schema builders
    pass unsupported keyword arguments (for example ``repr=False`` or
    ``frozen=True``).  These warnings are harmless for our use case but they
    flood stdout during CLI usage.  Import the warning class dynamically so the
    helper is safe even when Pydantic is not installed in lightweight
    environments (e.g. unit tests or offline analysis).
    """

    try:  # pragma: no cover - exercised indirectly when pydantic is installed
        from pydantic.warnings import UnsupportedFieldAttributeWarning  # type: ignore
    except Exception:  # noqa: BLE001 - best effort guard
        return

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


__all__ = ["silence_pydantic_field_warnings"]
