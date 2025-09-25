"""Utilities for working with :class:`torch.utils.data.DataLoader` objects."""

from __future__ import annotations

from typing import Optional, Tuple

import torch.utils.data.dataloader as _torch_dataloader


def _patch_iterator_finalizer() -> None:
    """Avoid ``AttributeError`` when worker start-up fails mid-construction."""

    iterator_cls = getattr(_torch_dataloader, "_MultiProcessingDataLoaderIter", None)
    if iterator_cls is None:  # pragma: no cover - backend dependent
        return

    if getattr(iterator_cls, "__mjepa_patched__", False):
        return

    original_del = getattr(iterator_cls, "__del__", None)
    if not callable(original_del):  # pragma: no cover - backend dependent
        return

    def _safe_del(self) -> None:
        # ``_workers_status`` is initialised near the end of the constructor.
        # When spawning worker processes fails (e.g. after running out of file
        # descriptors) PyTorch may invoke ``__del__`` on a partially constructed
        # iterator that never defined ``_workers_status``.  The upstream
        # finaliser assumes the attribute exists and crashes with ``AttributeError``
        # which is emitted as ``Exception ignored in ...`` noise.  Skipping the
        # original finaliser in this situation is safe because worker processes
        # were never launched successfully.
        if not hasattr(self, "_workers_status"):
            return
        try:
            original_del(self)
        except AttributeError as exc:  # pragma: no cover - defensive guard
            if "_workers_status" in str(exc):
                return
            raise

    iterator_cls.__del__ = _safe_del  # type: ignore[assignment]
    setattr(iterator_cls, "__mjepa_patched__", True)


_patch_iterator_finalizer()


def normalize_prefetch_factor(prefetch_factor: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    """Return a DataLoader-compatible ``prefetch_factor`` and the original value.

    PyTorch requires ``prefetch_factor`` to be a positive integer whenever
    ``num_workers > 0``.  Older configuration files may use ``0`` or negative
    numbers to indicate that prefetching should be disabled entirely, which
    triggers an assertion in :class:`~torch.utils.data.DataLoader` when worker
    processes are enabled.  This helper coerces such values to ``1`` – the
    smallest legal value – and returns the offending input so callers may log a
    helpful warning.  Passing ``None`` leaves the value untouched, allowing
    callers to skip setting ``prefetch_factor`` altogether.

    Args:
        prefetch_factor: Desired number of batches to prefetch per worker or
            ``None`` to defer to PyTorch's default behaviour.

    Returns:
        A tuple ``(normalised, original_bad_value)`` where ``normalised`` is the
        value that should be forwarded to ``DataLoader`` (possibly ``None``) and
        ``original_bad_value`` is the user-specified value when coercion was
        required.  When no adjustment is needed ``original_bad_value`` is
        ``None``.
    """

    if prefetch_factor is None:
        return None, None

    if prefetch_factor <= 0:
        return 1, prefetch_factor

    return prefetch_factor, None
