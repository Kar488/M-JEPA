"""Utilities for working with :class:`torch.utils.data.DataLoader` objects."""

from __future__ import annotations

from typing import Optional, Tuple


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
