"""Utilities for working with :class:`torch.utils.data.DataLoader` objects."""

from __future__ import annotations

from typing import Optional, Tuple

import logging
import math
import os

import torch
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


def ensure_file_system_sharing_strategy() -> None:
    """Prefer filesystem-backed tensor sharing to avoid FD exhaustion."""

    mp = getattr(torch, "multiprocessing", None)
    if mp is None:  # pragma: no cover - backend dependent
        return

    get_strategy = getattr(mp, "get_sharing_strategy", None)
    set_strategy = getattr(mp, "set_sharing_strategy", None)
    if not callable(get_strategy) or not callable(set_strategy):  # pragma: no cover
        return

    try:
        if get_strategy() != "file_system":
            set_strategy("file_system")
    except RuntimeError:  # pragma: no cover - backend dependent
        pass


def _auto_worker_budget(max_auto_workers: Optional[int] = None) -> int:
    cpu_budget = max(1, (os.cpu_count() or 2) - 1)
    if max_auto_workers is not None:
        return max(1, min(cpu_budget, max_auto_workers))
    return max(1, min(cpu_budget, 8))


def autotune_worker_pool(
    *,
    requested_workers: Optional[int],
    dataset_size: int,
    batch_size: int,
    device_type: str = "cuda",
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    stage: str = "loader",
    max_auto_workers: Optional[int] = None,
) -> Tuple[int, bool, Optional[int]]:
    """Return tuned ``(num_workers, persistent_workers, prefetch_factor)``."""

    if dataset_size <= 0 or batch_size <= 0:
        return 0, False, None

    approx_batches = max(1, math.ceil(dataset_size / max(1, batch_size)))
    auto_requested = requested_workers is None or requested_workers < 0
    tuned_workers = int(requested_workers or 0)

    if auto_requested:
        if device_type == "cpu":
            tuned_workers = 0
        else:
            tuned_workers = min(approx_batches, _auto_worker_budget(max_auto_workers))
    else:
        tuned_workers = max(0, min(int(requested_workers or 0), approx_batches))

    if tuned_workers > 0:
        ensure_file_system_sharing_strategy()

    stage_prefix = f"[{stage}] " if stage else ""
    if logger is not None:
        if auto_requested and tuned_workers > 0:
            logger.info(
                "%sAuto-selected %d DataLoader workers for %d graphs (batch_size=%d)",
                stage_prefix,
                tuned_workers,
                dataset_size,
                batch_size,
            )
        elif not auto_requested and tuned_workers != int(requested_workers or 0):
            logger.info(
                "%sClamped DataLoader workers from %s to %d to match %d batches",
                stage_prefix,
                requested_workers,
                tuned_workers,
                approx_batches,
            )

    tuned_persistent = bool(persistent_workers and tuned_workers > 0)

    normalized_prefetch, bad_prefetch = normalize_prefetch_factor(prefetch_factor)
    if bad_prefetch is not None and logger is not None:
        logger.warning(
            "%sInvalid prefetch_factor=%s; using %s instead",
            stage_prefix,
            bad_prefetch,
            normalized_prefetch,
        )

    tuned_prefetch: Optional[int]
    if tuned_workers <= 0:
        tuned_prefetch = None
        tuned_persistent = False
    else:
        # Limit prefetch to the number of unique batches each worker can service
        max_batches_per_worker = max(1, math.ceil(approx_batches / tuned_workers))
        max_prefetch = max(2, min(8, max_batches_per_worker))
        if normalized_prefetch is None:
            tuned_prefetch = min(4, max_prefetch)
        else:
            tuned_prefetch = max(1, min(normalized_prefetch, max_prefetch))
        if tuned_prefetch == 1 and approx_batches > tuned_workers:
            tuned_prefetch = min(2, max_prefetch)

    return tuned_workers, tuned_persistent, tuned_prefetch
