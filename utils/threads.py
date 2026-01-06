from __future__ import annotations

"""Helpers for tuning OpenMP threading across training stages."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _coerce_positive_int(value: Optional[str], fallback: int = 1) -> int:
    try:
        parsed = int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def recommend_omp_threads(num_workers: int | None = None, world_size: int | None = None) -> int:
    """Return a per‑process OpenMP thread count that avoids CPU oversubscription.

    The heuristic splits available CPU cores evenly across distributed ranks and
    leaves headroom for DataLoader workers. Negative ``num_workers`` values are
    treated as ``0`` because the actual worker pool will be auto‑tuned later.
    """

    cpu_count = os.cpu_count() or 1
    workers = 0 if num_workers is None or num_workers < 0 else int(num_workers)
    distributed_world_size = world_size
    if distributed_world_size is None:
        distributed_world_size = _coerce_positive_int(os.environ.get("WORLD_SIZE"), 1)
    distributed_world_size = max(1, distributed_world_size)

    available = max(1, cpu_count - workers)
    return max(1, available // distributed_world_size)


def configure_omp_threads(
    *,
    stage: str,
    num_workers: int | None = None,
    world_size: int | None = None,
    log: Optional[logging.Logger] = None,
) -> int:
    """Set ``OMP_NUM_THREADS`` to a sensible default when the launcher chose ``1``.

    ``torchrun`` forces ``OMP_NUM_THREADS=1`` unless the caller overrides it,
    which is often too low for CPU‑heavy preprocessing. This helper respects a
    user‑supplied value when it differs from the default ``1`` but otherwise
    bumps the variable to a recommended per‑rank budget. Returns the active
    thread count for visibility in logs and checkpoints.
    """

    active_logger = log or logger
    current_raw = os.environ.get("OMP_NUM_THREADS")
    current = _coerce_positive_int(current_raw, fallback=1)
    recommended = recommend_omp_threads(num_workers=num_workers, world_size=world_size)

    # Respect explicit overrides that differ from the launcher default.
    if current_raw not in (None, "", "1") and current > 1:
        if active_logger and active_logger.isEnabledFor(logging.DEBUG):
            active_logger.debug(
                "Keeping existing OMP_NUM_THREADS=%s for %s stage", current_raw, stage
            )
        return current

    os.environ["OMP_NUM_THREADS"] = str(recommended)
    if active_logger:
        active_logger.info(
            "Setting OMP_NUM_THREADS=%d for %s (cpu=%s, workers=%s, world_size=%s)",
            recommended,
            stage,
            os.cpu_count() or "unknown",
            num_workers,
            world_size or _coerce_positive_int(os.environ.get("WORLD_SIZE"), 1),
        )
    return recommended
