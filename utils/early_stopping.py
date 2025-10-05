"""Utility for early stopping based on validation loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopping:
    """Simple early stopping mechanism supporting min/max objectives."""

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"
    best: Optional[float] = None
    counter: int = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Update with the latest validation metric and report stop condition."""

        if self._is_improvement(value):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
