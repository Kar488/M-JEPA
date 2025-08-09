"""Utility for early stopping based on validation loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopping:
    """Simple early stopping mechanism.

    Parameters
    ----------
    patience: int
        Number of epochs with no improvement after which training will be
        stopped.
    min_delta: float, optional
        Minimum change in the monitored quantity to qualify as an
        improvement.
    """

    patience: int = 5
    min_delta: float = 0.0
    best: Optional[float] = None
    counter: int = 0

    def step(self, value: float) -> bool:
        """Update with the latest validation loss.

        Returns True if training should stop."""
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
