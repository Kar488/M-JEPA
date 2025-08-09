"""Seed setting utility for reproducibility.

This module provides a helper function to set Python, NumPy and PyTorch
random seeds. Setting a fixed seed allows experiments to be
deterministic when using the same environment.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
