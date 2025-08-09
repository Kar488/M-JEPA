"""Make experiments repeatable by fixing randomness.

This module offers a helper function that seeds Python's ``random`` module,
NumPy, and PyTorch so that runs produce identical results when executed in
the same environment. By controlling these sources of randomness, it enables
deterministic behavior for reproducible experiments.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Tell random number makers to start at the same place.

    Initializes the seed for Python's ``random`` module, NumPy, and PyTorch.
    When CUDA is available, the same seed is applied across all GPU devices,
    ensuring that model training and evaluation are deterministic when run in
    identical environments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
