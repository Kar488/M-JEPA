"""MLP predictor for target embeddings.

The predictor maps context graph embeddings to predicted target
embeddings using a two‑layer multilayer perceptron. This module is
imported in the training routines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import PredictorBase


class MLPPredictor(PredictorBase):
    """Two‑layer MLP for predicting target graph embeddings.

    Args:
        embed_dim: Dimensionality of the input and output embeddings.
        hidden_dim: Number of hidden units in the intermediate layer.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
