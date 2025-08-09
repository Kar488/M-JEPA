from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

import torch
import torch.nn as nn

from data.dataset import GraphData


class EncoderBase(nn.Module, ABC):
    """Abstract base encoder operating on :class:`GraphData`."""

    @abstractmethod
    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        """Encode a single graph and return its embedding."""

    def forward(self, batch: Union[GraphData, List[GraphData]]) -> torch.Tensor:  # type: ignore[override]
        device = next(self.parameters()).device
        if isinstance(batch, list):
            embs = [self.encode_graph(g, device) for g in batch]
            return torch.stack(embs, dim=0)
        return self.encode_graph(batch, device).unsqueeze(0)


class PredictorBase(nn.Module, ABC):
    """Abstract base predictor mapping embeddings to targets."""

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict target embeddings from input embeddings."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.predict(x)
