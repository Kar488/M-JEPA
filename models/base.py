from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, List, Union

import torch
import torch.nn as nn

from data.mdataset import GraphData


def _coerce_graph_like(obj: Any) -> GraphData | SimpleNamespace:
    """Return a graph-like object exposing ``x``/``edge_index`` attributes."""

    if hasattr(obj, "x") and hasattr(obj, "edge_index"):
        return obj  # already graph-like

    if isinstance(obj, Mapping):
        payload = dict(obj)
        return SimpleNamespace(
            x=payload.get("x"),
            edge_index=payload.get("edge_index"),
            edge_attr=payload.get("edge_attr"),
            pos=payload.get("pos"),
            graph_ptr=payload.get("graph_ptr"),
            batch=payload.get("batch"),
            ptr=payload.get("ptr"),
        )

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for candidate in obj:
            try:
                return _coerce_graph_like(candidate)
            except AttributeError:
                continue

    raise AttributeError(
        "Graph-like input must provide 'x' and 'edge_index' attributes or mappings"
    )


class EncoderBase(nn.Module, ABC):
    """Abstract base encoder operating on :class:`GraphData`."""

    @abstractmethod
    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        """Encode a single graph and return its embedding."""

    def forward(self, batch: Union[GraphData, List[GraphData], Any]) -> torch.Tensor:  # type: ignore[override]
        device = next(self.parameters()).device
        if isinstance(batch, list):
            embs = [self.encode_graph(_coerce_graph_like(g), device) for g in batch]
            return torch.stack(embs, dim=0)
        coerced = _coerce_graph_like(batch)
        return self.encode_graph(coerced, device).unsqueeze(0)


class PredictorBase(nn.Module, ABC):
    """Abstract base predictor mapping embeddings to targets."""

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict target embeddings from input embeddings."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.predict(x)
