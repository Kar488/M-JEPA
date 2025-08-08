from __future__ import annotations
from typing import Optional
from models.encoder import GNNEncoder
from models.edge_encoder import EdgeGNNEncoder

def build_encoder(*, gnn_type: str, input_dim: int, hidden_dim: int, num_layers: int,
                  edge_dim: Optional[int] = None) -> object:
    """Return the right encoder by name. New type: 'edge_mpnn'."""
    gt = gnn_type.lower()
    if gt in ("edge_mpnn", "mpnn_edge", "edge"):
        if edge_dim is None:
            raise ValueError("edge_dim is required for edge_mpnn; dataset GraphData.edge_attr must be present.")
        return EdgeGNNEncoder(input_dim=input_dim, edge_dim=edge_dim,
                              hidden_dim=hidden_dim, num_layers=num_layers)
    # fall back to your existing encoder
    return GNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type)
