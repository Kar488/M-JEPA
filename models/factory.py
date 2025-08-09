from __future__ import annotations

from typing import Optional

from models.edge_encoder import EdgeGNNEncoder
from models.encoder import GNNEncoder
from models.gnn_variants import GIN, GATMultiHead, GraphSAGE


def build_encoder(
    *,
    gnn_type: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    edge_dim: Optional[int] = None,
    heads: int = 4,
):
    gt = gnn_type.lower()
    if gt in ("edge_mpnn", "mpnn_edge", "edge"):
        if edge_dim is None:
            raise ValueError("edge_dim is required for edge_mpnn encoder.")
        return EdgeGNNEncoder(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    if gt in ("graphsage", "sage"):
        return GraphSAGE(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
        )
    if gt in ("gin",):
        return GIN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    if gt in ("gat_multi", "gatmh", "gatv2_multi"):
        return GATMultiHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        )
    # fallback to your base encoder (mpnn|gcn|gat single-head)
    return GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    )
