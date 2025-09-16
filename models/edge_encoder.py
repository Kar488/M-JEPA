from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.mdataset import GraphData
from models.base import EncoderBase
from utils.graph_ops import _pool_graph_emb, _to_tensor
from utils.scatter import scatter_sum


class EdgeMPNNLayer(nn.Module):
    """A small MPNN that uses edge attributes: m_ij = φ([h_i, h_j, e_ij]);  h_i' = ψ([h_i, Σ_j m_ij])."""

    def __init__(
        self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.upd = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(node_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # edge_index: [2, E], directed (j -> i)
        i_idx, j_idx = edge_index[0], edge_index[1]

        # Ensure consistent dtype under autocast (msg() may yield fp16/bf16)
        if edge_attr.dtype != x.dtype:
            edge_attr = edge_attr.to(x.dtype)

        # messages
        m_in = torch.cat([x[i_idx], x[j_idx], edge_attr], dim=-1)  # [E, 2*D + F]
        m_ij = self.msg(m_in)  # [E, H]
        if m_ij.dtype != x.dtype:
            m_ij = m_ij.to(x.dtype)

        # aggregate by target index i
        agg = scatter_sum(i_idx, m_ij, dim_size=x.size(0))

        # update + residual + norm
        out = self.upd(torch.cat([x, agg], dim=-1))
        out = self.drop(out)
        return self.norm(out + x)


class EdgeGNNEncoder(EncoderBase):
    """Stacks EdgeMPNN layers and returns a graph embedding via global mean pool."""

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(hidden_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EdgeMPNNLayer(
                    hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)


    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = _to_tensor(g.x, device=device)      
        e = _to_tensor(g.edge_index, dtype=torch.long, device=device)  # [2, E]
        edge_attr = getattr(g, "edge_attr", None)
        if edge_attr is None:
            # No edge features → use a zero-width matrix [E, 0]
            a = torch.zeros((e.size(1), 0), dtype=x.dtype, device=device)
        else:
            a = _to_tensor(edge_attr, dtype=x.dtype, device=device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e, a)
        x = self.out_norm(x)
        return _pool_graph_emb(x, g)