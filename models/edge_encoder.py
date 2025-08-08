from __future__ import annotations
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import GraphData
from utils.pooling import global_mean_pool


class EdgeMPNNLayer(nn.Module):
    """A small MPNN that uses edge attributes: m_ij = φ([h_i, h_j, e_ij]);  h_i' = ψ([h_i, Σ_j m_ij])."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_index: [2, E], directed (j -> i)
        i_idx, j_idx = edge_index[0], edge_index[1]
        # messages
        m_in = torch.cat([x[i_idx], x[j_idx], edge_attr], dim=-1)  # [E, 2*D + F]
        m_ij = self.msg(m_in)                                      # [E, H]

        # aggregate by target index i
        agg = torch.zeros(x.size(0), m_ij.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, i_idx, m_ij)

        # update + residual + norm
        out = self.upd(torch.cat([x, agg], dim=-1))
        out = self.drop(out)
        return self.norm(out + x)


class EdgeGNNEncoder(nn.Module):
    """Stacks EdgeMPNN layers and returns a graph embedding via global mean pool."""

    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [EdgeMPNNLayer(hidden_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, dropout=dropout)
             for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    @torch.no_grad()
    def _to_tensors(self, g: GraphData, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)                # [N, Din]
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)         # [2, E]
        if g.edge_attr is None:
            raise ValueError("edge_attr is required for EdgeGNNEncoder (set add_3d_features=True in dataset to get bond lengths).")
        a = torch.as_tensor(g.edge_attr, dtype=torch.float32, device=device)       # [E, Fe]
        return x, e, a

    def _encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x, e, a = self._to_tensors(g, device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e, a)
        x = self.out_norm(x)
        return global_mean_pool(x)  # [hidden]

    def forward(self, batch: Union[GraphData, List[GraphData]]) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(batch, list):
            embs = [self._encode_graph(g, device) for g in batch]
            return torch.stack(embs, dim=0)
        return self._encode_graph(batch, device).unsqueeze(0)
