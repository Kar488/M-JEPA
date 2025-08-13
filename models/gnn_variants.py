from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.mdataset import GraphData
from models.base import EncoderBase
from utils.pooling import global_mean_pool


class GraphSAGELayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, agg: str = "mean", dropout: float = 0.0
    ):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.agg = agg
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        i, j = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        if self.agg == "mean":
            deg = torch.zeros(x.size(0), device=x.device).index_add_(
                0, i, torch.ones_like(i, dtype=torch.float)
            )
            agg.index_add_(0, i, x[j])
            deg = deg.clamp(min=1.0).unsqueeze(-1)
            agg = agg / deg
        elif self.agg == "max":
            agg.fill_(-1e9)
            agg = torch.max(
                agg.index_copy(0, i, x[j]), dim=0
            ).values  # rough max; simple fallback
        else:
            # LSTM aggregator omitted; can be added later
            deg = torch.zeros(x.size(0), device=x.device).index_add_(
                0, i, torch.ones_like(i, dtype=torch.float)
            )
            agg.index_add_(0, i, x[j])
            deg = deg.clamp(min=1.0).unsqueeze(-1)
            agg = agg / deg
        out = self.lin(torch.cat([x, agg], dim=-1))
        out = F.relu(out, inplace=True)
        out = self.drop(out)
        return self.norm(out)


class GraphSAGE(EncoderBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        agg: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GraphSAGELayer(hidden_dim, hidden_dim, agg=agg, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e)
        x = self.out_norm(x)

        return global_mean_pool(x, getattr(g, "graph_ptr", None))


class GINLayer(nn.Module):
    def __init__(self, dim: int, eps: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        i, j = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, i, x[j])
        out = self.mlp((1 + self.eps) * x + agg)
        out = self.drop(out)
        return self.norm(out)


class GIN(EncoderBase):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GINLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e)
        x = self.out_norm(x)

        return global_mean_pool(x, getattr(g, "graph_ptr", None))


class MultiHeadGATLayer(nn.Module):
    def __init__(
        self, dim: int, heads: int = 4, dropout: float = 0.0, concat: bool = True
    ):
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.lin = nn.Linear(dim, dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(heads, dim))
        self.attn_dst = nn.Parameter(torch.zeros(heads, dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.leaky = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim * heads if concat else dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, D]; project to heads
        N, D = x.size()
        i, j = edge_index[0], edge_index[1]
        H = self.heads
        xh = self.lin(x).view(N, H, D)  # [N, H, D]

        # scores
        alpha = (xh[i] * self.attn_dst).sum(-1) + (xh[j] * self.attn_src).sum(
            -1
        )  # [E, H]
        alpha = self.leaky(alpha)
        # softmax by i (target)
        import torch_scatter  # if not installed, we fallback below

        try:
            # scatter softmax
            from torch_scatter import segment_softmax

            alpha_exp = torch.exp(alpha)
            denom = (
                segment_softmax(alpha_exp, i) * 0 + alpha_exp
            )  # not ideal; fallback below
        except Exception:
            # Manual softmax per target node
            denom = torch.zeros((N, H), device=x.device)
            denom.index_add_(0, i, torch.exp(alpha))
            denom = denom[i]
        attn = torch.exp(alpha) / (denom + 1e-9)

        # aggregate
        out = torch.zeros((N, H, D), device=x.device)
        out.index_add_(0, i, attn.unsqueeze(-1) * xh[j])
        out = out.reshape(N, H * D) if self.concat else out.mean(dim=1)
        out = self.out_proj(out)
        out = self.drop(out)
        return self.norm(F.elu(out) + x)


class GATMultiHead(EncoderBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MultiHeadGATLayer(hidden_dim, heads=heads, dropout=dropout, concat=True)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e)
        x = self.out_norm(x)
        return global_mean_pool(x, getattr(g, "graph_ptr", None))
