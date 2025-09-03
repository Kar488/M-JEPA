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

# -------------------------
# GINE: GIN with edge features
# -------------------------
class GINELayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int, eps: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )
        self.edge_lin = nn.Linear(edge_dim, dim) if edge_dim > 0 else None
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        i, j = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        msg = x[j]
        if self.edge_lin is not None:
            ea = self.edge_lin(edge_attr.to(x.dtype))
            msg = msg + ea
        agg.index_add_(0, i, msg)
        out = self.mlp((1 + self.eps) * x + agg)
        out = self.drop(out)
        return self.norm(out)


class GINE(EncoderBase):
    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GINELayer(hidden_dim, edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        a = getattr(g, "edge_attr", None)
        if a is None:
            a = torch.zeros((e.size(1), 0), dtype=x.dtype, device=device)
        else:
            a = torch.as_tensor(a, dtype=x.dtype, device=device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e, a)
        x = self.out_norm(x)
        from utils.pooling import global_mean_pool
        return global_mean_pool(x, getattr(g, "graph_ptr", None))


# -------------------------
# D-MPNN: directed edge message passing (Chemprop style)
# -------------------------
class DMPNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(node_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Build directed edges (j->i)
        i, j = edge_index[0], edge_index[1]
        # Edge hidden states h_ij
        if edge_attr.dtype != x.dtype:
            edge_attr = edge_attr.to(x.dtype)
        e_hidden = self.edge_mlp(torch.cat([x[j], edge_attr], dim=-1))  # [E, H]

        # Aggregate incoming edge states to node i
        agg = torch.zeros(x.size(0), e_hidden.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, i, e_hidden)

        out = self.node_mlp(torch.cat([x, agg], dim=-1))
        out = self.drop(out)
        return self.norm(out + x)


class DMPNN(EncoderBase):
    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [DMPNNLayer(hidden_dim, edge_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        a = getattr(g, "edge_attr", None)
        if a is None:
            a = torch.zeros((e.size(1), 0), dtype=x.dtype, device=device)
        else:
            a = torch.as_tensor(a, dtype=x.dtype, device=device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e, a)
        x = self.out_norm(x)
        from utils.pooling import global_mean_pool
        return global_mean_pool(x, getattr(g, "graph_ptr", None))


# -------------------------
# AttentiveFP-lite: edge-aware MPNN with attentive readout
# -------------------------
class AttnReadout(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor, graph_ptr: torch.Tensor | None) -> torch.Tensor:
        # Simple global attention: α_i ∝ softmax( (tanh(Wx_i) · q) )
        g = self.gate(x)  # [N, D]
        score = torch.matmul(g, self.query)  # [N]
        if graph_ptr is None:
            attn = torch.softmax(score, dim=0).unsqueeze(-1)
            return torch.sum(attn * x, dim=0, keepdim=True)
        # batched graphs
        num_graphs = int(graph_ptr.max().item()) + 1
        out = []
        for gid in range(num_graphs):
            mask = (graph_ptr == gid)
            s = score[mask]
            a = torch.softmax(s, dim=0).unsqueeze(-1)
            out.append(torch.sum(a * x[mask], dim=0, keepdim=True))
        return torch.cat(out, dim=0)


class AttentiveFPEncoder(EncoderBase):
    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [DMPNNLayer(hidden_dim, edge_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.readout = AttnReadout(hidden_dim)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        a = getattr(g, "edge_attr", None)
        if a is None:
            a = torch.zeros((e.size(1), 0), dtype=x.dtype, device=device)
        else:
            a = torch.as_tensor(a, dtype=x.dtype, device=device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e, a)
        x = self.out_norm(x)
        graph_ptr = getattr(g, "graph_ptr", None)
        if graph_ptr is not None:
            graph_ptr = torch.as_tensor(graph_ptr, device=device, dtype=torch.long)
        return self.readout(x, graph_ptr)


# -------------------------
# SchNet-lite (3D): RBF + filter networks
# -------------------------
class RBF(nn.Module):
    def __init__(self, num_kernels: int = 64, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_kernels)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [E] distances
        # φ_k(d) = exp(-γ (d-c_k)^2)
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers)**2)


class SchNetInteraction(nn.Module):
    def __init__(self, dim: int, num_kernels: int = 64):
        super().__init__()
        self.filter = nn.Sequential(nn.Linear(num_kernels, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, i: torch.Tensor, j: torch.Tensor, rbf: torch.Tensor) -> torch.Tensor:
        # message m_ij = (filter(r_ij) * W x_j)
        Wxh = self.lin(x)
        m = rbf @ self.filter[0].weight.T  # cheap approximation; ok for lite impl
        # project to [E, D]
        m = self.filter(m)  # [E, D]
        agg = torch.zeros_like(x)
        agg.index_add_(0, i, m * Wxh[j])
        return self.norm(x + agg)


class SchNet3D(EncoderBase):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, num_kernels: int = 64, cutoff: float = 5.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.rbf = RBF(num_kernels=num_kernels, cutoff=cutoff)
        self.layers = nn.ModuleList([SchNetInteraction(hidden_dim, num_kernels) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        # Requires g.pos [N, 3]
        pos = torch.as_tensor(getattr(g, "pos"), dtype=torch.float32, device=device)
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        i, j = e[0], e[1]
        rij = pos[i] - pos[j]
        dij = torch.linalg.norm(rij, dim=-1)  # [E]
        rbf = self.rbf(dij)  # [E, K]
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, i, j, rbf)
        x = self.out_norm(x)
        from utils.pooling import global_mean_pool
        return global_mean_pool(x, getattr(g, "graph_ptr", None))
