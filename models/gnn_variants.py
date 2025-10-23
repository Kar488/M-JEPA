from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.mdataset import GraphData
from models.base import EncoderBase
from utils.indexing import gather_nodes
from utils.pooling import global_mean_pool
from utils.scatter import scatter_sum


def _edge_attr_or_default(
    edge_attr: Optional[torch.Tensor],
    num_edges: int,
    expected_dim: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return ``edge_attr`` as a tensor with a well-defined feature width.

    Some MoleculeNet datasets omit bond features even when models expect them.
    Previously we produced a zero-width matrix which later triggered shape
    mismatches inside ``nn.Linear`` layers.  Instead, materialise the requested
    number of features so downstream layers always receive the configured
    ``edge_dim`` (typically padded with zeros when data is missing).
    """

    if expected_dim is not None and expected_dim < 0:
        raise ValueError("expected_dim must be non-negative")

    if edge_attr is None:
        width = expected_dim if expected_dim is not None else 0
        return torch.zeros((num_edges, width), dtype=dtype, device=device)

    edge_attr = torch.as_tensor(edge_attr, dtype=dtype, device=device)
    if expected_dim is not None and edge_attr.size(-1) != expected_dim:
        raise ValueError(
            "edge_attr has unexpected feature dimension: "
            f"expected {expected_dim}, got {edge_attr.size(-1)}"
        )
    return edge_attr


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
        msg = gather_nodes(x, j)
        if self.agg == "mean":
            agg = scatter_sum(i, msg, dim_size=x.size(0))
            deg = scatter_sum(
                i,
                i.new_ones(i.size(), dtype=x.dtype),
                dim_size=x.size(0),
            )
            deg = deg.clamp(min=1.0).unsqueeze(-1)
            agg = agg / deg
        elif self.agg == "max":
            agg = torch.zeros_like(x)
            agg.fill_(-1e9)
            agg = torch.max(
                agg.index_copy(0, i, msg), dim=0
            ).values  # rough max; simple fallback
        else:
            # LSTM aggregator omitted; can be added later
            agg = scatter_sum(i, msg, dim_size=x.size(0))
            deg = scatter_sum(
                i,
                i.new_ones(i.size(), dtype=x.dtype),
                dim_size=x.size(0),
            )
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
        agg = scatter_sum(i, gather_nodes(x, j), dim_size=x.size(0))
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
        xh_i = gather_nodes(xh, i)
        xh_j = gather_nodes(xh, j)
        alpha = (xh_i * self.attn_dst).sum(-1) + (xh_j * self.attn_src).sum(
            -1
        )  # [E, H]
        alpha = self.leaky(alpha)
        # softmax by i (target)
        alpha_exp = torch.exp(alpha)
        try:  # pragma: no cover - optional dependency tested via integration tests
            from torch_scatter import segment_softmax  # type: ignore[import]
        except Exception:  # pragma: no cover - gracefully handle missing torch_scatter
            segment_softmax = None

        if segment_softmax is not None:
            denom = segment_softmax(alpha_exp, i) * 0 + alpha_exp
        else:
            # Manual softmax per target node
            denom = scatter_sum(i, alpha_exp, dim_size=N)
            denom = gather_nodes(denom, i)
        attn = alpha_exp / (denom + 1e-9)

        # aggregate
        out = scatter_sum(i, attn.unsqueeze(-1) * xh_j, dim_size=N)
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
        msg = gather_nodes(x, j)
        if self.edge_lin is not None:
            ea = self.edge_lin(edge_attr.to(x.dtype))
            msg = msg + ea
        agg = scatter_sum(i, msg, dim_size=x.size(0))
        out = self.mlp((1 + self.eps) * x + agg)
        out = self.drop(out)
        return self.norm(out)


class GINE(EncoderBase):
    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GINELayer(hidden_dim, edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _encode_nodes(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        a = _edge_attr_or_default(getattr(g, "edge_attr", None), e.size(1), self.edge_dim, x.dtype, device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e, a)
        return self.out_norm(x)  # [N, D]

    # forward must NOT take 'device'; infer from parameters
    def forward(self, g: GraphData) -> torch.Tensor:
        device = next(self.parameters()).device
        return self._encode_nodes(g, device)  # [N, D]

    # test convenience: pool to [D]
    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = self._encode_nodes(g, device)  # [N, D]
        from utils.pooling import global_mean_pool
        graph_ptr = getattr(g, "graph_ptr", None)
        return global_mean_pool(x, graph_ptr)  # [D]

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
        src_x = gather_nodes(x, j)
        e_hidden = self.edge_mlp(torch.cat([src_x, edge_attr], dim=-1))  # [E, H]

        # Aggregate incoming edge states to node i
        # Use e_hidden.dtype to match mixed‑precision (bf16/fp32) when autocast is enabled.
        agg = scatter_sum(i, e_hidden, dim_size=x.size(0))

        out = self.node_mlp(torch.cat([x, agg], dim=-1))
        out = self.drop(out)
        return self.norm(out + x)


class DMPNN(EncoderBase):
    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [DMPNNLayer(hidden_dim, edge_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _encode_nodes(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        a = _edge_attr_or_default(getattr(g, "edge_attr", None), e.size(1), self.edge_dim, x.dtype, device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e, a)
        return self.out_norm(x)  # [N, D]

    def forward(self, g: GraphData) -> torch.Tensor:
        device = next(self.parameters()).device
        return self._encode_nodes(g, device)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = self._encode_nodes(g, device)
        from utils.pooling import global_mean_pool
        graph_ptr = getattr(g, "graph_ptr", None)
        return global_mean_pool(x, graph_ptr)


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
            return torch.sum(attn * x, dim=0)
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
        self.edge_dim = int(edge_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [DMPNNLayer(hidden_dim, edge_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.readout = AttnReadout(hidden_dim)

    def _encode_nodes(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        a = _edge_attr_or_default(getattr(g, "edge_attr", None), e.size(1), self.edge_dim, x.dtype, device)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, e, a)
        return self.out_norm(x)  # [N, D]

    def forward(self, g: GraphData) -> torch.Tensor:
        device = next(self.parameters()).device
        return self._encode_nodes(g, device)  # [N, D]

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = self._encode_nodes(g, device)  # [N, D]
        graph_ptr = getattr(g, "graph_ptr", None)
        if graph_ptr is not None:
            graph_ptr = torch.as_tensor(graph_ptr, device=device, dtype=torch.long)
        z = self.readout(x, graph_ptr)     # [D] or [1, D]
        if z.dim() == 2 and z.size(0) == 1:
            z = z.squeeze(0)
        return z

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
        """
        x   : [N, D]     node embeddings
        i,j : [E]        edge indices (messages are j -> i)
        rbf : [E, K]     radial basis features for each edge distance
        """
        # Linear projection of source nodes
        Wxh = self.lin(x)                  # [N, D]
        if Wxh.dtype != x.dtype:
            Wxh = Wxh.to(x.dtype)

        # Produce edge-wise filters from RBFs via the filter network
        f_ij = self.filter(rbf)            # [E, D]
        if f_ij.dtype != x.dtype:
            f_ij = f_ij.to(x.dtype)

        # Edge messages: modulate source embedding with edge filter
        Wxh_j = gather_nodes(Wxh, j)
        msg = f_ij * Wxh_j                # [E, D]
        if msg.dtype != x.dtype:
            msg = msg.to(x.dtype)

        # Aggregate into destination nodes
        agg = scatter_sum(i, msg, dim_size=x.size(0))

        # Residual + normalization
        return self.norm(x + agg)


class SchNet3D(EncoderBase):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, num_kernels: int = 64, cutoff: float = 5.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.rbf = RBF(num_kernels=num_kernels, cutoff=cutoff)
        self.layers = nn.ModuleList([SchNetInteraction(hidden_dim, num_kernels) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self._missing_pos_warned = False

    def _encode_nodes(self, g: GraphData, device: torch.device) -> torch.Tensor:
        pos = getattr(g, "pos", None)
        if pos is None and hasattr(g, "graphs"):
            # ``GraphBatch`` stores the individual graphs when collating.
            # Attempt to rebuild the concatenated coordinate tensor on-the-fly
            # so SchNet3D can operate on DataLoader batches even when the
            # collate function skipped populating ``batch.pos``. When some
            # graphs are missing coordinates, fall back to zeros for those
            # entries to avoid hard failures during sweeps.
            pos_list = []
            pos_width = 0
            used_fallback = False
            for graph in getattr(g, "graphs", []):
                coords = getattr(graph, "pos", None)
                if coords is None:
                    used_fallback = True
                    try:
                        num_nodes = int(graph.num_nodes())
                    except Exception:
                        x_field = getattr(graph, "x", None)
                        if x_field is None:
                            num_nodes = 0
                        else:
                            shape = getattr(x_field, "shape", None)
                            if shape is not None and len(shape) > 0:
                                num_nodes = int(shape[0])
                            else:
                                try:
                                    num_nodes = int(len(x_field))
                                except Exception:
                                    num_nodes = 0
                    width = pos_width or 3
                    coords_t = torch.zeros((num_nodes, width), dtype=torch.float32, device=device)
                else:
                    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=device)
                    if coords_t.ndim != 2:
                        coords_t = coords_t.view(coords_t.size(0), -1)
                    width = int(coords_t.size(-1))

                if width > pos_width:
                    # Expand previously collected tensors to the new width.
                    diff = width - pos_width
                    for idx, existing in enumerate(pos_list):
                        pad = torch.zeros((existing.size(0), diff), dtype=existing.dtype, device=existing.device)
                        pos_list[idx] = torch.cat([existing, pad], dim=-1)
                    pos_width = width

                if width < pos_width:
                    pad = torch.zeros((coords_t.size(0), pos_width - width), dtype=coords_t.dtype, device=device)
                    coords_t = torch.cat([coords_t, pad], dim=-1)


                pos_list.append(coords_t)

            if pos_list:
                pos = torch.cat(pos_list, dim=0)
            else:
                width = pos_width or 3
                pos = torch.zeros((0, width), dtype=torch.float32, device=device)

            if used_fallback and not self._missing_pos_warned:
                logging.getLogger(__name__).warning(
                    "SchNet3D received graphs without `pos`; filled zeros as fallback."
                )
                self._missing_pos_warned = True

            try:
                setattr(g, "pos", pos)
            except Exception:
                # ``g`` may be a lightweight shim; cache best-effort only.
                pass
        
        if pos is None:
            raise ValueError(
                "SchNet3D requires 3D coordinates `pos`. Enable --add-3d and ensure loader populates g.pos."
            )
        pos = torch.as_tensor(pos, dtype=torch.float32, device=device)
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        e = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        i, j = e[0], e[1]
        pos_i = gather_nodes(pos, i)
        pos_j = gather_nodes(pos, j)
        rij = pos_i - pos_j
        dij = torch.linalg.norm(rij, dim=-1)        # [E]
        rbf = self.rbf(dij)                         # [E, K]
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, i, j, rbf)
        return self.out_norm(x)  # [N, D]

    def forward(self, g: GraphData) -> torch.Tensor:
        device = next(self.parameters()).device
        return self._encode_nodes(g, device)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = self._encode_nodes(g, device)
        from utils.pooling import global_mean_pool
        graph_ptr = getattr(g, "graph_ptr", None)
        return global_mean_pool(x, graph_ptr)



