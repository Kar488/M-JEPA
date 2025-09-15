"""GNN encoder for molecular graphs.

This module defines a simple message‑passing Graph Neural Network (GNN)
encoder that processes molecular graphs represented by block‑diagonal
adjacency matrices. The encoder propagates information across the graph
for a fixed number of layers and returns node embeddings. Graph
embeddings can be obtained by pooling node embeddings using functions
from the `utils` package.
"""

from __future__ import annotations

"""GNN encoder definitions with optional torch dependency.

This module is imported by some tests simply to ensure that the file exists.
Those tests do not require a real PyTorch installation, so we guard the import
of ``torch`` and provide minimal stubs when it is unavailable.  The actual
encoder implementation still requires real PyTorch; attempting to instantiate
the classes without ``torch`` present will raise ``ModuleNotFoundError``.
"""

try:  # pragma: no cover - exercised only when torch is missing
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # noqa: BLE001 - broad to catch import errors
    class _MissingTorch:
        """Stub module that raises if any attribute is accessed."""

        def __getattr__(self, name: str) -> None:
            raise ModuleNotFoundError("torch is required to use GNNEncoder")

    torch = _MissingTorch()  # type: ignore[assignment]

    class _MissingNN:  # minimal namespace with Module for subclassing
        class Module:  # noqa: D401 - simple stub
            """Placeholder base class when torch is unavailable."""

        class Linear:  # pragma: no cover - used only without torch
            def __init__(self, *a, **k):  # noqa: D401
                raise ModuleNotFoundError("torch is required to use GNNEncoder")

    nn = _MissingNN()  # type: ignore[assignment]

    class _MissingF:
        def __getattr__(self, name: str) -> None:  # pragma: no cover
            raise ModuleNotFoundError("torch is required to use GNNEncoder")

    F = _MissingF()  # type: ignore[assignment]


class GNNEncoder(nn.Module):
    """A simple Graph Neural Network with configurable update rule.

    The encoder supports three types of message passing:
        - "mpnn": A custom message‑passing network where messages from
          neighbours are linearly transformed then added to the current node
          embedding.
        - "gcn": A Graph Convolution Network (GCN) that uses symmetric
          normalisation of the adjacency matrix (D^(-1/2) A D^(-1/2)).
        - "gat": A single‑head Graph Attention Network (GAT) that learns
          attention coefficients between neighbouring nodes and performs
          weighted aggregation.

    Args:
        input_dim: Dimension of input node features.
        hidden_dim: Size of the hidden and output embeddings.
        num_layers: Number of message passing layers.
        gnn_type: Type of GNN ("mpnn" or "gcn").
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        gnn_type: str = "mpnn",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        if gnn_type == "mpnn":
            # Separate message and update transformations for MPNN
            self.message_lin = nn.Linear(hidden_dim, hidden_dim)
            self.update_lin = nn.Linear(hidden_dim, hidden_dim)
        elif gnn_type == "gcn":
            # In GCN we apply the same weight after normalised adjacency
            self.gcn_lin = nn.Linear(hidden_dim, hidden_dim)
        elif gnn_type == "gat":
            # Graph Attention Network (GAT) parameters. We implement a simple
            # single‑head GAT. Attention coefficients are computed using two
            # linear projections on the transformed features.
            self.gat_lin = nn.Linear(input_dim, hidden_dim)
            # Learnable attention weights: one for the source node and one for the destination
            self.att_src = nn.Linear(hidden_dim, 1, bias=False)
            self.att_dst = nn.Linear(hidden_dim, 1, bias=False)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings.

        Args:
            x: Node feature tensor of shape (N_total, F).
            adj: Block‑diagonal adjacency matrix of shape (N_total, N_total).

        Returns:
            Node embeddings tensor of shape (N_total, hidden_dim).
        """
        h = F.relu(self.input_lin(x))
        if self.gnn_type == "mpnn":
            for _ in range(self.num_layers):
                m = torch.matmul(adj, h)
                m = self.message_lin(m)
                h = F.relu(self.update_lin(h) + m)
        elif self.gnn_type == "gcn":
            # Compute symmetric normalised adjacency: D^{-1/2} A D^{-1/2}
            # Add small epsilon to avoid division by zero
            deg = torch.sum(adj, dim=1)
            deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            A_norm = D_inv_sqrt @ adj @ D_inv_sqrt
            for _ in range(self.num_layers):
                h = torch.matmul(A_norm, h)
                h = self.gcn_lin(h)
                h = F.relu(h)
        elif self.gnn_type == "gat":
            # GAT forward pass: compute node embeddings with attention over neighbours.
            # Initial linear transformation
            h = F.relu(self.gat_lin(x))
            for _ in range(self.num_layers):
                # Compute attention coefficients e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
                src = self.att_src(h)  # (N, 1)
                dst = self.att_dst(h)  # (N, 1)
                # Broadcast addition: e_ij = src_i + dst_j
                e = src + dst.T  # (N, N)
                e = F.leaky_relu(e, negative_slope=0.2)
                # Mask attention where no edge exists (adjacency weight <= 0)
                # Use a large negative value so that softmax yields zero
                zero_vec = -9e15 * torch.ones_like(e)
                attention = torch.where(adj > 0, e, zero_vec)
                attention = F.softmax(attention, dim=1)
                # Optionally weight by bond order (adj) to incorporate bond strengths
                attention = attention * adj
                # Compute new embeddings
                h = torch.matmul(attention, h)
                h = F.relu(h)
            return h
        return h
