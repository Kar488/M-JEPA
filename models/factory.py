"""Factory for constructing encoder models.

The original implementation imported a number of encoder variants at module
import time.  Those imports pull in PyTorch heavy modules which makes simply
importing :mod:`models.factory` fail on systems where ``torch`` is absent.

Tests in this kata import ``models.factory`` only to ensure the module exists;
they do not actually construct encoders.  To keep the import lightweight and to
avoid a hard dependency on PyTorch for unrelated tests, the concrete encoder
implementations are imported lazily inside :func:`build_encoder`.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from models.edge_encoder import EdgeGNNEncoder  # noqa: F401
    from models.encoder import GNNEncoder  # noqa: F401
    from models.gnn_variants import (  # noqa: F401
        GATMultiHead,
        GIN,
        GraphSAGE,
        AttentiveFPEncoder,
        DMPNN,
        GINE,
        SchNet3D,
    )


def build_encoder(
    *, gnn_type: str, input_dim: int, hidden_dim: int, num_layers: int,
    edge_dim: Optional[int] = None, heads: int = 4,
):
    """Construct an encoder by name.

    Parameters mirror the original function but the heavy modules are imported
    only when required.  This keeps ``models.factory`` importable even when
    PyTorch or optional dependencies are not installed.
    """

    gt = gnn_type.lower()

    # --- edge-aware families ---
    if gt in ("edge_mpnn", "mpnn_edge", "edge"):
        if edge_dim is None:
            raise ValueError("edge_dim is required for edge_mpnn encoder.")
        from models.edge_encoder import EdgeGNNEncoder

        return EdgeGNNEncoder(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    if gt in ("gine", "gin_edge", "gin+edge"):
        if edge_dim is None:
            raise ValueError("edge_dim is required for GINE.")
        from models.gnn_variants import GINE

        return GINE(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    if gt in ("dmpnn", "chemprop"):
        if edge_dim is None:
            raise ValueError("edge_dim is required for D-MPNN.")
        from models.gnn_variants import DMPNN

        return DMPNN(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    if gt in ("attentivefp", "attnfp"):
        if edge_dim is None:
            raise ValueError("edge_dim is required for AttentiveFP.")
        from models.gnn_variants import AttentiveFPEncoder

        return AttentiveFPEncoder(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    if gt in ("schnet3d", "schnet"):
        from models.gnn_variants import SchNet3D

        return SchNet3D(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    # --- existing variants ---
    if gt in ("graphsage", "sage"):
        from models.gnn_variants import GraphSAGE

        return GraphSAGE(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    if gt in ("gin",):
        from models.gnn_variants import GIN

        return GIN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    if gt in ("gat_multi", "gatmh", "gatv2_multi"):
        from models.gnn_variants import GATMultiHead

        return GATMultiHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        )

    # fallback to your base encoder (mpnn|gcn|gat single-head)
    from models.encoder import GNNEncoder

    return GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    )

