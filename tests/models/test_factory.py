import pytest

from models.factory import build_encoder
from models.edge_encoder import EdgeGNNEncoder
from models.encoder import GNNEncoder
from models.gnn_variants import GATMultiHead, GIN, GraphSAGE


def test_build_encoder_edge_mpnn_requires_edge_dim():
    with pytest.raises(ValueError):
        build_encoder(
            gnn_type="edge_mpnn",
            input_dim=4,
            hidden_dim=8,
            num_layers=2,
        )


def test_build_encoder_edge_mpnn():
    enc = build_encoder(
        gnn_type="edge_mpnn",
        input_dim=4,
        hidden_dim=8,
        num_layers=2,
        edge_dim=3,
    )
    assert isinstance(enc, EdgeGNNEncoder)


def test_build_encoder_graphsage():
    enc = build_encoder(
        gnn_type="graphsage",
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
    )
    assert isinstance(enc, GraphSAGE)


def test_build_encoder_gin():
    enc = build_encoder(
        gnn_type="gin",
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
    )
    assert isinstance(enc, GIN)


def test_build_encoder_gat_multi_head_custom_heads():
    enc = build_encoder(
        gnn_type="gat_multi",
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
        heads=2,
    )
    assert isinstance(enc, GATMultiHead)
    assert enc.layers[0].heads == 2


def test_build_encoder_default_fallback_to_gnnencoder():
    enc = build_encoder(
        gnn_type="gcn",
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
    )
    assert isinstance(enc, GNNEncoder)
    assert enc.gnn_type == "gcn"
