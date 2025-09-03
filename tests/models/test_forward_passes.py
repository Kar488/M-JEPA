import importlib
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from models.gnn_variants import (
    GATMultiHead, GIN, GraphSAGE, GINE, DMPNN, AttentiveFPEncoder
)


@dataclass
class GraphData:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray | None = None
    pos: np.ndarray | None = None 
    graph_ptr: np.ndarray | None = None


class GraphDataset:
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)


def global_mean_pool(node_embeddings, graph_ptr=None):
    if graph_ptr is None:
        return node_embeddings.mean(dim=0)
    start = 0
    out = []
    for end in graph_ptr:
        out.append(node_embeddings[start:end].mean(dim=0))
        start = int(end)
    return torch.stack(out, dim=0)


@pytest.fixture()
def stub_graph_dataset(monkeypatch):
    """Provide lightweight stubs for modules with heavy deps."""

    data_dataset = types.ModuleType("data.mdataset")
    data_dataset.GraphData = GraphData
    data_dataset.GraphDataset = GraphDataset
    monkeypatch.setitem(sys.modules, "data.mdataset", data_dataset)

    utils_pooling = types.ModuleType("utils.pooling")
    utils_pooling.global_mean_pool = global_mean_pool
    monkeypatch.setitem(sys.modules, "utils.pooling", utils_pooling)

    torch_scatter = types.ModuleType("torch_scatter")

    def segment_softmax(src, index):
        return torch.ones_like(src)

    torch_scatter.segment_softmax = segment_softmax
    monkeypatch.setitem(sys.modules, "torch_scatter", torch_scatter)

    yield

    for mod in [
        "data.mdataset",
        "utils.pooling",
        "torch_scatter",
        "models.edge_encoder",
        "models.encoder",
        "models.gnn_variants",
    ]:
        sys.modules.pop(mod, None)

    import data.mdataset as real_ds  # reload original module
    importlib.reload(real_ds)


def make_graph():
    x = np.random.randn(3, 4).astype(np.float32)
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    edge_attr = np.random.randn(edge_index.shape[1], 5).astype(np.float32)
    return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)


def test_edge_gnn_encoder_forward(stub_graph_dataset):
    from models.edge_encoder import EdgeGNNEncoder

    g = make_graph()
    model = EdgeGNNEncoder(
        input_dim=4, edge_dim=5, hidden_dim=8, num_layers=2, dropout=0.0
    )
    out = model.encode_graph(g, torch.device("cpu"))
    assert out.shape == (8,)


def test_gnn_encoder_forward(stub_graph_dataset):
    from models.encoder import GNNEncoder

    g = make_graph()
    adj = torch.zeros((g.x.shape[0], g.x.shape[0]), dtype=torch.float32)
    for src, dst in g.edge_index.T:
        adj[src, dst] = 1.0
    model = GNNEncoder(input_dim=4, hidden_dim=8, num_layers=2, gnn_type="mpnn")
    h = model(torch.tensor(g.x), adj)
    assert h.shape == (g.x.shape[0], 8)


def test_gnn_variants_forward(stub_graph_dataset):
    from models.gnn_variants import GATMultiHead, GIN, GraphSAGE

    g = make_graph()
    device = torch.device("cpu")
    for cls in (GraphSAGE, GIN, GATMultiHead, GINE, DMPNN, AttentiveFPEncoder):
        model = cls(input_dim=4, hidden_dim=8, num_layers=2, edge_dim=5) \
                if cls in (GINE, DMPNN, AttentiveFPEncoder) \
                else cls(input_dim=4, hidden_dim=8, num_layers=2)
        out = model.encode_graph(g, device)
        assert out.shape == (8,)

# helper for 3D graphs (SchNet)
def make_graph_3d():
    g = make_graph()
    g.pos = np.random.randn(3, 3).astype(np.float32)  # xyz coords
    return g

def test_gine_forward(stub_graph_dataset):
    from models.gnn_variants import GINE
    g = make_graph()
    m = GINE(input_dim=4, edge_dim=5, hidden_dim=8, num_layers=2, dropout=0.0)
    out = m.encode_graph(g, torch.device("cpu"))
    assert out.shape == (8,)

def test_dmpnn_forward(stub_graph_dataset):
    from models.gnn_variants import DMPNN
    g = make_graph()
    m = DMPNN(input_dim=4, edge_dim=5, hidden_dim=8, num_layers=2, dropout=0.0)
    out = m.encode_graph(g, torch.device("cpu"))
    assert out.shape == (8,)

def test_attentivefp_forward(stub_graph_dataset):
    from models.gnn_variants import AttentiveFPEncoder
    g = make_graph()
    m = AttentiveFPEncoder(input_dim=4, edge_dim=5, hidden_dim=8, num_layers=2, dropout=0.0)
    out = m.encode_graph(g, torch.device("cpu"))
    assert out.shape == (8,)

def test_schnet3d_forward(stub_graph_dataset):
    from models.gnn_variants import SchNet3D
    g = make_graph_3d()
    m = SchNet3D(input_dim=4, hidden_dim=8, num_layers=2)
    out = m.encode_graph(g, torch.device("cpu"))
    assert out.shape == (8,)
