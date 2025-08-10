import sys
import types
from dataclasses import dataclass

import numpy as np
import torch

# Stub out modules that require heavy dependencies (e.g., RDKit)
data_dataset = types.ModuleType("data.dataset")

@dataclass
class GraphData:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray | None = None


class GraphDataset:
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

data_dataset.GraphData = GraphData
data_dataset.GraphDataset = GraphDataset
sys.modules["data.dataset"] = data_dataset

# Provide a minimal global_mean_pool implementation
utils_pooling = types.ModuleType("utils.pooling")

def global_mean_pool(node_embeddings, graph_ptr=None):
    if graph_ptr is None:
        return node_embeddings.mean(dim=0)
    start = 0
    out = []
    for end in graph_ptr:
        out.append(node_embeddings[start:end].mean(dim=0))
        start = int(end)
    return torch.stack(out, dim=0)

utils_pooling.global_mean_pool = global_mean_pool
sys.modules["utils.pooling"] = utils_pooling

torch_scatter = types.ModuleType("torch_scatter")

def segment_softmax(src, index):
    return torch.ones_like(src)

torch_scatter.segment_softmax = segment_softmax
sys.modules["torch_scatter"] = torch_scatter

from models.edge_encoder import EdgeGNNEncoder
from models.encoder import GNNEncoder
from models.gnn_variants import GraphSAGE, GIN, GATMultiHead


def make_graph():
    x = np.random.randn(3, 4).astype(np.float32)
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    edge_attr = np.random.randn(edge_index.shape[1], 5).astype(np.float32)
    return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)


def test_edge_gnn_encoder_forward():
    g = make_graph()
    model = EdgeGNNEncoder(input_dim=4, edge_dim=5, hidden_dim=8, num_layers=2, dropout=0.0)
    out = model.encode_graph(g, torch.device("cpu"))
    assert out.shape == (8,)


def test_gnn_encoder_forward():
    g = make_graph()
    adj = torch.zeros((g.x.shape[0], g.x.shape[0]), dtype=torch.float32)
    for src, dst in g.edge_index.T:
        adj[src, dst] = 1.0
    model = GNNEncoder(input_dim=4, hidden_dim=8, num_layers=2, gnn_type="mpnn")
    h = model(torch.tensor(g.x), adj)
    assert h.shape == (g.x.shape[0], 8)


def test_gnn_variants_forward():
    g = make_graph()
    device = torch.device("cpu")
    for cls in (GraphSAGE, GIN, GATMultiHead):
        model = cls(input_dim=4, hidden_dim=8, num_layers=2)
        out = model.encode_graph(g, device)
        assert out.shape == (8,)
