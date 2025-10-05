import numpy as np
import torch
from types import SimpleNamespace

from utils.graph_ops import (
    _to_tensor,
    _edge_index_to_dense,
    _encode_graph,
    _pool_graph_emb,
    _ref_device,
    _ensure_edge_attr_np_or_torch,
)
from data.mdataset import GraphData


class DummyEncoder(torch.nn.Module):
    def forward(self, x, adj, edge_attr=None):
        # store inputs for assertions
        self.last = (x, adj, edge_attr)
        return x


def test_to_tensor_basic():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    t = _to_tensor(arr, dtype=torch.float64, device=torch.device("cpu"))
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float64
    assert t.device.type == "cpu"
    assert _to_tensor(None) is None


def test_edge_index_to_dense():
    ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    dense = _edge_index_to_dense(ei, num_nodes=3, device=torch.device("cpu"))
    expected = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    assert torch.equal(dense, expected)
    empty = _edge_index_to_dense(torch.zeros((2, 0), dtype=torch.long), 3, torch.device("cpu"))
    assert torch.equal(empty, torch.eye(3))


def test_encode_graph_builds_adj_from_edge_index():
    g = GraphData(
        x=np.array([[1.0], [2.0]], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    enc = DummyEncoder()
    out = _encode_graph(enc, g)
    assert torch.equal(out, enc.last[0])
    expected_adj = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    assert torch.equal(enc.last[1], expected_adj)
    assert enc.last[2] is None


def test_pool_graph_emb_batch_and_single():
    h = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    batch = torch.tensor([0, 0, 1, 1])
    g = SimpleNamespace(batch=batch)
    pooled = _pool_graph_emb(h, g)
    expected = torch.tensor([[2.0, 3.0], [6.0, 7.0]])
    assert torch.allclose(pooled, expected)

    g2 = SimpleNamespace()
    single = _pool_graph_emb(h[:2], g2)
    expected_single = torch.tensor([2.0, 3.0])
    assert torch.allclose(single, expected_single)

def test_ensure_edge_attr_np_or_torch(monkeypatch):
    g = SimpleNamespace(x=torch.ones(2, 3), edge_index=torch.tensor([[0,1],[1,0]], dtype=torch.long))
    g2 = _ensure_edge_attr_np_or_torch(g, need_dim=4)
    assert g2.edge_attr.shape == (2, 4)
    # now with existing smaller edge_attr
    g3 = SimpleNamespace(x=torch.ones(2,3), edge_index=torch.tensor([[0,1],[1,0]], dtype=torch.long), edge_attr=torch.ones(2,2))
    g3 = _ensure_edge_attr_np_or_torch(g3, need_dim=4)
    assert g3.edge_attr.shape == (2,4)
    g4 = SimpleNamespace(x=torch.ones(2,3), edge_index=torch.tensor([[0,1],[1,0]], dtype=torch.long), edge_attr=torch.ones(2,6))
    g4 = _ensure_edge_attr_np_or_torch(g4, need_dim=3)
    assert g4.edge_attr.shape == (2,3)


def test_encode_graph_flex_handles_factory_and_legacy():
    from utils.graph_ops import _encode_graph_flex
    g = SimpleNamespace(x=np.ones((2,3), dtype=np.float32), edge_index=np.array([[0,1],[1,0]], dtype=np.int64))
    class Factory:
        def __call__(self, graph):
            self.graph = graph
            return "ok"
    out = _encode_graph_flex(Factory(), g)
    assert out == "ok"
    class Legacy(torch.nn.Module):
        def forward(self, x, adj):
            self.x, self.adj = x, adj
            return x
    enc = Legacy()
    out2 = _encode_graph_flex(enc, g, device=torch.device("cpu"))
    assert isinstance(out2, torch.Tensor)
    assert enc.x.shape[0] == 2 and enc.adj.shape[0] == 2


def test_ref_device_no_parameters():
    class NoParam(torch.nn.Module):
        def __init__(self):
            super().__init__()

    module = NoParam()
    assert _ref_device(module).type == "cpu"
