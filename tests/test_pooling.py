import torch

from utils.pooling import global_mean_pool


def test_global_mean_pool_basic():
    node_emb = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    graph_ptr = torch.tensor([0, 3, 4])
    pooled = global_mean_pool(node_emb, graph_ptr)
    expected = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
    assert torch.allclose(pooled, expected)


def test_global_mean_pool_handles_empty_graph():
    node_emb = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    graph_ptr = torch.tensor([0, 2, 2])
    pooled = global_mean_pool(node_emb, graph_ptr)
    expected = torch.tensor([[1.5, 1.5], [0.0, 0.0]])
    assert torch.allclose(pooled, expected)
