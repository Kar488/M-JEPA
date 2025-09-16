from __future__ import annotations

import torch

from utils.indexing import gather_nodes


def test_gather_nodes_matches_index_select_gradients():
    x = torch.randn(5, 3, requires_grad=True)
    index = torch.tensor([0, 2, 2, 4])

    custom = gather_nodes(x, index)
    expected = x.index_select(0, index)
    assert torch.allclose(custom, expected)

    loss = (custom**2).sum()
    loss.backward()
    grad_custom = x.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    ref_out = x_ref.index_select(0, index)
    ref_loss = (ref_out**2).sum()
    ref_loss.backward()
    assert torch.allclose(grad_custom, x_ref.grad)


def test_gather_nodes_supports_higher_rank_tensors():
    x = torch.randn(4, 2, 3, requires_grad=True)
    index = torch.tensor([3, 1])

    out = gather_nodes(x, index)
    expected = x.index_select(0, index)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected)

    loss = out.sum()
    loss.backward()
    assert x.grad.shape == x.shape
