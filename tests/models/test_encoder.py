import torch
import pytest
import torch.nn.functional as F

from models.encoder import GNNEncoder


def test_gnn_encoder_mpnn_shape():
    x = torch.randn(3, 4)
    adj = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]]
    )
    enc = GNNEncoder(input_dim=4, hidden_dim=8, num_layers=2, gnn_type="mpnn")
    out = enc(x, adj)
    assert out.shape == (3, 8)


def test_gnn_encoder_gcn_normalization():
    x = torch.randn(3, 2)
    adj = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]]
    )
    enc = GNNEncoder(input_dim=2, hidden_dim=2, num_layers=1, gnn_type="gcn")
    with torch.no_grad():
        enc.input_lin.weight.copy_(torch.eye(2))
        enc.input_lin.bias.zero_()
        enc.gcn_lin.weight.copy_(torch.eye(2))
        enc.gcn_lin.bias.zero_()
    out = enc(x, adj)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    D = torch.diag(deg_inv_sqrt)
    A_norm = D @ adj @ D
    expected = torch.relu(A_norm @ torch.relu(x))
    assert torch.allclose(out, expected, atol=1e-6)
    expected_row_sums = torch.tensor([
        1 / (2 ** 0.5),
        2 ** 0.5,
        1 / (2 ** 0.5),
    ])
    assert torch.allclose(A_norm.sum(dim=1), expected_row_sums, atol=1e-6)


def test_gnn_encoder_gat_attention_masking():
    x = torch.randn(4, 2)
    adj = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0],
         [1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]]
    )
    enc = GNNEncoder(input_dim=2, hidden_dim=2, num_layers=1, gnn_type="gat")
    with torch.no_grad():
        enc.gat_lin.weight.copy_(torch.eye(2))
        enc.gat_lin.bias.zero_()
        enc.att_src.weight.fill_(1.0)
        enc.att_dst.weight.fill_(1.0)
    out = enc(x, adj)
    h = torch.relu(x)
    src = enc.att_src(h)
    dst = enc.att_dst(h)
    e = F.leaky_relu(src + dst.T, negative_slope=0.2)
    zero_vec = -9e15 * torch.ones_like(e)
    attention = torch.where(adj > 0, e, zero_vec)
    attention = torch.softmax(attention, dim=1)
    attention = attention * adj
    expected_out = torch.relu(attention @ h)
    assert torch.allclose(out, expected_out, atol=1e-6)
    row_sums = attention.sum(dim=1)
    expected = torch.tensor([1.0, 1.0, 1.0, 0.0])
    assert torch.allclose(row_sums, expected, atol=1e-6)
    assert torch.allclose(attention[adj == 0], torch.zeros_like(attention[adj == 0]))
    assert out.shape == (4, 2)


def test_gnn_encoder_invalid_type():
    with pytest.raises(ValueError):
        GNNEncoder(input_dim=1, hidden_dim=2, num_layers=1, gnn_type="bad_type")