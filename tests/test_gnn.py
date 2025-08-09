import torch

from models.encoder import GNNEncoder


def test_gnn_forward_pass():
    x = torch.tensor([[1.0], [0.0], [1.0]])
    adj = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    gnn = GNNEncoder(input_dim=1, hidden_dim=4, num_layers=2, gnn_type="gcn")
    out = gnn(x, adj)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()
