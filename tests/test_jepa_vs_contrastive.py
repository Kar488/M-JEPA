import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from models.ema import EMA
from training.unsupervised import train_jepa, train_contrastive

# Runtime import: needed because we *instantiate* GraphDataset
from data.mdataset import GraphDataset, GraphData
from torch import Tensor

# Type-only import: avoids circular import / “variable in type expr”
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.mdataset import GraphData
    

class ConstantEncoder(torch.nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 4) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.constant_(self.fc.weight, 1.0)

    def forward(self, g: "GraphData") -> Tensor:
        x = torch.from_numpy(g.x).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = x.mean(dim=0, keepdim=True)
        return self.fc(h)


def _make_graph(n_nodes: int) -> GraphData:
    x = np.ones((n_nodes, 2), dtype=np.float32)
    edges = []
    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return GraphData(x=x, edge_index=edge_index)


def _make_dataset() -> GraphDataset:
    graphs = [_make_graph(3), _make_graph(4)]
    return GraphDataset(graphs)


def test_jepa_vs_contrastive():
    dataset = _make_dataset()

    # --- JEPA ---
    jepa_encoder = ConstantEncoder()
    ema_encoder = ConstantEncoder()
    ema = EMA(jepa_encoder)
    ema.copy_to(ema_encoder)
    predictor = torch.nn.Identity()
    jepa_losses = train_jepa(
        dataset=dataset,
        encoder=jepa_encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema,
        epochs=2,
        batch_size=2,
        lr=0.0,
        reg_lambda=0.0,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
    )

    # --- Contrastive baseline ---
    contrastive_encoder = ConstantEncoder()
    contrastive_losses = train_contrastive(
        dataset=dataset,
        encoder=contrastive_encoder,
        projection_dim=4,
        epochs=2,
        batch_size=2,
        lr=0.0,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
    )

    assert jepa_losses and contrastive_losses
    assert jepa_losses[-1] <= contrastive_losses[-1]
