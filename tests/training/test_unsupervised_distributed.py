import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from data.mdataset import GraphData
from models.ema import EMA
from training import unsupervised as unsup
import utils.ddp as ddp
import os, socket, contextlib, pytest
import pytest
pytestmark = pytest.mark.notgpu

class DummyEncoder(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, g: GraphData):
        x = torch.from_numpy(g.x).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.numel() == 0:
            x = torch.zeros((1, self.fc.in_features), dtype=torch.float32)
        h = x.mean(dim=0, keepdim=True)
        return self.fc(h)


def _make_graph(n_nodes: int) -> GraphData:
    x = np.random.randn(n_nodes, 2).astype(np.float32)
    edge_index = np.zeros((2, 0), dtype=np.int64)
    return GraphData(x=x, edge_index=edge_index)


def test_train_jepa_distributed(monkeypatch):

    monkeypatch.delenv("DISABLE_DDP", raising=False)
    graphs = [_make_graph(3), _make_graph(4)]
    dataset = type("DS", (), {"graphs": graphs})()

    encoder = DummyEncoder()
    ema_encoder = DummyEncoder()
    ema = EMA(encoder); ema.copy_to(ema_encoder)
    predictor = nn.Linear(2, 2)

    losses = unsup.train_jepa(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema,
        epochs=1,
        batch_size=1,
        device="cpu",
        use_amp=False,
        # devices defaults to 1; ensure  training code only initialises DDP when devices > 1
    )

    # No DDP initialised -> no need to destroy process group
    assert isinstance(losses, list) and len(losses) == 1