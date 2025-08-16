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

def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

@pytest.fixture(autouse=True)
def _ddp_sane_env(monkeypatch):
    # either disable entirely:
    monkeypatch.setenv("DISABLE_DDP", "1")
    # or, if you prefer to allow single-process DDP, use loopback:
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_find_free_port()))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    
def test_train_jepa_distributed(monkeypatch):
    # set env vars for single-process distributed setup
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_find_free_port()))

    # ensure CUDA device setup is a no-op on CPU-only machines
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)
    monkeypatch.setenv("DDP_BACKEND", "gloo")

    # force init_distributed to use gloo backend
    # force gloo backend in the unsupervised module, regardless of signature
    def _force_gloo():
        try:
            return ddp.init_distributed("gloo")
        except TypeError:
            # older signature without backend param; steer via env
            monkeypatch.setenv("DDP_BACKEND", "gloo")
            return ddp.init_distributed()
    monkeypatch.setattr(ddp, "init_distributed", _force_gloo)

    graphs = [_make_graph(3), _make_graph(4)]
    dataset = type("DS", (), {"graphs": graphs})()

    encoder = DummyEncoder()
    ema_encoder = DummyEncoder()
    ema = EMA(encoder)
    ema.copy_to(ema_encoder)
    predictor = nn.Linear(2, 2)

    try:
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
        )
        pass
    finally:
        # Clean up the PG so subsequent tests don’t inherit it
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    assert isinstance(losses, list) and len(losses) == 1
