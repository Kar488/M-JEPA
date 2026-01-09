from __future__ import annotations

import multiprocessing as mp
import os
import socket
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.supervised import train_linear_head
import utils.ddp as ddp


class TinyGraph:
    def __init__(self, feat: np.ndarray) -> None:
        self.feat = torch.tensor(feat, dtype=torch.float32)

    def to_tensors(self):
        return self.feat.unsqueeze(0), torch.zeros((1, 1), dtype=torch.float32)


class TinyDataset:
    def __init__(self, labels: List[int], feat_dim: int = 2) -> None:
        self.labels = np.array(labels, dtype=np.float32)
        self.graphs = [
            TinyGraph(np.full((feat_dim,), fill_value=float(i), dtype=np.float32))
            for i in range(len(labels))
        ]
        self.smiles = None

    def __len__(self) -> int:
        return len(self.graphs)

    def get_batch(self, indices):
        xs = []
        adjs = []
        for idx in indices:
            x, adj = self.graphs[idx].to_tensors()
            xs.append(x)
            adjs.append(adj)
        batch_x = torch.cat(xs, dim=0)
        batch_adj = torch.block_diag(*adjs)
        ptr = torch.arange(0, len(indices) + 1, dtype=torch.long)
        batch_labels = torch.tensor(self.labels[indices], dtype=torch.float32)
        return batch_x, batch_adj, ptr, batch_labels


class DummyEncoder(nn.Module):
    def __init__(self, dim: int = 2) -> None:
        super().__init__()
        self.hidden_dim = dim
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))

    def forward(self, x, adj):  # noqa: ARG002
        return self.linear(x)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_worker(
    rank: int, world_size: int, port: int, queue: mp.Queue, oversample: bool
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["DDP_FORCE_BACKEND"] = "gloo"
    os.environ.pop("DISABLE_DDP", None)

    torch.manual_seed(0)
    np.random.seed(0)

    if oversample:
        labels = [0] * 12 + [1] * 3
    else:
        labels = [0, 1] * 7 + [0]

    dataset = TinyDataset(labels)
    encoder = DummyEncoder()
    metrics = train_linear_head(
        dataset,
        encoder,
        "classification",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        patience=0,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        devices=world_size,
        train_indices=list(range(13)),
        val_indices=[13],
        test_indices=[14],
        oversample_minority=oversample,
    )
    if rank == 0:
        queue.put(float(metrics.get("train/epoch_batches", 0.0)))
    ddp.cleanup()


def _run_ddp_training(oversample: bool):
    if "spawn" not in mp.get_all_start_methods():
        pytest.skip("spawn multiprocessing start method is unavailable")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is not available")
    port = _free_port()
    world_size = 2
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    torch.multiprocessing.spawn(
        _ddp_worker,
        args=(world_size, port, queue, oversample),
        nprocs=world_size,
        join=True,
    )
    return queue.get(timeout=10)


def test_ddp_train_loader_no_truncation():
    metrics = _run_ddp_training(oversample=False)
    assert metrics == pytest.approx(2.0)


def test_ddp_oversample_minority_runs():
    metrics = _run_ddp_training(oversample=True)
    assert metrics >= 1.0
