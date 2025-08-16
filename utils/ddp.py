from __future__ import annotations

import os
from typing import Iterator, List, Sequence

import torch
import torch.distributed as dist


import os, socket, platform, contextlib, logging
import torch
import torch.distributed as dist
logger = logging.getLogger(__name__)


def _find_free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def init_distributed(backend: str | None = None) -> bool:
     # allow explicit disable via env or single-process world size
    if os.environ.get("DISABLE_DDP") == "1":
        return False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if not dist.is_available() or dist.is_initialized():
            return dist.is_initialized()

    # default to safe addr/port (loopback)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))

    # Pick a safe default backend
    if backend is None:
        want_nccl = (
            torch.cuda.is_available()
            and hasattr(dist, "is_nccl_available")
            and dist.is_nccl_available()
            and platform.system() != "Windows"
        )
        backend = "nccl" if want_nccl else "gloo"

    if backend == "nccl" and (not hasattr(dist, "is_nccl_available") or not dist.is_nccl_available()):
        logger.warning("NCCL not available; falling back to gloo")
        backend = "gloo"

   
    rank = int(os.environ.get("RANK", "0"))
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    
    return True

def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class DistributedSamplerList:
    """Very small sampler for a list dataset; shards indices across ranks."""

    def __init__(self, data: Sequence, shuffle: bool = True):
        self.data = data
        self.shuffle = shuffle
        self.rank = get_rank()
        self.world = get_world_size()

    def __iter__(self) -> Iterator:
        import numpy as np

        n = len(self.data)
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(seed=self.rank)  # simple per-rank seed
            rng.shuffle(idx)
        # shard
        shard = idx[self.rank :: self.world]
        for i in shard:
            yield self.data[i]

    def __len__(self) -> int:
        n = len(self.data)
        # number of items this rank will see
        return (n + self.world - 1 - self.rank) // self.world
