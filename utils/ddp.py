from __future__ import annotations

import os
from typing import Iterator, List, Sequence

import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl") -> bool:
    """Initialize DDP if env vars indicate multi-process launch; return True if initialized."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return True
    return False


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
