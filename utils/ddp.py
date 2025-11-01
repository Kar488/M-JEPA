from __future__ import annotations

import contextlib
import logging
import os
import platform
import socket
import types
from typing import Iterator, Sequence, TYPE_CHECKING

logger = logging.getLogger(__name__)


def _pin_visible_cuda_device_to_local_rank() -> None:
    """Restrict ``CUDA_VISIBLE_DEVICES`` to the device assigned to this rank."""

    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is None:
        return

    is_available = getattr(cuda_mod, "is_available", None)
    if not callable(is_available) or not is_available():
        return

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        local_world_size = int(
            os.environ.get(
                "LOCAL_WORLD_SIZE",
                os.environ.get("WORLD_SIZE", "1"),
            )
        )
    except ValueError:
        return

    if local_world_size <= 1:
        return

    raw_mask = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if raw_mask:
        devices = [entry.strip() for entry in raw_mask.split(",") if entry.strip()]
    else:
        device_count_fn = getattr(cuda_mod, "device_count", None)
        try:
            available = int(device_count_fn()) if callable(device_count_fn) else 0
        except Exception:
            available = 0
        devices = [str(i) for i in range(max(0, available))]

    if not devices:
        return

    if len(devices) < local_world_size:
        logger.error(
            "Distributed launch requested %d CUDA devices per node but only %d are visible (%s).",
            local_world_size,
            len(devices),
            raw_mask or "device_count",
        )
        raise RuntimeError(
            "Insufficient CUDA devices for distributed launch. Set CUDA_VISIBLE_DEVICES "
            "to a comma-separated list with at least LOCAL_WORLD_SIZE entries."
        )

    if not (0 <= local_rank < local_world_size):
        logger.warning(
            "LOCAL_RANK=%s outside expected range [0, %d); skipping CUDA pinning.",
            local_rank,
            local_world_size,
        )
        return

    selected = devices[local_rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    logger.debug(
        "Pinned CUDA_VISIBLE_DEVICES to '%s' for LOCAL_RANK=%d", selected, local_rank
    )

    set_device = getattr(cuda_mod, "set_device", None)
    if callable(set_device):
        try:
            set_device(0)
        except Exception:
            logger.debug("Failed to set CUDA device after pinning", exc_info=True)


def _visible_cuda_device_count() -> int:
    """Return the number of CUDA devices visible to the current process."""

    mask = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if mask:
        devices = [entry.strip() for entry in mask.split(",") if entry.strip()]
        return len(devices)

    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is None:
        return 0

    device_count_fn = getattr(cuda_mod, "device_count", None)
    try:
        available = int(device_count_fn()) if callable(device_count_fn) else 0
    except Exception:
        available = 0
    return max(0, available)


def _find_free_port() -> int:
    """Return an available TCP port on the current host."""

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


try:  # pragma: no cover - exercised when torch is installed
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore
except Exception:  # pragma: no cover - torch optional in some environments
    torch = types.SimpleNamespace(  # type: ignore[assignment]
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    dist = types.SimpleNamespace(  # type: ignore[assignment]
        is_available=lambda: False,
        is_initialized=lambda: False,
        init_process_group=lambda **_: None,
        get_rank=lambda: 0,
        get_world_size=lambda: 1,
        destroy_process_group=lambda: None,
    )
else:
    if not hasattr(dist, "is_nccl_available"):
        dist.is_nccl_available = lambda: False  # type: ignore[attr-defined]


if TYPE_CHECKING:  # pragma: no cover - typing only
    import pytest
else:  # pragma: no cover - import lazily to avoid hard dependency at runtime
    try:
        import pytest  # type: ignore
    except Exception:  # pragma: no cover - pytest not installed outside tests
        pytest = None  # type: ignore[assignment]


if pytest is not None:  # pragma: no cover - executed only during tests
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    dist_stub = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: False,
        init_process_group=lambda **_: None,
        get_rank=lambda: 0,
        get_world_size=lambda: 1,
        destroy_process_group=lambda: None,
        is_nccl_available=lambda: False,
    )

    @pytest.fixture(autouse=True)
    def _patch_ddp(monkeypatch):
        """Provide lightweight DDP stubs during unit tests."""

        import utils.ddp as ddp

        monkeypatch.setattr(ddp, "torch", torch_stub, raising=False)
        monkeypatch.setattr(ddp, "dist", dist_stub, raising=False)
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(_find_free_port()))
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "0")


def init_distributed(backend: str | None = None) -> bool:
    """Initialise ``torch.distributed`` when running with multiple ranks."""

    if os.environ.get("DISABLE_DDP") == "1":
        return False

    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except (TypeError, ValueError):
        world_size = 1
    try:
        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
        )
    except (TypeError, ValueError):
        local_world_size = world_size
    if world_size <= 1:
        return False

    available_devices = _visible_cuda_device_count()
    cuda_mod = getattr(torch, "cuda", None)
    cuda_available = bool(
        getattr(cuda_mod, "is_available", lambda: False)()
    ) if cuda_mod is not None else False

    if not dist.is_available() or dist.is_initialized():
        return dist.is_initialized()

    # Default to a loopback rendezvous so unit tests / local runs avoid hostname lookups.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))

    if backend is None:
        want_nccl = (
            cuda_available
            and hasattr(dist, "is_nccl_available")
            and dist.is_nccl_available()
            and platform.system() != "Windows"
        )
        backend = "nccl" if want_nccl else "gloo"

    if backend == "nccl" and (
        not hasattr(dist, "is_nccl_available") or not dist.is_nccl_available()
    ):
        logger.warning("NCCL not available; falling back to gloo")
        backend = "gloo"

    if backend == "nccl":
        try:
            _pin_visible_cuda_device_to_local_rank()
        except RuntimeError:
            raise
        except Exception:
            logger.debug("Unable to pin CUDA devices", exc_info=True)

    if backend == "nccl" and local_world_size > 1 and available_devices < local_world_size:
        message = (
            "Insufficient CUDA devices for distributed launch: requested "
            f"{local_world_size} per node but only {available_devices} visible. "
            "Reduce --devices or set CUDA_VISIBLE_DEVICES accordingly."
        )
        logger.error(message)
        raise RuntimeError(message)

    rank = int(os.environ.get("RANK", "0"))
    dist.init_process_group(
        backend=backend, init_method="env://", rank=rank, world_size=world_size
    )
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
        shard = idx[self.rank :: self.world]
        for i in shard:
            yield self.data[i]

    def __len__(self) -> int:
        n = len(self.data)
        return (n + self.world - 1 - self.rank) // self.world
