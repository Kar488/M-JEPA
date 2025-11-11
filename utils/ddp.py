from __future__ import annotations

import contextlib
import logging
import math
import os
import platform
import socket
import types
from typing import Iterator, Sequence, TYPE_CHECKING

logger = logging.getLogger(__name__)

_CUDA_VISIBLE_DEVICE_STACK: list[tuple[bool, str]] = []


def _restore_cuda_mask_snapshot() -> None:
    """Ensure the visible CUDA mask matches the latest saved snapshot."""

    if not _CUDA_VISIBLE_DEVICE_STACK:
        return

    had_env, mask = _CUDA_VISIBLE_DEVICE_STACK[-1]
    if had_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = mask
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def _remember_original_cuda_mask() -> None:
    """Push the current CUDA visibility mask onto a stack."""

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        _CUDA_VISIBLE_DEVICE_STACK.append((True, os.environ["CUDA_VISIBLE_DEVICES"]))
    else:
        _CUDA_VISIBLE_DEVICE_STACK.append((False, ""))


def _restore_original_cuda_mask() -> None:
    """Pop and restore the most recent CUDA visibility mask."""

    if not _CUDA_VISIBLE_DEVICE_STACK:
        return

    had_env, mask = _CUDA_VISIBLE_DEVICE_STACK.pop()
    if had_env and mask:
        os.environ["CUDA_VISIBLE_DEVICES"] = mask
    elif had_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def should_retry_with_gloo(exc: BaseException) -> bool:
    """Return ``True`` when a distributed failure suggests a gloo retry."""

    needles = (
        "Duplicate GPU detected",
        "ncclInvalidUsage",
        "contains duplicate entries",
        "Each distributed rank must map to a unique GPU",
    )

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if any(token in message for token in needles):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    return False


def _resolve_visible_cuda_devices() -> tuple[list[str], list[str], str]:
    """Return the unique CUDA device entries and any duplicates in the mask."""

    raw_mask = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    duplicates: list[str] = []

    if raw_mask:
        unique: list[str] = []
        seen: dict[str, str] = {}
        for entry in raw_mask.split(","):
            token = entry.strip()
            if not token:
                continue
            canonical = token
            if ":" in canonical:
                prefix, _, suffix = canonical.partition(":")
                if prefix.lower() == "cuda":
                    canonical = suffix
            key = canonical.strip().lower()
            if key in seen:
                duplicates.append(token)
                continue
            seen[key] = token
            unique.append(token)
        return unique, duplicates, raw_mask

    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is None:
        return [], [], ""

    device_count_fn = getattr(cuda_mod, "device_count", None)
    try:
        available = int(device_count_fn()) if callable(device_count_fn) else 0
    except Exception:
        available = 0

    unique = [str(i) for i in range(max(0, available))]
    return unique, duplicates, ""


def _pin_visible_cuda_device_to_local_rank() -> str | None:
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

    if _CUDA_VISIBLE_DEVICE_STACK:
        _restore_cuda_mask_snapshot()

    devices, duplicates, raw_mask = _resolve_visible_cuda_devices()
    _remember_original_cuda_mask()

    if not devices:
        return None

    if len(devices) < local_world_size:
        descriptor = raw_mask or "device_count"
        if duplicates:
            descriptor = f"{descriptor} (duplicates trimmed)".strip()
        logger.error(
            "Distributed launch requested %d CUDA devices per node but only %d are visible (%s).",
            local_world_size,
            len(devices),
            descriptor,
        )
        try:
            raise RuntimeError(
                "Insufficient CUDA devices for distributed launch. Set CUDA_VISIBLE_DEVICES "
                "to a comma-separated list with at least LOCAL_WORLD_SIZE entries."
            )
        finally:
            _restore_original_cuda_mask()

    if not (0 <= local_rank < local_world_size):
        logger.warning(
            "LOCAL_RANK=%s outside expected range [0, %d); skipping CUDA pinning.",
            local_rank,
            local_world_size,
        )
        return None

    _remember_original_cuda_mask()
    selected = devices[local_rank]

    try:
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
    except Exception:
        _restore_original_cuda_mask()
        raise

    return selected


def _visible_cuda_device_count() -> int:
    """Return the number of CUDA devices visible to the current process."""

    devices, _, _ = _resolve_visible_cuda_devices()
    return len(devices)


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

    visible_devices, duplicate_entries, raw_mask = _resolve_visible_cuda_devices()
    available_devices = len(visible_devices)
    cuda_mod = getattr(torch, "cuda", None)
    cuda_available = bool(
        getattr(cuda_mod, "is_available", lambda: False)()
    ) if cuda_mod is not None else False

    if not dist.is_available() or dist.is_initialized():
        return dist.is_initialized()

    # Default to a loopback rendezvous so unit tests / local runs avoid hostname lookups.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))

    backend_override = os.environ.get("DDP_FORCE_BACKEND")
    if backend_override:
        override = backend_override.strip().lower()
        if override:
            if backend is not None and override != backend:
                logger.info(
                    "Overriding distributed backend via DDP_FORCE_BACKEND=%s (was %s)",
                    override,
                    backend,
                )
            backend = override

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

    if backend == "nccl" and local_world_size > 1 and duplicate_entries:
        deduped = sorted({entry.strip() for entry in duplicate_entries if entry.strip()})
        detail = ", ".join(deduped) or raw_mask or "CUDA_VISIBLE_DEVICES"
        message = (
            "CUDA_VISIBLE_DEVICES contains duplicate entries "
            f"({detail}) but LOCAL_WORLD_SIZE={local_world_size}. "
            "Each distributed rank must map to a unique GPU; adjust the mask or "
            "reduce --devices."
        )
        logger.error(message)
        raise RuntimeError(message)

    pinned_device: str | None = None
    if backend == "nccl":
        try:
            pinned_device = _pin_visible_cuda_device_to_local_rank()
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

    if pinned_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned_device

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
    while _CUDA_VISIBLE_DEVICE_STACK:
        _restore_original_cuda_mask()


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
        if n == 0:
            return iter(())

        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(seed=self.rank)
            rng.shuffle(indices)

        world = max(1, self.world)
        shard_size = math.ceil(n / world)
        total_needed = shard_size * world
        if total_needed > n:
            reps = math.ceil(total_needed / n)
            padded = np.resize(indices, reps * n)[:total_needed]
        else:
            padded = indices[:total_needed]

        start = self.rank
        stop = start + shard_size * world
        shard_indices = padded[start:stop:world]
        for idx in shard_indices:
            yield self.data[int(idx)]

    def __len__(self) -> int:
        n = len(self.data)
        if n == 0:
            return 0
        return math.ceil(n / max(1, self.world))
