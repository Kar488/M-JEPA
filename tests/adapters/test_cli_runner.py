import os
import sys
import types

import pytest

import utils.ddp as ddp

# @pytest.fixture(autouse=True)
# def _disable_ddp(monkeypatch):
#     # Tell your DDP helper to bail out early; prevents c10d socket spam and NCCL errors
#     monkeypatch.setenv("DISABLE_DDP", "1")
#     # Also neutralize CUDA device selection on CPU runners
#     #monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)


def test_init_distributed_disabled(monkeypatch):
    monkeypatch.setenv("DISABLE_DDP", "1")
    assert ddp.init_distributed() is False


def test_init_distributed_worldsize_one(monkeypatch):
    monkeypatch.delenv("DISABLE_DDP", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert ddp.init_distributed() is False


def test_init_distributed_initializes(monkeypatch):
    monkeypatch.delenv("DISABLE_DDP", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")

    state = {"init": False}

    ddp.dist.is_initialized = lambda: state["init"]
    monkeypatch.setattr(ddp.dist, "is_available", lambda: True, raising=False)

    def fake_init_process_group(backend, init_method, rank, world_size):
        state["init"] = True
        fake_init_process_group.called = True
        fake_init_process_group.backend = backend
        fake_init_process_group.world_size = world_size

    monkeypatch.setattr(ddp.dist, "init_process_group", fake_init_process_group, raising=False)

    assert ddp.init_distributed() is True
    assert fake_init_process_group.called
    assert fake_init_process_group.backend in {"gloo", "nccl"}
    assert fake_init_process_group.world_size == 2


def test_init_distributed_gloo_skips_cuda_pinning(monkeypatch):
    monkeypatch.delenv("DISABLE_DDP", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class _FakeCuda:
        def __init__(self):
            self.set_calls = []

        def is_available(self):
            return True

        def device_count(self):
            return 1

        def set_device(self, idx):
            self.set_calls.append(idx)

    fake_cuda = _FakeCuda()

    class _FakeDist:
        def __init__(self):
            self.initialized = False

        def is_available(self):
            return True

        def is_initialized(self):
            return self.initialized

        def init_process_group(self, **_):
            self.initialized = True

        def get_rank(self):
            return 0

        def get_world_size(self):
            return 1

        def destroy_process_group(self):
            self.initialized = False

        def is_nccl_available(self):
            return False

    fake_torch = types.SimpleNamespace(cuda=fake_cuda)

    monkeypatch.setattr(ddp, "torch", fake_torch, raising=False)
    monkeypatch.setattr(ddp, "dist", _FakeDist(), raising=False)

    assert ddp.init_distributed(backend="gloo") is True
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is None
    assert fake_cuda.set_calls == []


def test_init_distributed_nccl_pins_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("DISABLE_DDP", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    class _FakeCuda:
        def __init__(self):
            self.set_calls = []

        def is_available(self):
            return True

        def device_count(self):
            return 2

        def set_device(self, idx):
            self.set_calls.append(idx)

    fake_cuda = _FakeCuda()

    class _FakeDist:
        def __init__(self):
            self.initialized = False

        def is_available(self):
            return True

        def is_initialized(self):
            return self.initialized

        def init_process_group(self, **_):
            self.initialized = True

        def get_rank(self):
            return 0

        def get_world_size(self):
            return 1

        def destroy_process_group(self):
            self.initialized = False

        def is_nccl_available(self):
            return True

    fake_torch = types.SimpleNamespace(cuda=fake_cuda)

    monkeypatch.setattr(ddp, "torch", fake_torch, raising=False)
    monkeypatch.setattr(ddp, "dist", _FakeDist(), raising=False)

    assert ddp.init_distributed(backend="nccl") is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    assert fake_cuda.set_calls == [0]


def test_init_distributed_nccl_requires_visible_cuda(monkeypatch):
    monkeypatch.delenv("DISABLE_DDP", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    class _FakeCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 1

    class _FakeDist:
        def is_available(self):
            return True

        def is_initialized(self):
            return False

        def init_process_group(self, **_):
            raise AssertionError("init_process_group should not be called")

        def get_rank(self):
            return 0

        def get_world_size(self):
            return 1

        def destroy_process_group(self):
            pass

        def is_nccl_available(self):
            return True

    fake_torch = types.SimpleNamespace(cuda=_FakeCuda())

    monkeypatch.setattr(ddp, "torch", fake_torch, raising=False)
    monkeypatch.setattr(ddp, "dist", _FakeDist(), raising=False)

    with pytest.raises(RuntimeError):
        ddp.init_distributed(backend="nccl")


def test_init_distributed_nccl_rejects_duplicate_visible_cuda(monkeypatch):
    monkeypatch.delenv("DISABLE_DDP", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0")

    class _FakeCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 2

        def set_device(self, idx):  # pragma: no cover - should not be called
            raise AssertionError("set_device should not be invoked when duplicates are present")

    class _FakeDist:
        def is_available(self):
            return True

        def is_initialized(self):
            return False

        def init_process_group(self, **_):  # pragma: no cover - should not run
            raise AssertionError("init_process_group should not be called")

        def get_rank(self):
            return 0

        def get_world_size(self):
            return 1

        def destroy_process_group(self):
            pass

        def is_nccl_available(self):
            return True

    fake_torch = types.SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setattr(ddp, "torch", fake_torch, raising=False)
    monkeypatch.setattr(ddp, "dist", _FakeDist(), raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        ddp.init_distributed(backend="nccl")

    message = str(excinfo.value).lower()
    assert "duplicate" in message


def test_get_rank_world_size_default(monkeypatch):
    ddp.dist.is_initialized = lambda: False
    assert ddp.get_rank() == 0
    assert ddp.get_world_size() == 1


def test_distributed_sampler_list(monkeypatch):
    monkeypatch.setattr(ddp, "get_rank", lambda: 1)
    monkeypatch.setattr(ddp, "get_world_size", lambda: 3)
    class _FakeNp:
        def arange(self, n):
            return list(range(n))

        class random:  # type: ignore[valid-type]
            @staticmethod
            def default_rng(seed=None):  # noqa: D401 - simple stub
                class _Rng:
                    def shuffle(self, seq):
                        pass

                return _Rng()

        def resize(self, array, new_size):  # type: ignore[override]
            return (array * ((new_size + len(array) - 1) // len(array)))[:new_size]

    monkeypatch.setitem(sys.modules, "numpy", _FakeNp())
    data = list(range(10))
    sampler = ddp.DistributedSamplerList(data, shuffle=False)
    assert list(sampler) == [1, 4, 7, 0]
    assert len(sampler) == 4


def test_distributed_sampler_list_handles_small_datasets(monkeypatch):
    monkeypatch.setattr(ddp, "get_rank", lambda: 2)
    monkeypatch.setattr(ddp, "get_world_size", lambda: 4)

    class _FakeNp:
        def arange(self, n):
            return list(range(n))

        class random:  # type: ignore[valid-type]
            @staticmethod
            def default_rng(seed=None):  # noqa: D401 - simple stub
                class _Rng:
                    def shuffle(self, seq):
                        pass

                return _Rng()

        def resize(self, array, new_size):  # type: ignore[override]
            return (array * ((new_size + len(array) - 1) // len(array)))[:new_size]

    monkeypatch.setitem(sys.modules, "numpy", _FakeNp())
    data = [42]
    sampler = ddp.DistributedSamplerList(data, shuffle=False)
    assert list(sampler) == [42]
    assert len(sampler) == 1
