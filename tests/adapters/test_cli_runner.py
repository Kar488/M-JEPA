import pytest
import torch
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


def test_get_rank_world_size_default(monkeypatch):
    ddp.dist.is_initialized = lambda: False
    assert ddp.get_rank() == 0
    assert ddp.get_world_size() == 1


def test_distributed_sampler_list(monkeypatch):
    monkeypatch.setattr(ddp, "get_rank", lambda: 1)
    monkeypatch.setattr(ddp, "get_world_size", lambda: 3)
    data = list(range(10))
    sampler = ddp.DistributedSamplerList(data, shuffle=False)
    assert list(sampler) == data[1::3]
    assert len(sampler) == (len(data) + 3 - 1 - 1) // 3