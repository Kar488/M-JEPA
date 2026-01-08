from __future__ import annotations

import os

import utils.ddp as ddp


def test_nccl_watchdog_env_defaults(monkeypatch):
    monkeypatch.delenv("NCCL_BLOCKING_WAIT", raising=False)
    monkeypatch.delenv("NCCL_ASYNC_ERROR_HANDLING", raising=False)
    monkeypatch.delenv("NCCL_TIMEOUT", raising=False)
    monkeypatch.setenv("DDP_NCCL_TIMEOUT", "240")
    monkeypatch.setattr(ddp, "_NCCL_ENV_CONFIGURED", False, raising=False)

    ddp._ensure_nccl_watchdog_env()

    assert os.environ["NCCL_BLOCKING_WAIT"] == "1"
    assert os.environ["NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert os.environ["NCCL_TIMEOUT"] == "240"


def test_nccl_watchdog_env_respects_existing(monkeypatch):
    monkeypatch.setenv("NCCL_BLOCKING_WAIT", "0")
    monkeypatch.setenv("NCCL_ASYNC_ERROR_HANDLING", "0")
    monkeypatch.setenv("NCCL_TIMEOUT", "99")
    monkeypatch.setenv("DDP_NCCL_TIMEOUT", "120")
    monkeypatch.setattr(ddp, "_NCCL_ENV_CONFIGURED", False, raising=False)

    ddp._ensure_nccl_watchdog_env()

    assert os.environ["NCCL_BLOCKING_WAIT"] == "0"
    assert os.environ["NCCL_ASYNC_ERROR_HANDLING"] == "0"
    assert os.environ["NCCL_TIMEOUT"] == "99"
