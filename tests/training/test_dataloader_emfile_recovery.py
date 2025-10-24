import errno
import types

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from training.unsupervised import (
    _backoff_data_loader_workers,
    _is_too_many_open_files,
)
from utils import dataloader as dataloader_utils
from utils.dataloader import check_fd_budget, ensure_open_file_limit


def _make_nested_emfile() -> RuntimeError:
    err = RuntimeError("loader failed")
    err.__cause__ = OSError(errno.EMFILE, "Too many open files")
    return err


def test_is_too_many_open_files_detects_nested_oseerror():
    exc = _make_nested_emfile()
    assert _is_too_many_open_files(exc)


def test_is_too_many_open_files_detects_plain_oserror():
    err = OSError(errno.EMFILE, "Too many open files")
    assert _is_too_many_open_files(err)


@pytest.mark.parametrize(
    "persistent, prefetch, expected",
    [
        (True, 4, (True, False, 2)),
        (True, 1, (True, False, 1)),
        (False, 8, (True, False, 4)),
        (False, 1, (False, False, 1)),
        (True, None, (True, False, None)),
    ],
)
def test_backoff_data_loader_workers(persistent, prefetch, expected):
    assert _backoff_data_loader_workers(persistent, prefetch) == expected


def test_is_too_many_open_files_handles_plain_runtimeerror():
    assert not _is_too_many_open_files(RuntimeError("worker exit"))


def test_is_too_many_open_files_handles_oserror_without_errno():
    err = OSError("weird failure")
    assert not _is_too_many_open_files(err)


def test_is_too_many_open_files_detects_message():
    exc = RuntimeError("DataLoader crashed: Too many open files")
    assert _is_too_many_open_files(exc)


def test_ensure_open_file_limit_raises_soft_cap(monkeypatch):
    pytest.importorskip("resource")

    calls = []

    stub = types.SimpleNamespace()
    stub.RLIMIT_NOFILE = 7
    stub.getrlimit = lambda limit: (1024, 8192)

    def fake_set(limit, values):
        calls.append(values)

    stub.setrlimit = fake_set

    monkeypatch.setattr(dataloader_utils, "resource", stub, raising=False)

    soft, hard = ensure_open_file_limit(4096)

    assert calls[-1] == (4096, 8192)
    assert soft == 4096
    assert hard == 8192


def test_ensure_open_file_limit_falls_back_to_hard_limit(monkeypatch):
    pytest.importorskip("resource")

    calls = []

    stub = types.SimpleNamespace()
    stub.RLIMIT_NOFILE = 7
    stub.getrlimit = lambda limit: (1024, 2048)

    def fake_set(limit, values):
        calls.append(values)
        if values[0] > 2048:
            raise ValueError

    stub.setrlimit = fake_set

    monkeypatch.setattr(dataloader_utils, "resource", stub, raising=False)

    soft, hard = ensure_open_file_limit(4096)

    assert calls[-1] == (2048, 2048)
    assert soft == 2048
    assert hard == 2048


def test_check_fd_budget_reports_available(monkeypatch):
    pytest.importorskip("resource")

    stub_resource = types.SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=lambda limit: (8192, 16384),
    )
    monkeypatch.setattr(dataloader_utils, "resource", stub_resource, raising=False)
    monkeypatch.setattr(dataloader_utils.os.path, "isdir", lambda _: True)
    monkeypatch.setattr(
        dataloader_utils.os,
        "listdir",
        lambda _: [str(i) for i in range(1024)],
    )

    budget = check_fd_budget(512)

    assert budget.ok
    assert budget.soft_limit == 8192
    assert budget.available == 8192 - 1024
    assert budget.open_files == 1024


def test_check_fd_budget_detects_shortfall(monkeypatch):
    pytest.importorskip("resource")

    stub_resource = types.SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=lambda limit: (2048, 4096),
    )
    monkeypatch.setattr(dataloader_utils, "resource", stub_resource, raising=False)
    monkeypatch.setattr(dataloader_utils.os.path, "isdir", lambda _: True)
    monkeypatch.setattr(
        dataloader_utils.os,
        "listdir",
        lambda _: ["fd"] * 1536,
    )

    budget = check_fd_budget(1024)

    assert not budget.ok
    assert budget.available == 512
