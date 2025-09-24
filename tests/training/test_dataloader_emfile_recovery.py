import errno

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from training.unsupervised import _backoff_data_loader_workers, _is_too_many_open_files


def _make_nested_emfile() -> RuntimeError:
    err = RuntimeError("loader failed")
    err.__cause__ = OSError(errno.EMFILE, "Too many open files")
    return err


def test_is_too_many_open_files_detects_nested_oseerror():
    exc = _make_nested_emfile()
    assert _is_too_many_open_files(exc)


@pytest.mark.parametrize(
    "persistent, prefetch, expected",
    [
        (True, 4, (True, False, 2)),
        (True, 1, (True, False, 1)),
        (False, 8, (True, False, 4)),
        (False, 1, (False, False, 1)),
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
