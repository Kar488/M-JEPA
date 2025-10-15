"""Tests for the W&B schema discovery helpers."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("yaml")

from reports import discover_schema


class _FailingIterator:
    def __iter__(self):  # pragma: no cover - iterator protocol shim
        return self

    def __next__(self):  # pragma: no cover - iterator protocol shim
        raise RuntimeError("timeout")


class _FailingApi:
    def runs(self, project_path, *, per_page=None):  # noqa: D401 - signature parity
        return _FailingIterator()


class _Run:
    def __init__(self):
        self.id = "run-1"
        self.name = "run-1"
        self.tags = ["ci"]
        self.group = "group"
        self.job_type = "job"
        self.summary = {"accuracy": 0.9}
        self.config = {"lr": 0.001}


class _SuccessfulApi:
    def runs(self, project_path, *, per_page=None):  # noqa: D401 - signature parity
        return [_Run()]


@pytest.fixture(autouse=True)
def _clear_timeout_env(monkeypatch):
    monkeypatch.delenv("WANDB_HTTP_TIMEOUT", raising=False)
    yield
    monkeypatch.delenv("WANDB_HTTP_TIMEOUT", raising=False)


def test_collect_remote_schema_retries_on_timeout(monkeypatch):
    calls = []
    apis = iter([_FailingApi(), _SuccessfulApi()])

    def _fake_get_wandb_api(*, timeout, **kwargs):
        calls.append(timeout)
        return next(apis)

    monkeypatch.setattr(discover_schema, "get_wandb_api", _fake_get_wandb_api)
    monkeypatch.setenv("WANDB_API_KEY", "testing")

    payload = discover_schema._collect_remote_schema(
        "entity", "project", max_runs=5
    )

    assert payload["metrics"]["accuracy"] == {"summary"}
    assert payload["configs"]["lr"] == {"config"}
    assert calls == [90, 180]
    assert os.environ["WANDB_HTTP_TIMEOUT"] == "180"


def test_collect_remote_schema_stops_when_api_unavailable(monkeypatch):
    monkeypatch.setattr(discover_schema, "get_wandb_api", lambda **_: None)
    monkeypatch.setenv("WANDB_API_KEY", "testing")

    payload = discover_schema._collect_remote_schema("entity", "project", max_runs=5)

    assert payload == {}
