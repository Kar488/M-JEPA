from __future__ import annotations

import sys
import os
import types

import pytest


fake_pandas = types.ModuleType("pandas")


class _FakeSeries(list):
    def mean(self):
        return sum(self) / len(self)

    def std(self, ddof=0):  # pragma: no cover - compatibility shim
        return 0.0

    def median(self):  # pragma: no cover - compatibility shim
        sorted_values = sorted(self)
        mid = len(sorted_values) // 2
        return sorted_values[mid]

    def quantile(self, q):  # pragma: no cover - compatibility shim
        if not self:
            raise ValueError("empty series")
        sorted_values = sorted(self)
        index = min(int(q * (len(sorted_values) - 1)), len(sorted_values) - 1)
        return sorted_values[index]


class _FakeDataFrame(dict):
    empty = True

    def dropna(self, *args, **kwargs):  # pragma: no cover - compatibility shim
        return self


fake_pandas.DataFrame = _FakeDataFrame
fake_pandas.Series = lambda values: _FakeSeries(values)

sys.modules.setdefault("pandas", fake_pandas)

from reports import wandb_utils


class DummyRun:
    def __init__(self, *, summary, config=None):
        self.id = "run-1"
        self.name = "dummy"
        self.tags = []
        self.summary = summary
        self.config = config or {}
        self.group = None
        self.job_type = None
        self.url = "https://wandb.ai/dummy/run-1"


class StubApi:
    def __init__(self, runs):
        self._runs = runs

    def runs(self, project_path, *, filters=None, per_page=None):  # pragma: no cover - signature parity
        return self._runs


@pytest.fixture(autouse=True)
def patch_history(monkeypatch):
    monkeypatch.setattr(
        wandb_utils,
        "_load_history",
        lambda run, keys=None, **_: None,
    )


def test_fetch_runs_parses_json_summary(monkeypatch):
    run = DummyRun(summary="{\"accuracy\": 0.9}")
    monkeypatch.setattr(wandb_utils, "get_wandb_api", lambda **_: StubApi([run]))

    records = wandb_utils.fetch_runs(entity=None, project="proj")

    assert len(records) == 1
    assert records[0].summary == {"accuracy": 0.9}


def test_fetch_runs_handles_non_mapping_summary(monkeypatch):
    run = DummyRun(summary="not a mapping")
    monkeypatch.setattr(wandb_utils, "get_wandb_api", lambda **_: StubApi([run]))

    records = wandb_utils.fetch_runs(entity=None, project="proj")

    assert len(records) == 1
    assert records[0].summary == {}


def test_get_wandb_api_clamps_low_env_timeout(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "testing")
    monkeypatch.setenv("WANDB_HTTP_TIMEOUT", "5")

    class _StubModule:
        def __init__(self) -> None:
            self.calls = []

        def Api(self, timeout=None):  # noqa: N803 - mirror wandb.Api
            self.calls.append(timeout)
            return timeout

    stub = _StubModule()
    monkeypatch.setattr(wandb_utils, "maybe_init_wandb", lambda **_: stub)

    api = wandb_utils.get_wandb_api(project="proj")

    assert api == 30
    assert stub.calls == [30]
    assert os.environ["WANDB_HTTP_TIMEOUT"] == "30"


def test_get_wandb_api_falls_back_when_timeout_unsupported(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "testing")
    monkeypatch.delenv("WANDB_HTTP_TIMEOUT", raising=False)

    calls = []

    class _StubModule:
        def Api(self, timeout=None):  # noqa: N803 - mirror wandb.Api
            calls.append(timeout)
            if timeout is not None:
                raise TypeError("timeout unsupported")
            return "api"

    stub = _StubModule()
    monkeypatch.setattr(wandb_utils, "maybe_init_wandb", lambda **_: stub)

    api = wandb_utils.get_wandb_api(project="proj")

    assert api == "api"
    assert calls == [60, None]
    assert os.environ["WANDB_HTTP_TIMEOUT"] == "60"


def test_resolve_timeout_clamps_preferred(monkeypatch):
    monkeypatch.delenv("WANDB_HTTP_TIMEOUT", raising=False)

    resolved = wandb_utils.resolve_wandb_http_timeout(10)

    assert resolved == 30
    assert os.environ["WANDB_HTTP_TIMEOUT"] == "30"
