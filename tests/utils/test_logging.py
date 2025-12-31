import sys
import types
import logging
import os
import pytest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging import DummyWandb, maybe_init_wandb


def _install_fake_wandb(monkeypatch):
    calls = []

    class FakeRun:
        def __init__(self, rid):
            self.id = rid
            self.name = None
            self.finished = False
            self.saved = False

        def finish(self):
            self.finished = True

        def save(self):
            self.saved = True

    def login(*, key=None):
        calls.append(("login", {"key": key}))

    def init(**kwargs):
        run = FakeRun(kwargs["id"])
        fake.run = run
        calls.append(("init", kwargs))
        return run

    fake = types.SimpleNamespace(login=login, init=init, Settings=lambda **kwargs: {"settings": kwargs}, run=None)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake, calls


def test_maybe_init_wandb_login_called(monkeypatch):
    calls = {}

    def login(*, key=None):
        calls['login'] = key

    def init(*args, **kwargs):
        calls['init'] = kwargs

    fake = types.SimpleNamespace(login=login, init=init)
    monkeypatch.setitem(sys.modules, 'wandb', fake)

    monkeypatch.setenv('WANDB_ENTITY', 'env-entity')

    result = maybe_init_wandb(True, api_key='token', entity='explicit-entity')

    assert result is fake
    assert calls['login'] == 'token'
    assert 'init' in calls
    assert calls['init']['entity'] == 'explicit-entity'

def test_maybe_init_wandb_real():
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        pytest.skip("WANDB_API_KEY not set")
    wandb = maybe_init_wandb(True, api_key=api_key)  # no monkeypatch here
    assert wandb.run is not None, "maybe_init_wandb did not create a run"
    wandb.log({"ok": True})
    url = wandb.run.url
    assert "wandb.ai" in url

def test_maybe_init_wandb_warns_on_failure(monkeypatch, caplog):
    calls = {}

    def login(*, key=None):
        calls['login'] = key

    def init(*args, **kwargs):
        raise RuntimeError('boom')

    fake = types.SimpleNamespace(login=login, init=init)
    monkeypatch.setitem(sys.modules, 'wandb', fake)

    with caplog.at_level(logging.WARNING):
        result = maybe_init_wandb(True, api_key='secret')

    assert isinstance(result, DummyWandb)
    assert calls['login'] == 'secret'
    assert 'Failed to initialise wandb: boom' in caplog.text


def test_maybe_init_wandb_creates_fresh_run_when_existing_run_present(monkeypatch):
    fake, calls = _install_fake_wandb(monkeypatch)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)

    maybe_init_wandb(True, project="proj", job_type="bench")
    first_run = fake.run
    first_kwargs = calls[-1][1]
    assert first_kwargs["resume"] == "never"
    assert first_run.finished is False

    maybe_init_wandb(True, project="proj", job_type="tox21")
    second_run = fake.run
    second_kwargs = calls[-1][1]

    assert first_run.finished is True
    assert second_run is not first_run
    assert second_kwargs["id"] != first_kwargs["id"]
    assert second_kwargs["job_type"] == "tox21"
    assert second_kwargs["resume"] == "never"


def test_maybe_init_wandb_stage_payloads_are_distinct(monkeypatch):
    fake, calls = _install_fake_wandb(monkeypatch)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)

    monkeypatch.setenv("WANDB_RUN_GROUP", "exp-bench")
    monkeypatch.setenv("WANDB_NAME", "evaluate-val")
    maybe_init_wandb(True, config={"stage": "bench"})
    bench_kwargs = calls[-1][1]
    bench_run = fake.run

    assert bench_kwargs["name"] == "evaluate-val"
    assert bench_kwargs["group"] == "exp-bench"
    assert bench_kwargs["resume"] == "never"

    monkeypatch.setenv("WANDB_RUN_GROUP", "exp-tox21")
    monkeypatch.setenv("WANDB_NAME", "tox21-baseline")
    monkeypatch.setenv("WANDB_JOB_TYPE", "tox21")
    maybe_init_wandb(True, config={"stage": "tox21"})
    tox_kwargs = calls[-1][1]

    assert bench_run.finished is True
    assert tox_kwargs["name"] == "tox21-baseline"
    assert tox_kwargs["group"] == "exp-tox21"
    assert tox_kwargs["job_type"] == "tox21"
    assert tox_kwargs["id"] != bench_kwargs["id"]
