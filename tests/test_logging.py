import sys
import types
import logging
import os
import pytest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging import DummyWandb, maybe_init_wandb


def test_maybe_init_wandb_login_called(monkeypatch):
    calls = {}

    def login(*, key=None):
        calls['login'] = key

    def init(*args, **kwargs):
        calls['init'] = True

    fake = types.SimpleNamespace(login=login, init=init)
    monkeypatch.setitem(sys.modules, 'wandb', fake)

    result = maybe_init_wandb(True, api_key='token')

    assert result is fake
    assert calls['login'] == 'token'
    assert calls['init'] is True

def test_maybe_init_wandb_real():
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        pytest.skip("WANDB_API_KEY not set")
    wandb = maybe_init_wandb(True, api_key=api_key)  # no monkeypatch here
    run = wandb.init(project="m-jepa", name="m-jepa-unit-test", reinit=True)
    wandb.log({"ok": True})
    url = run.url
    wandb.finish()
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
