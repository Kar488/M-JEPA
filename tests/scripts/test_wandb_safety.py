import importlib
import sys
import types


def test_wb_summary_update_logs_val_rmse(monkeypatch):
    logs = {}

    class DummyRun:
        def __init__(self):
            self.summary = {}

    dummy_wandb = types.SimpleNamespace(
        run=DummyRun(), log=lambda data: logs.update(data)
    )
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    ws.wb_summary_update({"rmse": 1.23})

    assert dummy_wandb.run.summary["val_rmse"] == 1.23
    assert logs["val_rmse"] == 1.23

def test_wb_summary_update_no_run(monkeypatch):
    dummy_wandb = types.SimpleNamespace(run=None)
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)
    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    # Should safely return without errors
    ws.wb_summary_update({"val_rmse": 0.5})


def test_wb_get_or_init_respects_disable(monkeypatch):
    calls = {"init": 0}
    def init(**kwargs):
        calls["init"] += 1
        return types.SimpleNamespace(id="run")
    dummy_wandb = types.SimpleNamespace(run=None, init=init)
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    assert ws.wb_get_or_init(args=None) is None
    assert calls["init"] == 0


def test_wb_finish_safely(monkeypatch):
    calls = {"finish": 0}
    def finish():
        calls["finish"] += 1
        raise RuntimeError("boom")
    dummy_wandb = types.SimpleNamespace(finish=finish)
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)
    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    ws.wb_finish_safely()
    assert calls["finish"] == 1
