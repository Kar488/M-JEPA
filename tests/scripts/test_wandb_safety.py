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