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


def test_wb_summary_update_uses_api_when_run_missing(monkeypatch):
    summary = {}

    class DummyApiRun:
        def __init__(self):
            self.summary = summary

    class DummyApi:
        def __init__(self):
            self.calls = []

        def run(self, path):
            self.calls.append(path)
            return DummyApiRun()

    dummy_api = DummyApi()
    dummy_wandb = types.SimpleNamespace(run=None, Api=lambda: dummy_api)
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)
    monkeypatch.setenv("WANDB_RUN_ID", "rid")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("WANDB_ENTITY", "ent")

    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    ws.wb_summary_update({"val_rmse": 0.5})

    assert summary["val_rmse"] == 0.5
    assert dummy_api.calls == ["ent/proj/rid"]


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


def test_wb_get_or_init_existing_run(monkeypatch):
    dummy_run = types.SimpleNamespace(id="run1")
    dummy_wandb = types.SimpleNamespace(run=dummy_run)
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)
    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    assert ws.wb_get_or_init(args=None) is dummy_run


def test_wb_get_or_init_uses_helper(monkeypatch):
    dummy_wandb = types.SimpleNamespace(run=None)
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)
    calls = {"count": 0}

    def helper(**kwargs):
        calls["count"] += 1
        dummy_wandb.run = types.SimpleNamespace(id="run2")
        return dummy_wandb.run

    monkeypatch.setitem(
        sys.modules, "utils.logging", types.SimpleNamespace(maybe_init_wandb=helper)
    )
    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    run = ws.wb_get_or_init(args=types.SimpleNamespace(foo=1))
    assert run is dummy_wandb.run
    assert calls["count"] == 1


def test_wb_summary_update_aliases(monkeypatch, tmp_path):
    class DummyRun:
        def __init__(self):
            self.summary = {}

    logs = {}
    dummy_wandb = types.SimpleNamespace(
        run=DummyRun(), log=lambda data: logs.update(data)
    )
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    ws.wb_summary_update({"mae_mean": 0.1, "auc": 0.9})
    assert dummy_wandb.run.summary["val_mae"] == 0.1
    assert dummy_wandb.run.summary["val_auc"] == 0.9


def test_wb_summary_update_accepts_mean_aliases(monkeypatch):
    class DummyRun:
        def __init__(self):
            self.summary = {}

    logs = {}
    dummy_wandb = types.SimpleNamespace(
        run=DummyRun(), log=lambda data: logs.update(data)
    )
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    ws.wb_summary_update({"roc_auc_mean": 0.77, "best_step_mean": 4})

    assert dummy_wandb.run.summary["val_auc"] == 0.77
    assert dummy_wandb.run.summary["best_step"] == 4
    assert logs["val_auc"] == 0.77


def test_wb_summary_update_fallback_on_update_error(monkeypatch):
    class DummySummary(dict):
        def update(self, *a, **kwargs):
            raise RuntimeError("boom")

    class DummyRun:
        def __init__(self):
            self.summary = DummySummary()

    dummy_wandb = types.SimpleNamespace(run=DummyRun(), log=lambda data: None)
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    ws = importlib.reload(importlib.import_module("scripts.wandb_safety"))
    ws.wb_summary_update({"pair_id": "abc123"})

    assert dummy_wandb.run.summary["pair_id"] == "abc123"
