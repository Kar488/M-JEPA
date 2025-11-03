from collections import Counter
import json
import sys
import types
from typing import Any

if "wandb" not in sys.modules:
    stub = types.ModuleType("wandb")
    stub.Api = lambda *args, **kwargs: None
    stub.run = None
    sys.modules["wandb"] = stub

import scripts.ci.recheck_topk_from_wandb as rc

np = rc.np


class DummyRun:
    def __init__(self, summary=None, history_rows=None, config=None, run_id="run"):
        self.summary = summary or {}
        self._history_rows = history_rows or []
        self.config = config or {}
        self.id = run_id

    def history(self, **kwargs):
        return [row.copy() for row in self._history_rows]


def test_config_and_numeric_helpers():
    class Wrapper:
        def __init__(self):
            self.value = {"inner": 3}
            self._json_dict = {"foo": "bar"}

        def to_dict(self):
            return {"a": 1}

    config = rc._coerce_config(Wrapper())
    assert config["a"] == 1
    assert rc._unwrap_config_value({"value": 5}) == 5
    nested = {"foo": {"bar": {"baz": 10}}}
    assert rc._lookup_nested(nested, "foo/bar/baz") == 10
    assert rc._lookup_nested(nested, "foo.bar.baz") == 10
    assert rc._coerce_numeric({"max": "1.5"}) == 1.5
    assert rc._coerce_numeric([None, "2.0"]) == 2.0
    assert rc._norm_key("Foo-Bar") == "foo_bar"
    assert rc._num_close(1.0000001, 1.0)
    assert rc._norm_cfg({"Foo": {"value": 1}})["foo"] == 1
    assert set(rc._metric_candidates("val/loss")) == {"val/loss", "val.loss"}


def test_override_devices_helpers(monkeypatch):
    assert rc._coerce_positive_int("5", "devices") == 5
    assert rc._coerce_positive_int("0", "devices") is None
    assert rc._coerce_positive_int("abc", "devices") is None

    assert rc._extract_positive_int(4) == 4
    assert rc._extract_positive_int("7") == 7
    assert rc._extract_positive_int("3.0") == 3
    assert rc._extract_positive_int("three") is None

    cfg = {"devices": 1, "device": "cuda:0"}
    rc._apply_config_overrides(cfg, override_devices=3)
    assert cfg["devices"] == 3
    assert "device" not in cfg

    cfg2: dict[str, Any] = {}
    rc._apply_config_overrides(cfg2, override_devices=None)
    assert cfg2 == {}


def test_history_latest_and_metric_of():
    run = DummyRun(
        summary={"metrics": {"val": {"value": 0.42}}},
        history_rows=[{"val": 0.5}, {"val": 0.6}],
    )
    value, key = rc._history_latest(run, ["val"])
    assert value == 0.6 and key == "val"
    assert rc.metric_of(run, "metrics/val") == 0.42

    # Fallback to history when summary missing
    run2 = DummyRun(summary={}, history_rows=[{"metrics": {"loss": 0.8}}])
    assert rc.metric_of(run2, "metrics/loss") == 0.8


def test_pick_topk_orders_runs(monkeypatch):
    class DummySweep:
        def __init__(self, runs):
            self.runs = runs

    run_good = DummyRun(summary={"val_rmse": 0.1}, config={"training_method": "jepa"}, run_id="good")
    run_bad = DummyRun(summary={}, history_rows=[], config={}, run_id="bad")
    sweep = DummySweep([run_good, run_bad])

    class DummyApi:
        def __init__(self):
            self.calls = 0

        def sweep(self, path):
            self.calls += 1
            return sweep

    api = DummyApi()
    ranked, diagnostics = rc.pick_topk(api, "entity/project/sweep", metric="val_rmse", maximize=False, k=1, attempts=1)
    assert ranked[0][0] is run_good
    assert diagnostics["total_runs"] == 2
    assert diagnostics["missing"] == ["bad"]
    assert diagnostics["method_counts"] == Counter({"jepa": 1, "unknown": 1})


def test_expected_name_group_mapping():
    assert rc._expected_group(3) == "recheck_cfg3"
    assert rc._expected_run_name(3, 1001) == "recheck_cfg3_seed1001"


def test_gpu_parallel_helpers(monkeypatch):
    assert rc._split_gpu_ids(["0", "1", "2"], 2) == ["0,1", "2"]
    assert rc._split_gpu_ids([], 2) == ["", ""]

    monkeypatch.setenv("PHASE2_RECHECK_AGENT_COUNT", "3")
    assert rc._resolve_agent_count(["0", "1"]) == 2
    monkeypatch.delenv("PHASE2_RECHECK_AGENT_COUNT", raising=False)

    monkeypatch.setenv("PHASE2_AGENT_COUNT", "4")
    assert rc._resolve_agent_count(["0", "1", "2"]) == 3
    monkeypatch.delenv("PHASE2_AGENT_COUNT", raising=False)

    assert rc._resolve_agent_count([], override_devices=2) == 1
    assert rc._resolve_agent_count(["0", "1"], override_devices=2) == 1

    workers, masks = rc._compute_worker_gpu_masks(["0", "1"], 2, 2)
    assert workers == 1
    assert masks == ["0,1"]

    workers, masks = rc._compute_worker_gpu_masks(["0", "1", "2", "3"], 4, 2)
    assert workers == 2
    assert masks == ["0,1", "2,3"]

    workers, masks = rc._compute_worker_gpu_masks(["0", "1", "2"], 2, 1)
    assert workers == 2
    assert masks == ["0,1", "2"]


def test_run_once_writes_log(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    started = {}

    class FakeProc:
        def __init__(self):
            self.pid = 123
            self._calls = 0

        def poll(self):
            self._calls += 1
            return 0 if self._calls > 1 else None

    def fake_popen(args, stdout, stderr, env):
        started["args"] = args
        started["env"] = env
        return FakeProc()

    times = iter([0.0, 30.0, 65.0, 120.0])

    monkeypatch.setattr(rc.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(rc.time, "time", lambda: next(times))
    monkeypatch.setattr(rc.time, "sleep", lambda *_: None)

    rc.run_once(
        mm="micromamba",
        program="train.py",
        subcmd="finetune",
        cfg={"training_method": "jepa", "hidden_dim": 64},
        seed=7,
        unlabeled="ul",  # type: ignore[arg-type]
        labeled="lb",  # type: ignore[arg-type]
        log_dir=str(log_dir),
        project="proj",
        group="grp",
        config_idx=2,
        exp_id="exp123",
        device_mask="1",
    )

    log_file = log_dir / "recheck_jepa_seed7.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "--hidden-dim" in content
    assert started["args"][0] == "micromamba"
    assert started["env"]["WANDB_NAME"] == "recheck_cfg2_seed7"
    assert started["env"]["WANDB_RUN_GROUP"] == "recheck_cfg2"
    assert started["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_run_once_retries_transient(tmp_path, monkeypatch, capsys):
    log_dir = tmp_path / "logs"
    started = {"calls": 0}

    class FakeProc:
        def __init__(self, exit_code):
            self.pid = 200 + started["calls"]
            self._calls = 0
            self._exit = exit_code

        def poll(self):
            self._calls += 1
            return self._exit if self._calls > 1 else None

    exit_codes = [255, 0]

    def fake_popen(args, stdout, stderr, env):
        started["calls"] += 1
        started.setdefault("env", env)
        return FakeProc(exit_codes.pop(0))

    times = iter([0.0, 61.0, 0.0, 61.0])

    monkeypatch.setenv("RECHECK_RUN_RETRIES", "2")
    monkeypatch.setenv("RECHECK_TRANSIENT_RCS", "255")
    monkeypatch.setenv("RECHECK_RUN_RETRY_DELAY", "0")
    monkeypatch.setattr(rc.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(rc.time, "time", lambda: next(times))
    monkeypatch.setattr(rc.time, "sleep", lambda *_: None)

    rc.run_once(
        mm="micromamba",
        program="train.py",
        subcmd="finetune",
        cfg={"training_method": "jepa", "hidden_dim": 64},
        seed=7,
        unlabeled="ul",  # type: ignore[arg-type]
        labeled="lb",  # type: ignore[arg-type]
        log_dir=str(log_dir),
        project="proj",
        group="grp",
        config_idx=2,
        exp_id="exp123",
    )

    assert started["calls"] == 2
    assert not exit_codes

    log_file = log_dir / "recheck_jepa_seed7.log"
    content = log_file.read_text()
    assert "retry attempt 2" in content
    assert "exit code 255" in content

    captured = capsys.readouterr()
    assert "transient exit code 255" in captured.out


def test_collect_seed_metrics_prefers_wandb(monkeypatch):
    config_idx = 1
    seeds = [1000]
    cfg = {"training_method": "jepa", "gnn_type": "gine", "hidden_dim": 64, "num_layers": 3, "contiguity": 0}

    class DummyRunObj:
        def __init__(self):
            self.config = {**cfg, "seed": seeds[0]}
            self.summary = {"val_rmse": 0.25}
            self.group = rc._expected_group(config_idx)
            self.name = rc._expected_run_name(config_idx, seeds[0])

    class DummyApi:
        def runs(self, project_path, filters):
            assert filters["group"] == rc._expected_group(config_idx)
            return [DummyRunObj()]

    monkeypatch.setattr(rc.time, "sleep", lambda *_: None)
    monkeypatch.setattr(rc, "_best_effort_wandb_sync", lambda: None)
    metrics = rc._collect_seed_metrics(
        DummyApi(),
        "entity/project",
        cfg,
        seeds,
        "val_rmse",
        config_idx,
        "exp",
        attempts=2,
        delay=0.0,
    )
    assert metrics == {seeds[0]: 0.25}


def test_collect_seed_metrics_uses_local_fallback(monkeypatch, tmp_path, capsys):
    config_idx = 2
    seeds = [1001]
    cfg = {"training_method": "jepa", "gnn_type": "gine", "hidden_dim": 64, "num_layers": 3, "contiguity": 0}

    class DummyApi:
        def runs(self, *_a, **_k):
            return []

    artifact_path = tmp_path / "cfg2_seed1001.json"
    artifact_path.write_text(json.dumps({"val_rmse": 0.34, "best_step": 7}))

    monkeypatch.setattr(rc, "_local_result_path", lambda _exp, _cfg, _seed: artifact_path)
    monkeypatch.setattr(rc.time, "sleep", lambda *_: None)
    monkeypatch.setattr(rc, "_best_effort_wandb_sync", lambda: None)

    metrics = rc._collect_seed_metrics(
        DummyApi(),
        "entity/project",
        cfg,
        seeds,
        "val_rmse",
        config_idx,
        "exp",
        attempts=1,
        delay=0.0,
    )
    captured = capsys.readouterr().out
    assert "using local fallback" in captured
    assert metrics == {seeds[0]: 0.34}


def test_collect_seed_metrics_handles_missing(monkeypatch, tmp_path, capsys):
    config_idx = 3
    seeds = [1002]
    cfg = {"training_method": "jepa", "gnn_type": "gine", "hidden_dim": 64, "num_layers": 3, "contiguity": 0}

    class DummyApi:
        def runs(self, *_a, **_k):
            return []

    # ensure no local file exists
    monkeypatch.setattr(rc, "_local_result_path", lambda _exp, _cfg, _seed: tmp_path / "missing.json")
    sleep_calls = []
    monkeypatch.setattr(rc.time, "sleep", lambda d: sleep_calls.append(d))
    monkeypatch.setattr(rc, "_best_effort_wandb_sync", lambda: None)

    metrics = rc._collect_seed_metrics(
        DummyApi(),
        "entity/project",
        cfg,
        seeds,
        "val_rmse",
        config_idx,
        "exp",
        attempts=1,
        delay=0.0,
    )
    assert metrics == {}
    assert sleep_calls == []

    rc._warn_missing_metric("val_rmse", seeds[0])
    captured = capsys.readouterr().out
    assert "missing metric 'val_rmse'" in captured
    assert "skipping" in captured


def test_ci95_monkeypatched(monkeypatch):
    samples = np.array([1.0, 2.0, 3.0])
    monkeypatch.setattr(np.random, "choice", lambda xs, size, replace: np.tile(xs.mean(), size))
    lo, hi = rc.ci95(samples)
    assert lo == hi == samples.mean()
