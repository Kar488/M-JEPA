from collections import Counter

import numpy as np

import scripts.ci.recheck_topk_from_wandb as rc


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
    )

    log_file = log_dir / "recheck_jepa_seed7.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "--hidden-dim" in content
    assert started["args"][0] == "micromamba"


def test_ci95_monkeypatched(monkeypatch):
    samples = np.array([1.0, 2.0, 3.0])
    monkeypatch.setattr(np.random, "choice", lambda xs, size, replace: np.tile(xs.mean(), size))
    lo, hi = rc.ci95(samples)
    assert lo == hi == samples.mean()

