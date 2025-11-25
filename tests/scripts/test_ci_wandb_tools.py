import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests

yaml = pytest.importorskip("yaml")

# Provide a lightweight stub so modules can import wandb during test collection.
sys.modules.setdefault("wandb", types.SimpleNamespace(Api=lambda: None))

from scripts.ci import export_best_from_wandb as eb
from scripts.ci import paired_effect_from_wandb as pe
from scripts.ci import phase1_decision as pd
from scripts.ci import recheck_topk_from_wandb as rc


class FakeRun:
    def __init__(
        self,
        name,
        config,
        summary,
        run_id="runid",
        history=None,
        summary_metrics=None,
        raw_attrs=None,
    ):
        self.name = name
        self.config = config
        self.summary = summary
        self.id = run_id
        self._history = history
        self.summary_metrics = summary_metrics if summary_metrics is not None else {}
        self._attrs = raw_attrs if raw_attrs is not None else {}

    def history(self, **kwargs):  # pragma: no cover - simple passthrough
        if self._history is None:
            return []
        if callable(self._history):
            return self._history(**kwargs)
        return list(self._history)


class FakeSweep:
    def __init__(self, runs, config=None):
        self.runs = runs
        self.config = config or {}


def _build_fake_api(runs):
    class _Api:
        def sweep(self, sweep_id):  # pragma: no cover - trivial wrapper
            return FakeSweep(runs)

    return _Api()


def test_metric_considers_summary_metrics_when_missing_in_summary():
    run = FakeRun(
        "with_metadata",
        config={},
        summary={"_runtime": 10, "_timestamp": 1234},
        summary_metrics={"val_auc": 0.78},
    )

    assert eb.metric(run, "val_auc") == 0.78


def test_metric_supports_nested_keys_from_summary_metrics():
    run = FakeRun(
        "nested_metrics",
        config={},
        summary={},
        summary_metrics={"validation": {"rmse": 0.43}},
    )

    assert eb.metric(run, "validation.rmse") == 0.43


def test_pick_primary_metric_handles_nested_aliases():
    args = types.SimpleNamespace(
        reg_primary="val_rmse",
        clf_primary="val_auc",
        reg_tb1=None,
        clf_tb1=None,
        clf_tb2=None,
    )

    run = FakeRun("nested", config={}, summary={"metrics": {"rmse_mean": 0.37}})

    primary, maximize = eb.pick_primary_metric([run], "regression", args)

    assert primary == "metrics/rmse_mean"
    assert maximize is False


def test_paired_effect_accepts_serialized_config(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_ENTITY", "ent")

    serialized_runs = [
        FakeRun(
            "jepa_serialized",
            json.dumps(
                {
                    "training_method": "jepa",
                    "pair_id": "pair-json",
                    "prediction_target_type": "regression",
                }
            ),
            json.dumps({"val_rmse": 0.4}),
            run_id="run-jepa",
        ),
        FakeRun(
            "contrastive_serialized",
            json.dumps(
                {
                    "training_method": "contrastive",
                    "pair_id": "pair-json",
                    "prediction_target_type": "regression",
                }
            ),
            json.dumps({"val_rmse": 0.6}),
            run_id="run-contrastive",
        ),
    ]

    class FakeApi:
        def runs(self, path, filters=None):
            return serialized_runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))

    out = tmp_path / "pe_serialized.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
        ],
    )

    pe.main()
    result = json.loads(out.read_text())
    assert result["winner"] == "jepa"
    assert result["pairs"] == 1


def test_paired_effect_unwraps_sweep_config_wrappers(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_ENTITY", "ent")

    wrapped_runs = [
        FakeRun(
            "jepa_wrapped_cfg",
            {
                "training_method": {"value": "jepa", "_type": "string"},
                "pair_id": {"value": "cfg-wrap"},
                "seed": {"value": 11},
                "task_type": {"value": "regression"},
            },
            {"val_rmse": {"value": 0.42}},
        ),
        FakeRun(
            "contrastive_wrapped_cfg",
            {
                "training_method": {"value": "contrastive"},
                "pair_id": {"value": "cfg-wrap"},
                "seed": {"value": 11},
                "task_type": {"value": "regression"},
            },
            {"val_rmse": {"value": 0.58}},
        ),
    ]

    class WrappedCfgApi:
        def runs(self, path, filters=None):
            return wrapped_runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: WrappedCfgApi()))

    out = tmp_path / "pe_cfg_wrapped.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())
    assert payload["winner"] == "jepa"
    assert payload["pairs"] == 1


def test_export_best_nonempty_json(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DIR", str(tmp_path))
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    monkeypatch.setenv("GRID_DIR", str(grid_dir))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("METHOD_WINNER", "contrastive")

    best_run = FakeRun(
        "best",
        config={
            "training_method": {"value": "jepa"},
            "gnn_type": {"value": "gine"},
            "hidden_dim": {"value": "384"},
            "num_layers": {"value": 4},
            "cache-dir": {"value": "/tmp/cache"},
            "misc": {"value": {"nested": 1}},
        },
        summary={"val_rmse": 0.42},
    )
    other_run = FakeRun(
        "worse",
        config={"training_method": "jepa"},
        summary={"val_rmse": 0.9},
    )

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: _build_fake_api([best_run, other_run])))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_path = grid_dir / "best_grid_config.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--sweep-id",
            "ent/proj/sw",
            "--out",
            str(out_path),
        ],
    )

    eb.main()

    payload = json.loads(out_path.read_text())
    for key in ("training_method", "gnn_type", "hidden_dim", "num_layers"):
        assert key in payload
    assert payload["training_method"] == "jepa"
    assert payload["gnn_type"] == "gine"
    assert payload["hidden_dim"] == 384
    assert payload["num_layers"] == 4

    metrics_path = grid_dir / "phase1_runs.csv"
    winner_path = grid_dir / "phase2_winner_config.csv"
    assert metrics_path.is_file()
    assert winner_path.is_file()
    header = metrics_path.read_text().splitlines()[0]
    assert "config.training_method" in header


def test_phase2_yaml_and_best_paths(monkeypatch, tmp_path):
    app_dir = tmp_path
    grid_dir = tmp_path / "grid"
    sweeps_dir = tmp_path / "sweeps"
    sweeps_dir.mkdir()
    grid_dir.mkdir()
    (sweeps_dir / "grid_sweep_phase2.yaml").write_text("parameters:\n  hidden_dim:\n    values: [256, 384]\n")

    monkeypatch.setenv("APP_DIR", str(app_dir))
    monkeypatch.setenv("GRID_DIR", str(grid_dir))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("METHOD_WINNER", "contrastive")

    runs = [
        FakeRun(
            "best",
            config={
                "training_method": {"value": "contrastive"},
                "gnn_type": {"value": "gine"},
                "hidden_dim": {"value": 512},
                "num_layers": {"value": 5},
                "learning_rate": {"value": 0.0003},
                "mask_ratio": {"value": 0.2},
            },
            summary={"val_auc": 0.91},
        ),
        FakeRun(
            "runner_up",
            config={
                "training_method": {"value": "contrastive"},
                "gnn_type": {"value": "gine"},
                "hidden_dim": {"value": 384},
                "num_layers": {"value": 4},
                "learning_rate": {"value": 0.0005},
                "mask_ratio": {"value": 0.15},
            },
            summary={"val_auc": 0.9},
        ),
    ]

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: _build_fake_api(runs)))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_path = grid_dir / "best_grid_config.json"
    phase2_path = grid_dir / "grid_sweep_phase2.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--sweep-id",
            "ent/proj/sw",
            "--emit-bounds",
            "--out",
            str(out_path),
            "--phase2-yaml",
            str(phase2_path),
        ],
    )

    eb.main()

    payload = json.loads(out_path.read_text())
    for key in ("training_method", "gnn_type", "hidden_dim", "num_layers"):
        assert key in payload

    phase2 = yaml.safe_load(phase2_path.read_text())
    assert phase2["parameters"]["training_method"]["value"] == "contrastive"
    assert phase2_path.is_file()

    metrics_path = grid_dir / "phase1_runs.csv"
    winner_path = grid_dir / "phase2_winner_config.csv"
    assert metrics_path.is_file()
    assert winner_path.is_file()


def test_export_best_errors_on_empty_sweep(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DIR", str(tmp_path))
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    monkeypatch.setenv("GRID_DIR", str(grid_dir))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: _build_fake_api([])))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_path = grid_dir / "best_grid_config.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--sweep-id",
            "ent/proj/sw",
            "--out",
            str(out_path),
        ],
    )

    with pytest.raises(RuntimeError, match="No runs found"):
        eb.main()

    assert not out_path.exists()


def test_paired_effect_uses_summary_pair_id(monkeypatch, tmp_path):
    """Runs missing pair_id in config should fall back to the summary payload."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")

    runs = [
        FakeRun(
            "jepa_summary_pair",
            {"training_method": "jepa"},
            {"pair_id": "summary-pair", "val_rmse": 0.4},
        ),
        FakeRun(
            "contrastive_summary_pair",
            {"training_method": "contrastive"},
            {"pair_id": "summary-pair", "val_rmse": 0.6},
        ),
    ]

    class Api:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: Api()))

    out = tmp_path / "pe_summary_pair.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())

    assert payload["pairs"] == 1
    assert payload["winner"] == "jepa"


def test_paired_effect_uses_summary_metrics_fallback(monkeypatch, tmp_path):
    """Ensure runs exposing summary metrics outside ``run.summary`` are parsed."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")

    class EmptySummary:
        _json_dict: Dict[str, Any] = {}

    runs = [
        FakeRun(
            "jepa_fallback",
            {"training_method": "jepa", "pair_id": "metrics"},
            EmptySummary(),
            summary_metrics={"val_rmse": 0.41},
            raw_attrs={"summaryMetrics": {"val_rmse": 0.41}},
        ),
        FakeRun(
            "contrastive_fallback",
            {"training_method": "contrastive", "pair_id": "metrics"},
            EmptySummary(),
            summary_metrics={"val_rmse": 0.55},
            raw_attrs={"summaryMetrics": {"val_rmse": 0.55}},
        ),
    ]

    class Api:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: Api()))

    out = tmp_path / "pe_summary_metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())

    assert payload["pairs"] == 1
    assert payload["winner"] == "jepa"


def test_paired_effect_winner_and_no_pairs(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WANDB_ENTITY", "ent")

    runs = [
        FakeRun("j1", {"training_method": "jepa", "pair_id": 1}, {"val_rmse": 0.5}),
        FakeRun("c1", {"training_method": "contrastive", "pair_id": 1}, {"val_rmse": 0.7}),
        FakeRun("j2", {"training_method": "jepa", "pair_id": 2}, {"val_rmse": 0.4}),
        FakeRun("c2", {"training_method": "contrastive", "pair_id": 2}, {"val_rmse": 0.6}),
    ]

    class FakeApi:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))

    out = tmp_path / "pe.json"
    monkeypatch.setattr(sys, "argv", ["pe", "--project", "proj", "--group", "grp", "--out", str(out)])
    pe.main()
    data = json.loads(out.read_text())
    assert data["winner"] == "jepa"

    # now simulate lack of pairs
    class EmptyApi:
        def runs(self, path, filters=None):
            return []

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: EmptyApi()))
    out2 = tmp_path / "pe2.json"
    # reset capture between invocations
    # reset capture between invocations
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["pe", "--project", "proj", "--group", "grp", "--out", str(out2)])
    pe.main()  # graceful exit, no exception
    # In the empty case, the tool no longer writes an output file (unless --strict)
    assert not out2.exists()

    # When runs exist for both methods but their pair_ids do not overlap, the
    # paired-effect utility should fall back to a global aggregate delta rather
    # than exiting with an error (even when --strict is provided).
    unpaired_runs = [
        FakeRun("j_only1", {"training_method": "jepa", "pair_id": "a"}, {"val_rmse": 0.48}),
        FakeRun("j_only2", {"training_method": "jepa", "pair_id": "b"}, {"val_rmse": 0.50}),
        FakeRun("c_only", {"training_method": "contrastive", "pair_id": "c"}, {"val_rmse": 0.55}),
    ]

    class UnpairedApi:
        def runs(self, path, filters=None):
            return unpaired_runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: UnpairedApi()))
    out3 = tmp_path / "pe3.json"
    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out3),
            "--strict",
        ],
    )
    pe.main()
    data3 = json.loads(out3.read_text())
    assert data3["pairs"] == 1
    assert data3["pairs_used"] == 0
    assert data3["winner"] == "jepa"


def test_paired_effect_retries_when_configs_flaky(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("PE_FETCH_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("PE_FETCH_RETRY_DELAY", "0")
    monkeypatch.setenv("PE_RUN_CONFIG_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("PE_RUN_CONFIG_RETRY_DELAY", "0")
    monkeypatch.setattr(pe.time, "sleep", lambda _: None)

    class FlakyRun:
        def __init__(
            self,
            name: str,
            config: Dict[str, Any],
            summary: Dict[str, Any],
            failures: int,
            run_id: str,
        ) -> None:
            self.name = name
            self._config = config
            self.summary = summary
            self.summary_metrics: Dict[str, Any] = {}
            self._history = None
            self._attrs = {}
            self._failures = failures
            self.id = run_id

        @property
        def config(self):  # pragma: no cover - exercised via paired_effect
            if self._failures > 0:
                self._failures -= 1
                raise requests.exceptions.RequestException("temporary config fetch failure")
            return self._config

        @config.setter
        def config(self, value):
            self._config = value

        def history(self, **kwargs):  # pragma: no cover - passthrough
            return []

    flaky_run = FlakyRun(
        "flaky_jepa",
        {"training_method": "jepa", "pair_id": "pid"},
        {"val_rmse": 0.5},
        failures=3,
        run_id="flaky",
    )

    stable_runs = [
        FakeRun("jepa", {"training_method": "jepa", "pair_id": "pid"}, {"val_rmse": 0.4}),
        FakeRun("contrastive", {"training_method": "contrastive", "pair_id": "pid"}, {"val_rmse": 0.6}),
    ]

    class FlakyApi:
        def __init__(self):
            self.calls = 0

        def runs(self, path, filters=None):  # pragma: no cover - passthrough
            self.calls += 1
            if self.calls == 1:
                return [flaky_run, stable_runs[1]]
            return stable_runs

    monkeypatch.setattr(
        pe,
        "wandb",
        types.SimpleNamespace(
            Api=lambda: FlakyApi(),
            errors=types.SimpleNamespace(CommError=Exception, AuthenticationError=Exception),
        ),
    )

    out = tmp_path / "pe_flaky.json"
    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
            "--strict",
        ],
    )

    pe.main()
    output = capsys.readouterr().out
    assert "encountered 1 run config error" in output
    data = json.loads(out.read_text())
    assert data["winner"] == "jepa"


def test_paired_effect_retries_without_group_when_empty(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("PE_FETCH_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("PE_FETCH_RETRY_DELAY", "0")
    monkeypatch.setattr(pe.time, "sleep", lambda _: None)

    runs = [
        FakeRun("j1", {"training_method": "jepa", "pair_id": 1}, {"val_rmse": 0.5}),
        FakeRun(
            "c1",
            {"training_method": "contrastive", "pair_id": 1},
            {"val_rmse": 0.7},
        ),
    ]

    class FilteredApi:
        def runs(self, path, filters=None):
            if filters and filters.get("group"):
                return []
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: FilteredApi()))

    out = tmp_path / "pe_group_fallback.json"
    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--sweep",
            "sw1",
            "--out",
            str(out),
            "--strict",
        ],
    )
    pe.main()
    payload = json.loads(out.read_text())
    assert payload["pairs"] == 1
    assert payload["winner"] == "jepa"


def test_paired_effect_missing_threshold_keys(monkeypatch, tmp_path):
    """Runs lacking the threshold config keys should not be discarded entirely."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")

    runs = [
        FakeRun("jepa", {"training_method": "jepa", "pair_id": "pid"}, {"val_rmse": 0.5}),
        FakeRun("contrastive", {"training_method": "contrastive", "pair_id": "pid"}, {"val_rmse": 0.7}),
    ]

    class Api:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: Api()))

    out = tmp_path / "pe_missing_thresholds.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
            "--strict",
            "--min_pretrain_epochs",
            "5",
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())

    assert payload["pairs"] == 1
    assert payload["winner"] == "jepa"


def test_paired_effect_ignores_disabled_thresholds(monkeypatch, tmp_path):
    """Optional thresholds (default None) must not trigger comparison failures."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")

    runs = [
        FakeRun("jepa", {"training_method": "jepa", "pair_id": "pid"}, {"val_rmse": 0.4}),
        FakeRun(
            "contrastive",
            {"training_method": "contrastive", "pair_id": "pid", "max_pretrain_batches": 3},
            {"val_rmse": 0.6},
        ),
    ]

    class Api:
        def runs(self, path, filters=None):  # pragma: no cover - simple stub
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: Api()))

    out = tmp_path / "pe_optional_thresholds.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
            "--strict",
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())

    assert payload["pairs"] == 1
    assert payload["winner"] == "jepa"


def test_paired_effect_handles_wandb_summary_wrappers(monkeypatch, tmp_path):
    """Some wandb summary scalars are wrapped in {'value': x} containers."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")

    wrapped_runs = [
        FakeRun(
            "jepa_wrapped",
            {"training_method": "jepa", "pair_id": "wrap"},
            {"val_rmse": {"value": 0.41}},
        ),
        FakeRun(
            "contrastive_wrapped",
            {"training_method": "contrastive", "pair_id": "wrap"},
            {"val_rmse": {"value": 0.55}},
        ),
    ]

    class WrappedApi:
        def runs(self, path, filters=None):
            return wrapped_runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: WrappedApi()))

    out = tmp_path / "pe_wrapped.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
        ],
    )

    pe.main()
    result = json.loads(out.read_text())
    assert result["winner"] == "jepa"
    assert result["pairs"] == 1


def test_paired_effect_reads_history_when_summary_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_ENTITY", "ent")

    runs = [
        FakeRun(
            "jepa_hist",
            {"training_method": "jepa", "pair_id": "hist"},
            {},
            history=[{"val_rmse": 0.45}],
        ),
        FakeRun(
            "contrastive_hist",
            {"training_method": "contrastive", "pair_id": "hist"},
            {},
            history=[{"val_rmse": 0.6}],
        ),
    ]

    class Api:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: Api()))

    out = tmp_path / "pe_history.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
            "--strict",
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())
    assert payload["winner"] == "jepa"
    assert payload["pairs"] == 1


def test_paired_effect_reads_summary_json_dict(monkeypatch, tmp_path):
    """Summary objects exposing _json_dict should be coerced into mappings."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")

    class SummaryWithJsonDict:
        def __init__(self, payload):
            self._json_dict = dict(payload)

    runs = [
        FakeRun(
            "jepa_json_dict",
            {"training_method": "jepa", "pair_id": "pid"},
            SummaryWithJsonDict({"val_rmse": 0.41}),
        ),
        FakeRun(
            "contrastive_json_dict",
            {"training_method": "contrastive", "pair_id": "pid"},
            SummaryWithJsonDict({"val_rmse": 0.56}),
        ),
    ]

    class Api:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: Api()))

    out = tmp_path / "pe_json_dict.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
            "--strict",
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())
    assert payload["winner"] == "jepa"
    assert payload["pairs"] == 1


def test_paired_effect_retries_on_http_error(monkeypatch, tmp_path):
    """Transient HTTP errors when listing runs should be retried."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    flaky_runs: List[FakeRun] = [
        FakeRun("jepa", {"training_method": "jepa", "pair_id": "pid"}, {"val_rmse": 0.4}),
        FakeRun(
            "contrastive",
            {"training_method": "contrastive", "pair_id": "pid"},
            {"val_rmse": 0.6},
        ),
    ]

    class Api:
        def __init__(self):
            self.calls = 0

        def runs(self, path, filters=None):
            self.calls += 1
            if self.calls == 1:
                raise requests.exceptions.HTTPError("502 Bad Gateway")
            return flaky_runs

    api = Api()
    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: api))

    out = tmp_path / "pe_retry.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
            "--strict",
        ],
    )

    pe.main()

    assert api.calls == 2
    payload = json.loads(out.read_text())
    assert payload["winner"] == "jepa"


def test_paired_effect_handles_summary_items_attribute_error(monkeypatch, tmp_path):
    """Summary.items implementations raising AttributeError should be ignored."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")

    class SummaryWithBrokenItems:
        def __init__(self):
            self._json_dict = "corrupted"

        def keys(self):  # pragma: no cover - exercised indirectly during failure
            return self._json_dict.keys()

        def items(self):  # pragma: no cover - exercised indirectly during failure
            # Mimic wandb.old.summary.Summary relying on ``keys()``.
            return [(key, None) for key in self.keys()]

    runs = [
        FakeRun(
            "jepa_items_error",
            {"training_method": "jepa", "pair_id": "pid"},
            SummaryWithBrokenItems(),
            summary_metrics={"val_rmse": 0.41},
        ),
        FakeRun(
            "contrastive_items_error",
            {"training_method": "contrastive", "pair_id": "pid"},
            SummaryWithBrokenItems(),
            summary_metrics={"val_rmse": 0.6},
        ),
    ]

    class Api:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: Api()))

    out = tmp_path / "pe_summary_items_error.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
            "--strict",
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())
    assert payload["winner"] == "jepa"
    assert payload["pairs"] == 1


def test_paired_effect_reports_method_counts_when_metrics_missing(monkeypatch, tmp_path, capsys):
    """Strict mode should include per-method diagnostics when no metrics exist."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")

    runs = [
        FakeRun(
            "jepa_missing_metrics",
            {"training_method": "jepa", "pair_id": "pid"},
            {},
        ),
        FakeRun(
            "contrastive_missing_metrics",
            {"training_method": "contrastive", "pair_id": "pid"},
            {},
        ),
    ]

    class Api:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: Api()))

    out = tmp_path / "pe_missing_metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
            "--strict",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        pe.main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "Eligible runs summary" in output
    assert "jepa: runs=1, eligible=1" in output
    assert "contrastive: runs=1, eligible=1" in output
    assert not out.exists()


def test_paired_effect_limits_pairs_to_shared_seeds(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_ENTITY", "ent")

    # Simulate a single backbone (pair_id="abc123") evaluated under three seeds for
    # both methods.  Each method also reports duplicate runs per seed to mimic the
    # behaviour of larger sweeps (e.g. WANDB_COUNT=30) exploring extra hyper-params
    # that do not alter the pairing key.
    runs = [
        FakeRun("j_seed1_a", {"training_method": "jepa", "pair_id": "abc123", "seed": 1}, {"val_rmse": 0.50}),
        FakeRun("j_seed1_b", {"training_method": "jepa", "pair_id": "abc123", "seed": 1}, {"val_rmse": 0.52}),
        FakeRun("j_seed2",   {"training_method": "jepa", "pair_id": "abc123", "seed": 2}, {"val_rmse": 0.48}),
        FakeRun("j_seed3",   {"training_method": "jepa", "pair_id": "abc123", "seed": 3}, {"val_rmse": 0.46}),
        FakeRun("c_seed1",   {"training_method": "contrastive", "pair_id": "abc123", "seed": 1}, {"val_rmse": 0.55}),
        FakeRun("c_seed1_b", {"training_method": "contrastive", "pair_id": "abc123", "seed": 1}, {"val_rmse": 0.53}),
        FakeRun("c_seed2",   {"training_method": "contrastive", "pair_id": "abc123", "seed": 2}, {"val_rmse": 0.50}),
        FakeRun("c_seed3",   {"training_method": "contrastive", "pair_id": "abc123", "seed": 3}, {"val_rmse": 0.47}),
    ]

    class FakeApi:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))

    out = tmp_path / "pe_seeds.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())

    # Even with eight total runs, the tool should report three paired deltas – one
    # per shared seed – and note that they came from a single pair_id.
    assert payload["pairs"] == 3
    assert payload["pairs_used"] == 1
    # mean contrastive minus JEPA per seed → ((0.54-0.51) + (0.50-0.48) + (0.47-0.46)) / 3
    expected_delta = ((0.54 - 0.51) + (0.50 - 0.48) + (0.47 - 0.46)) / 3
    assert pytest.approx(payload["mean_delta_contrastive_minus_jepa"], rel=1e-6) == expected_delta


def test_paired_effect_ties_default_to_jepa(monkeypatch, tmp_path):
    """When the aggregate delta is exactly zero the code should not invent a tiebreaker."""

    monkeypatch.setenv("WANDB_ENTITY", "ent")

    runs = [
        FakeRun("j_seed", {"training_method": "jepa", "pair_id": "pid", "seed": 1}, {"val_rmse": 0.5}),
        FakeRun("c_seed", {"training_method": "contrastive", "pair_id": "pid", "seed": 1}, {"val_rmse": 0.5}),
    ]

    class FakeApi:
        def runs(self, path, filters=None):
            return runs

    monkeypatch.setattr(pe, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))

    out = tmp_path / "pe_tie.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pe",
            "--project",
            "proj",
            "--group",
            "grp",
            "--out",
            str(out),
        ],
    )

    pe.main()
    payload = json.loads(out.read_text())

    assert payload["pairs"] == 1
    assert payload["winner"] == "jepa"
    assert payload["mean_delta_contrastive_minus_jepa"] == 0.0


def test_phase1_decision_handles_ties_and_missing_keys():
    payload = {
        "direction": "min",
        "winner": "contrastive",
        "mean_delta_contrastive_minus_jepa": 0.0,
        "pairs": 1,
    }

    winner, task, tie, tie_breaker, tied_primary = pd.resolve_phase1_decision(payload)
    assert winner == "contrastive"
    assert task == "regression"
    assert tie is False
    assert tie_breaker is False
    assert tied_primary is True

    # Missing winner but non-zero delta → derive from direction.
    payload2 = {
        "direction": "max",
        "mean_delta_contrastive_minus_jepa": 0.5,
        "task": None,
    }

    winner2, task2, tie2, tie_breaker2, tied_primary2 = pd.resolve_phase1_decision(payload2)
    assert winner2 == "contrastive"
    assert task2 == "classification"
    assert tie2 is False
    assert tie_breaker2 is False
    assert tied_primary2 is False


def test_phase1_decision_detects_tie_breaker_resolution():
    payload = {
        "direction": "min",
        "winner": "jepa",
        "mean_delta_contrastive_minus_jepa": 5e-4,
        "primary_metric": {"tied": True, "tolerance": 1e-2},
        "tie_breaker_used": True,
        "task": "regression",
    }

    winner, task, tie, tie_breaker, tied_primary = pd.resolve_phase1_decision(payload)
    assert winner == "jepa"
    assert task == "regression"
    assert tie is False
    assert tie_breaker is True
    assert tied_primary is True


def test_export_best_reads_summary_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DIR", str(tmp_path))
    monkeypatch.setenv("GRID_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")

    class EmptySummary:
        _json_dict: Dict[str, Any] = {}

    runs = [
        FakeRun(
            "jepa_summary_metrics",
            {
                "training_method": "jepa",
                "gnn_type": "gine",
                "hidden_dim": 384,
                "num_layers": 4,
                "mask_ratio": 0.15,
            },
            EmptySummary(),
            summary_metrics={"val_rmse": 0.42},
        ),
        FakeRun(
            "contrastive_summary_metrics",
            {
                "training_method": "contrastive",
                "gnn_type": "gine",
                "hidden_dim": 384,
                "num_layers": 4,
                "mask_ratio": 0.2,
            },
            EmptySummary(),
            summary_metrics={"val_rmse": 0.6},
        ),
    ]

    class Api:
        def sweep(self, sweep_id):
            return FakeSweep(runs)

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: Api()))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_json = tmp_path / "best_summary_metrics.json"
    out_yaml = tmp_path / "phase2_summary_metrics.yaml"

    monkeypatch.setenv("METHOD_WINNER", "jepa")
    monkeypatch.setenv("SWEEP_CACHE_DIR", "/tmp/cache")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eb",
            "--sweep-id",
            "ent/proj/sw1",
            "--task",
            "regression",
            "--out",
            str(out_json),
            "--phase2-yaml",
            str(out_yaml),
        ],
    )

    eb.main()

    data = json.loads(out_json.read_text())
    assert data["training_method"] == "jepa"
    assert data["mask_ratio"] == 0.15


def test_export_best_respects_winner_and_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DIR", str(tmp_path))
    monkeypatch.setenv("GRID_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")

    runs = [
        FakeRun(
            "r1",
            {
                "training_method": "jepa",
                "gnn_type": "gine",
                "hidden_dim": 384,
                "num_layers": 4,
                "mask_ratio": 0.1,
                "aug_rotate": 1,
            },
            {"val_rmse": 0.4, "val_mae": 0.3},
            "1",
        ),
        FakeRun(
            "r2",
            {
                "training_method": "jepa",
                "gnn_type": "gine",
                "hidden_dim": 384,
                "num_layers": 4,
                "mask_ratio": 0.2,
                "aug_rotate": 0,
            },
            {"val_rmse": 0.6, "val_mae": 0.5},
            "2",
        ),
    ]

    class FakeApi:
        def sweep(self, sweep_id):
            return FakeSweep(runs)

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_json = tmp_path / "best.json"
    out_yaml = tmp_path / "phase2.yaml"

    monkeypatch.setenv("SWEEP_CACHE_DIR", "/tmp/cache")
    monkeypatch.delenv("CI_FORCE_GPU_PHASE2", raising=False)
    monkeypatch.setenv("CI_FORCE_CPU_PHASE2", "1")

    # winner jepa
    monkeypatch.setenv("METHOD_WINNER", "jepa")
    monkeypatch.setattr(sys, "argv", [
        "eb", "--sweep-id", "ent/proj/sw1", "--task", "regression",
        "--out", str(out_json), "--phase2-yaml", str(out_yaml), "--emit-bounds"
    ])
    eb.main()
    data = yaml.safe_load(out_yaml.read_text())
    assert data["parameters"]["training_method"]["value"] == "jepa"
    assert "aug_rotate" not in data["parameters"]
    assert data["parameters"]["cache-datasets"]["value"] == 1
    assert data["parameters"]["cache-dir"]["value"] == "${env:SWEEP_CACHE_DIR}"
    assert data["parameters"]["num-workers"]["value"] == 4
    assert data["parameters"]["prefetch-factor"]["value"] == 2
    assert data["parameters"]["persistent-workers"]["value"] == 0
    assert data["parameters"]["pin-memory"]["value"] == 0
    assert data["parameters"]["bf16"]["value"] == 1
    assert data["parameters"]["devices"]["value"] == 1
    assert data["parameters"]["use-wandb"]["value"] == 1

    # winner contrastive
    monkeypatch.setenv("METHOD_WINNER", "contrastive")
    monkeypatch.setattr(sys, "argv", [
        "eb", "--sweep-id", "ent/proj/sw1", "--task", "regression",
        "--out", str(out_json), "--phase2-yaml", str(out_yaml), "--emit-bounds"
    ])
    eb.main()
    data = yaml.safe_load(out_yaml.read_text())
    assert data["parameters"]["training_method"]["value"] == "contrastive"
    assert "aug_rotate" in data["parameters"]


def test_export_best_prefers_gpu_outside_ci(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DIR", str(tmp_path))
    monkeypatch.setenv("GRID_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("SWEEP_CACHE_DIR", "/tmp/cache")
    monkeypatch.delenv("CI_FORCE_CPU_PHASE2", raising=False)
    monkeypatch.delenv("CI_FORCE_GPU_PHASE2", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    runs = [
        FakeRun(
            "r1",
            {
                "training_method": "jepa",
                "gnn_type": "gine",
                "hidden_dim": 384,
                "num_layers": 4,
                "mask_ratio": 0.2,
            },
            {"val_rmse": 0.4, "val_mae": 0.3},
            "1",
        ),
    ]

    class FakeApi:
        def sweep(self, sweep_id):
            return FakeSweep(runs)

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_json = tmp_path / "best.json"
    out_yaml = tmp_path / "phase2.yaml"

    monkeypatch.setattr(sys, "argv", [
        "eb", "--sweep-id", "ent/proj/sw1", "--task", "regression",
        "--out", str(out_json), "--phase2-yaml", str(out_yaml), "--emit-bounds"
    ])

    eb.main()
    data = yaml.safe_load(out_yaml.read_text())

    assert data["parameters"]["persistent-workers"]["value"] == 1
    assert data["parameters"]["pin-memory"]["value"] == 1
    assert data["parameters"]["devices"]["value"] == 2


def test_export_best_forces_cache_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DIR", str(tmp_path))
    monkeypatch.setenv("GRID_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.delenv("SWEEP_CACHE_DIR", raising=False)

    runs = [
        FakeRun(
            "best",
            {
                "training_method": "jepa",
                "gnn_type": "gine",
                "hidden_dim": 256,
                "num_layers": 3,
                "cache-datasets": 0,
            },
            {"val_rmse": 0.4},
        ),
        FakeRun(
            "runner_up",
            {
                "training_method": "jepa",
                "gnn_type": "gine",
                "hidden_dim": 256,
                "num_layers": 3,
                "cache-datasets": 1,
            },
            {"val_rmse": 0.5},
        ),
    ]

    class FakeApi:
        def sweep(self, sweep_id):
            return FakeSweep(runs)

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_json = tmp_path / "best.json"
    out_yaml = tmp_path / "phase2.yaml"

    monkeypatch.setenv("METHOD_WINNER", "jepa")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eb",
            "--sweep-id",
            "ent/proj/sw1",
            "--task",
            "regression",
            "--out",
            str(out_json),
            "--phase2-yaml",
            str(out_yaml),
            "--emit-bounds",
        ],
    )

    eb.main()

    data = yaml.safe_load(out_yaml.read_text())
    cache_datasets = data["parameters"]["cache-datasets"]
    assert cache_datasets["value"] == 1
    assert "values" not in cache_datasets

    cache_dir = data["parameters"]["cache-dir"]
    assert cache_dir["value"] == "cache/graphs"
    assert "values" not in cache_dir

    # missing runs
    class EmptyApi:
        def sweep(self, sweep_id):
            return FakeSweep([])

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: EmptyApi()))
    monkeypatch.setattr(sys, "argv", [
        "eb", "--sweep-id", "ent/proj/sw1", "--task", "regression",
        "--out", str(out_json), "--phase2-yaml", str(out_yaml)
    ])
    with pytest.raises(RuntimeError):
        eb.main()


def test_export_best_extends_fixed_params(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DIR", str(tmp_path))
    monkeypatch.setenv("GRID_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")

    runs = [
        FakeRun(
            "r1",
            {
                "training_method": "jepa",
                "gnn_type": "gine",
                "hidden_dim": 256,
                "num_layers": 3,
                "mask_ratio": 0.25,
                "contiguity": 1,
            },
            {"val_rmse": 0.4},
            "r1",
        ),
        FakeRun(
            "r2",
            {
                "training_method": "jepa",
                "gnn_type": "gine",
                "hidden_dim": 256,
                "num_layers": 3,
                "mask_ratio": 0.25,
                "contiguity": 1,
            },
            {"val_rmse": 0.5},
            "r2",
        ),
    ]

    sweep_cfg = {
        "parameters": {
            "hidden_dim": {"values": [128, 256, 512]},
            "num_layers": {"values": [2, 3, 4]},
            "mask_ratio": {"min": 0.1, "max": 0.4},
            "contiguity": {"values": [0, 1]},
        }
    }

    class FakeApi:
        def sweep(self, sweep_id):
            return FakeSweep(runs, config=sweep_cfg)

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_json = tmp_path / "best_ext.json"
    out_yaml = tmp_path / "phase2_ext.yaml"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eb",
            "--sweep-id",
            "ent/proj/sw1",
            "--task",
            "regression",
            "--out",
            str(out_json),
            "--phase2-yaml",
            str(out_yaml),
            "--emit-bounds",
        ],
    )
    eb.main()
    data = yaml.safe_load(out_yaml.read_text())
    params = data["parameters"]
    assert params["mask_ratio"]["min"] == pytest.approx(0.2, rel=1e-6)
    assert params["mask_ratio"]["max"] == pytest.approx(0.3, rel=1e-6)
    assert params["hidden_dim"]["values"] == [256, 512]
    assert params["num_layers"]["values"] == [3, 4]
    assert params["contiguity"]["values"] == [1, 0]

    # Disable extension to recover legacy behaviour.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eb",
            "--sweep-id",
            "ent/proj/sw1",
            "--task",
            "regression",
            "--out",
            str(out_json),
            "--phase2-yaml",
            str(out_yaml),
            "--emit-bounds",
            "--no-extend-fixed",
        ],
    )
    eb.main()
    data = yaml.safe_load(out_yaml.read_text())
    params = data["parameters"]
    assert params["hidden_dim"]["value"] == 256
    assert params["num_layers"]["value"] == 3
    assert params["contiguity"]["value"] == 1


def test_export_best_rejects_tracked_template(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    monkeypatch.setenv("APP_DIR", str(repo_root))
    monkeypatch.setenv("GRID_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")

    runs = [
        FakeRun(
            "r1",
            {
                "training_method": "jepa",
                "gnn_type": "gine",
                "hidden_dim": 384,
                "num_layers": 4,
                "mask_ratio": 0.1,
            },
            {"val_rmse": 0.4, "val_mae": 0.3},
            "1",
        ),
    ]

    class FakeApi:
        def sweep(self, sweep_id):
            return FakeSweep(runs)

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_json = tmp_path / "best.json"
    tracked_yaml = repo_root / "sweeps" / "grid_sweep_phase2.yaml"

    monkeypatch.setenv("METHOD_WINNER", "jepa")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eb",
            "--sweep-id",
            "ent/proj/sw1",
            "--task",
            "regression",
            "--out",
            str(out_json),
            "--phase2-yaml",
            str(tracked_yaml),
            "--emit-bounds",
        ],
    )

    with pytest.raises(RuntimeError, match="Refusing to overwrite tracked Phase-2 template"):
        eb.main()


def test_recheck_topk_summary_and_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_ENTITY", "ent")

    top_cfg = {"training_method": "jepa", "mask_ratio": 0.1}
    sweep_runs = [
        FakeRun("r1", top_cfg, {"val_rmse": 0.2}),
        FakeRun("r2", {"training_method": "jepa", "mask_ratio": 0.2}, {"val_rmse": 0.5}),
    ]

    class FakeApi:
        def __init__(self):
            self._sweep = FakeSweep(sweep_runs)

        def sweep(self, sweep_id):
            return self._sweep

        def runs(self, path, filters=None):
            return [
                FakeRun("a", {**top_cfg, "seed": 1000}, {"val_rmse": 0.3}),
                FakeRun("b", {**top_cfg, "seed": 1001}, {"val_rmse": 0.5}),
            ]

    monkeypatch.setattr(rc, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    device_masks: List[Optional[str]] = []

    def fake_run_once(*_a, device_mask=None, **_k):
        device_masks.append(device_mask)
        return 0

    monkeypatch.setattr(rc, "run_once", fake_run_once)
    monkeypatch.setattr(rc, "time", types.SimpleNamespace(sleep=lambda *_a, **_k: None))
    monkeypatch.setenv("PHASE2_RECHECK_AGENT_COUNT", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "rc", "--sweep", "ent/proj/sw1", "--program", "prog.py",
        "--unlabeled-dir", "u", "--labeled-dir", "l", "--topk", "1",
        "--extra_seeds", "2", "--project", "proj", "--group", "grp",
        "--metric", "val_rmse", "--direction", "min", "--out", str(out)
    ])
    rc.main()
    data = json.loads(out.read_text())
    assert data["results"][0]["n"] == 2
    assert pytest.approx(data["results"][0]["mean"], rel=1e-6) == 0.4
    assert sorted(mask for mask in device_masks if mask is not None) == ["0", "1"]

    # empty sweep
    class EmptyApi:
        def sweep(self, sweep_id):
            return FakeSweep([])

        def runs(self, path, filters=None):
            return []

    calls = {"run_once": 0}

    def fake_run_once(*a, device_mask=None, **k):
        calls["run_once"] += 1
        return 0

    monkeypatch.setattr(rc, "wandb", types.SimpleNamespace(Api=lambda: EmptyApi()))
    monkeypatch.setattr(rc, "run_once", fake_run_once)
    monkeypatch.setattr(sys, "argv", [
        "rc", "--sweep", "ent/proj/sw1", "--program", "prog.py",
        "--unlabeled-dir", "u", "--labeled-dir", "l", "--topk", "1",
        "--extra_seeds", "2", "--project", "proj", "--group", "grp",
        "--metric", "val_rmse", "--direction", "min", "--out", str(out)
    ])
    rc.main()
    data = json.loads(out.read_text())
    assert data["results"] == []
    assert calls["run_once"] == 0

    def test_recheck_topk_handles_string_seeds(monkeypatch, tmp_path):
        monkeypatch.setenv("WANDB_ENTITY", "ent")

        top_cfg = {"training_method": "jepa", "mask_ratio": 0.1}
        sweep_runs = [
            FakeRun("r1", top_cfg, {"val_rmse": 0.2}),
        ]

        class FakeApi:
            def __init__(self):
                self._sweep = FakeSweep(sweep_runs)

            def sweep(self, sweep_id):
                return self._sweep

            def runs(self, path, filters=None):
                return [
                    FakeRun("a", {**top_cfg, "seed": "1000"}, {"val_rmse": 0.25}),
                    FakeRun("b", {**top_cfg, "seed": "1001"}, {"val_rmse": 0.35}),
                ]

        monkeypatch.setattr(rc, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
        monkeypatch.setattr(rc, "run_once", lambda *a, **k: 0)
        monkeypatch.setattr(rc, "time", types.SimpleNamespace(sleep=lambda x: None))

        out = tmp_path / "summary.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rc",
                "--sweep",
                "ent/proj/sw1",
                "--program",
                "prog.py",
                "--unlabeled-dir",
                "u",
                "--labeled-dir",
                "l",
                "--topk",
                "1",
                "--extra_seeds",
                "2",
                "--project",
                "proj",
                "--group",
                "grp",
                "--metric",
                "val_rmse",
                "--direction",
                "min",
                "--out",
                str(out),
            ],
        )
        rc.main()
        data = json.loads(out.read_text())
        assert data["results"][0]["n"] == 2
        assert pytest.approx(data["results"][0]["mean"], rel=1e-6) == 0.3