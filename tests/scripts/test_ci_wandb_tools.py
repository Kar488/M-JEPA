import json
import sys
import types
from pathlib import Path

import pytest
import yaml

# Provide a lightweight stub so modules can import wandb during test collection.
sys.modules.setdefault("wandb", types.SimpleNamespace(Api=lambda: None))

from scripts.ci import export_best_from_wandb as eb
from scripts.ci import paired_effect_from_wandb as pe
from scripts.ci import phase1_decision as pd
from scripts.ci import recheck_topk_from_wandb as rc


class FakeRun:
    def __init__(self, name, config, summary, run_id="runid"):
        self.name = name
        self.config = config
        self.summary = summary
        self.id = run_id


class FakeSweep:
    def __init__(self, runs, config=None):
        self.runs = runs
        self.config = config or {}

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

    winner, task, tie = pd.resolve_phase1_decision(payload)
    assert winner == "jepa"
    assert task == "regression"
    assert tie is True

    # Missing winner but non-zero delta → derive from direction.
    payload2 = {
        "direction": "max",
        "mean_delta_contrastive_minus_jepa": 0.5,
        "task": None,
    }

    winner2, task2, tie2 = pd.resolve_phase1_decision(payload2)
    assert winner2 == "contrastive"
    assert task2 == "classification"
    assert tie2 is False

def test_export_best_respects_winner_and_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DIR", str(tmp_path))
    monkeypatch.setenv("GRID_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")

    runs = [
        FakeRun("r1", {"mask_ratio": 0.1, "aug_rotate": 1}, {"val_rmse": 0.4, "val_mae": 0.3}, "1"),
        FakeRun("r2", {"mask_ratio": 0.2, "aug_rotate": 0}, {"val_rmse": 0.6, "val_mae": 0.5}, "2"),
    ]

    class FakeApi:
        def sweep(self, sweep_id):
            return FakeSweep(runs)

    monkeypatch.setattr(eb, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    monkeypatch.setattr(eb, "maybe_init_wandb", lambda *a, **k: None)

    out_json = tmp_path / "best.json"
    out_yaml = tmp_path / "phase2.yaml"

    monkeypatch.setenv("SWEEP_CACHE_DIR", "/tmp/cache")

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
            {"hidden_dim": 256, "num_layers": 3, "mask_ratio": 0.25, "contiguity": 1},
            {"val_rmse": 0.4},
            "r1",
        ),
        FakeRun(
            "r2",
            {"hidden_dim": 256, "num_layers": 3, "mask_ratio": 0.25, "contiguity": 1},
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
        FakeRun("r1", {"mask_ratio": 0.1}, {"val_rmse": 0.4, "val_mae": 0.3}, "1"),
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
    monkeypatch.setattr(rc, "run_once", lambda *a, **k: 0)
    monkeypatch.setattr(rc, "time", types.SimpleNamespace(sleep=lambda x: None))

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

    # empty sweep
    class EmptyApi:
        def sweep(self, sweep_id):
            return FakeSweep([])

        def runs(self, path, filters=None):
            return []

    calls = {"run_once": 0}

    def fake_run_once(*a, **k):
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