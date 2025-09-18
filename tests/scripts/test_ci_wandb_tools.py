import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from scripts.ci import export_best_from_wandb as eb
from scripts.ci import paired_effect_from_wandb as pe
from scripts.ci import recheck_topk_from_wandb as rc


class FakeRun:
    def __init__(self, name, config, summary, run_id="runid"):
        self.name = name
        self.config = config
        self.summary = summary
        self.id = run_id


class FakeSweep:
    def __init__(self, runs):
        self.runs = runs

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