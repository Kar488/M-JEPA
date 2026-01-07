"""Tests for the lightweight Tox21 fallback pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import types

import pytest


class _DummyWB:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []

    def log(self, payload):  # pragma: no cover - trivial container
        self.logged.append(dict(payload))

    def finish(self):  # pragma: no cover - trivial container
        pass


@pytest.fixture
def tox21_module(monkeypatch):
    """Import ``scripts.commands.tox21`` with the fallback implementation active."""

    if "experiments.case_study" in sys.modules:
        monkeypatch.delitem(sys.modules, "experiments.case_study", raising=False)

    if "torch" not in sys.modules:
        dummy_cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=dummy_cuda))

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "experiments.case_study":
            raise ImportError("case study unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    sys.modules.pop("scripts.commands.tox21", None)

    module = importlib.import_module("scripts.commands.tox21")
    assert getattr(module, "_TOX21_FALLBACK_ACTIVE", False) is True
    return module


def test_cmd_tox21_runs_with_fallback(tmp_path, monkeypatch, tox21_module):
    csv_path = tmp_path / "tox21.csv"
    csv_path.write_text(
        "smiles,NR-AR\n"
        "C,1\n"
        "CC,0\n"
        "CCC,1\n"
        "CCCC,0\n"
    )

    report_dir = tmp_path / "reports"
    env_path = tmp_path / "github.env"

    monkeypatch.setattr(tox21_module, "maybe_init_wandb", lambda *a, **k: _DummyWB())
    monkeypatch.setattr(tox21_module, "log_effective_gnn", lambda *a, **k: None)
    monkeypatch.setenv("GITHUB_ENV", str(env_path))

    args = argparse.Namespace(
        csv=str(csv_path),
        tasks=["NR-AR"],
        dataset="tox21",
        tox21_dir=str(report_dir),
        pretrain_epochs=1,
        finetune_epochs=1,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        class_weights="auto",
        hidden_dim=64,
        num_layers=2,
        gnn_type="edge_mpnn",
        add_3d=False,
        contrastive=False,
        triage_pct=0.2,
        no_calibrate=True,
        device="cpu",
        devices=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        encoder_checkpoint="ckpt.pt",
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_shape_coercion=False,
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
        full_finetune=False,
        unfreeze_top_layers=0,
        tox21_head_batch_size=64,
        use_wandb=False,
        wandb_project="proj",
        wandb_tags=[],
    )

    tox21_module.cmd_tox21(args)

    summary_path = report_dir / "tox21_summary.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text())
    assert payload["dataset"] == "tox21"
    assert payload["tasks"]["NR-AR"]["task"] == "NR-AR"
    assert payload.get("overall_gate_passed") is True
    task_payload = payload["tasks"]["NR-AR"]
    diagnostics = task_payload["diagnostics"]
    assert diagnostics["benchmark_threshold"] is None
    assert diagnostics["benchmark_threshold_available"] is False
    assert diagnostics["benchmark_threshold_original"] is not None
    assert diagnostics["benchmark_comparison_performed"] is False
    assert diagnostics["benchmark_override"] is True
    assert diagnostics["benchmark_override_reason"] == "skipped roc_auc comparison in fallback"
    assert task_payload["met_benchmark_selected"] is True
    assert task_payload["tox21_gate_passed"] is True
    eval_payload = task_payload["evaluations"][0]
    assert eval_payload["met_benchmark"] is True
    assert eval_payload["benchmark_threshold"] is None
    assert eval_payload["metrics"]["roc_auc"] is not None
    assert eval_payload["tox21_gate_passed"] is True

    stage_dir = report_dir / "stage-outputs"
    stage_files = list(stage_dir.glob("*.json"))
    assert stage_files, "stage outputs were not produced by fallback"

    gate_env = env_path.read_text()
    assert "TOX21_MET_GATE=true" in gate_env


def test_cmd_tox21_exits_on_threading_failure(monkeypatch, tox21_module):
    def boom(*args, **kwargs):
        raise TypeError("bad threads")

    monkeypatch.setattr(tox21_module, "configure_omp_threads", boom)

    args = argparse.Namespace(num_workers=0)

    with pytest.raises(SystemExit) as excinfo:
        tox21_module.cmd_tox21(args)

    assert excinfo.value.code == 5


def test_module_main_delegates_to_train_jepa(monkeypatch, tmp_path):
    import scripts.commands.tox21 as tox21

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tox21,
        "cmd_tox21",
        lambda args: captured.update({"csv": getattr(args, "csv", None)}),
    )

    def fake_build_parser():
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command", required=True)
        tox = sub.add_parser("tox21")
        tox.add_argument("--csv", required=True)
        tox.set_defaults(func=tox21.cmd_tox21)
        return parser

    fake_module = types.SimpleNamespace(build_parser=fake_build_parser)
    monkeypatch.setitem(sys.modules, "scripts.train_jepa", fake_module)
    import scripts

    monkeypatch.setattr(scripts, "train_jepa", fake_module, raising=False)

    path = tmp_path / "data.csv"
    path.write_text("stub", encoding="utf-8")

    rc = tox21.main(["--csv", str(path)])

    assert rc == 0
    assert captured["csv"] == str(path)


def test_module_main_standalone_parser_handles_flags(monkeypatch, tmp_path):
    import builtins
    import scripts.commands.tox21 as tox21

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tox21,
        "cmd_tox21",
        lambda args: captured.update(
            {
                "persistent": getattr(args, "persistent_workers", None),
                "pin": getattr(args, "pin_memory", None),
                "bf16": getattr(args, "bf16", None),
                "use_wandb": getattr(args, "use_wandb", None),
                "hidden_provided": getattr(args, "_hidden_dim_provided", False),
            }
        ),
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[override]
        if name == "scripts.train_jepa":
            raise ModuleNotFoundError("train_jepa missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("smiles,NR-AR\nC,1\n", encoding="utf-8")

    rc = tox21.main(
        [
            "--csv",
            str(csv_path),
            "--tasks",
            "NR-AR",
            "--encoder-checkpoint",
            str(tmp_path / "encoder.pt"),
            "--persistent-workers",
            "0",
            "--pin-memory",
            "0",
            "--bf16",
            "1",
            "--use-wandb",
            "--num-workers",
            "4",
            "--prefetch-factor",
            "2",
            "--hidden-dim",
            "128",
        ]
    )

    assert rc == 0
    assert captured["persistent"] is False
    assert captured["pin"] is False
    assert captured["bf16"] is True
    assert captured["use_wandb"] is True
    assert captured["hidden_provided"] is True


def test_module_main_standalone_parser_accepts_legacy_task(monkeypatch, tmp_path):
    import builtins
    import scripts.commands.tox21 as tox21

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tox21,
        "cmd_tox21",
        lambda args: captured.update(
            {
                "task": getattr(args, "task", None),
                "tasks": getattr(args, "tasks", None),
            }
        ),
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[override]
        if name == "scripts.train_jepa":
            raise ModuleNotFoundError("train_jepa missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("smiles,NR-AR\nC,1\n", encoding="utf-8")

    rc = tox21.main([
        "--csv",
        str(csv_path),
        "--task",
        "NR-AR",
    ])

    assert rc == 0
    assert captured["task"] == "NR-AR"
    assert captured["tasks"] == []

def test_tox21_evaluation_emits_ig_artifacts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import numpy as np
    import torch.nn as nn

    from data.mdataset import GraphData, GraphDataset
    import training.supervised as supervised_mod

    class TinyEncoder(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.hidden_dim = dim
            self.linear = nn.Linear(dim, dim, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(dim))

        def forward(self, x, adj):  # noqa: ARG002 - match encoder signature
            return self.linear(x)

    graphs: list[GraphData] = []
    labels: list[float] = []
    smiles: list[str] = []
    for i in range(4):
        x = np.full((3, 2), float(i + 1), dtype=np.float32)
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        edge_attr = np.ones((edge_index.shape[1], 1), dtype=np.float32)
        graphs.append(GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr))
        labels.append(float(i % 2))
        smiles.append("C" * (i + 1))

    dataset = GraphDataset(graphs, labels, smiles)
    encoder = TinyEncoder(dataset.node_dim)
    train_idx = [0, 1]
    val_idx = [2]
    test_idx = [3]
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stage_config = {"report_dir": str(report_dir), "task_name": "nr-ar"}
    supervised_mod.set_stage_config(stage_config)

    torch.manual_seed(42)
    baseline_metrics = supervised_mod.train_linear_head(
        dataset,
        encoder,
        "classification",
        epochs=0,
        batch_size=2,
        lr=1e-3,
        patience=0,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        stage_config=stage_config,
    )

    torch.manual_seed(42)
    metrics = supervised_mod.train_linear_head(
        dataset,
        encoder,
        "classification",
        epochs=0,
        batch_size=2,
        lr=1e-3,
        patience=0,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        stage_config=stage_config,
        explain_mode="ig",
        explain_config={"steps": 4, "task_name": "nr-ar"},
    )

    assert (
        metrics["roc_auc"] == baseline_metrics["roc_auc"]
        or (math.isnan(metrics["roc_auc"]) and math.isnan(baseline_metrics["roc_auc"]))
    )

    ig_dir = report_dir / "ig_explanations" / "nr-ar"
    assert ig_dir.is_dir()
    atom_csvs = sorted(ig_dir.glob("*_atoms.csv"))
    bond_csvs = sorted(ig_dir.glob("*_bonds.csv"))
    heatmaps = sorted(ig_dir.glob("*_heatmap.png"))
    assert len(atom_csvs) == len(test_idx)
    assert len(bond_csvs) == len(test_idx)
    assert len(heatmaps) == len(test_idx)
    assert "ig_score" in atom_csvs[0].read_text(encoding="utf-8")
    assert metrics.get("ig_artifacts")
    for entry in metrics["ig_artifacts"]:
        assert os.path.exists(entry["atom_csv"])  # type: ignore[index]
        assert os.path.exists(entry["bond_csv"])  # type: ignore[index]
        assert os.path.exists(entry["heatmap_png"])  # type: ignore[index]

    supervised_mod.set_stage_config({})
