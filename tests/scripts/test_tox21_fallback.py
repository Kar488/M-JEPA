"""Tests for the lightweight Tox21 fallback pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import sys

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
        encoder_manifest="manifest.json",
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

    stage_dir = report_dir / "stage-outputs"
    stage_files = list(stage_dir.glob("*.json"))
    assert stage_files, "stage outputs were not produced by fallback"

    gate_env = env_path.read_text()
    assert "TOX21_MET_GATE=" in gate_env
