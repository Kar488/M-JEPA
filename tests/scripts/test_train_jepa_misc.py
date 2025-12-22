import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import pytest
from types import SimpleNamespace
from typing import Optional

import models.encoder  # noqa: F401
import models.factory  # noqa: F401
import scripts.commands.tox21 as tox_cmd
from scripts import train_jepa as tj

torch = pytest.importorskip("torch")


class DummyArray:
    def __init__(self, shape):
        self.shape = shape


class DummyGraph:
    def __init__(self):
        self.x = DummyArray((1, 3))
        self.edge_attr = None


class DummyDataset:
    def __init__(self):
        self.graphs = [DummyGraph()]

    def __len__(self):
        return 1


# ---------------------------------------------------------------------------
# cmd_benchmark tests
# ---------------------------------------------------------------------------


def test_cmd_benchmark_selects_best_method(tmp_path, monkeypatch):
    dataset = DummyDataset()

    def load_dataset_stub(path, label_col=None, add_3d=False, **kwargs):
        assert path == str(tmp_path)
        return dataset

    monkeypatch.setattr(tj, "load_directory_dataset", load_dataset_stub)

    class DummyEncoder:
        def load_state_dict(self, state):
            pass

    monkeypatch.setattr(tj, "build_encoder", lambda **kwargs: DummyEncoder())

    calls = []

    def train_linear_head_stub(**kwargs):
        if not calls:
            calls.append("jepa")
            return {"roc_auc": 0.6}
        calls.append("contrastive")
        return {"roc_auc": 0.8}

    monkeypatch.setattr(tj, "train_linear_head", train_linear_head_stub)
    monkeypatch.setattr(tj.torch, "load", lambda *a, **k: {"encoder": {}})

    class DummyWB:
        def __init__(self):
            self.logs = []

        def log(self, data):
            self.logs.append(data)

        def finish(self):
            pass

    holder = {}

    def maybe_init_wandb_stub(*args, **kwargs):
        holder["config"] = kwargs.get("config")
        holder["wb"] = DummyWB()
        return holder["wb"]

    monkeypatch.setattr(tj, "maybe_init_wandb", maybe_init_wandb_stub)

    args = argparse.Namespace(
        labeled_dir=str(tmp_path),
        test_dir=None,
        label_col="y",
        task_type="classification",
        epochs=1,
        batch_size=1,
        lr=0.01,
        temperature=(0.1),
        patience=1,
        devices=1,
        seeds=[0],
        jepa_encoder="jepa.pt",
        contrastive_encoder="cont.pt",
        dataset="esol",
        gnn_type="gcn",
        hidden_dim=16,
        num_layers=2,
        device="cpu",
        use_wandb=True,
        wandb_project="test",
        wandb_tags=[],
        add_3d=False,
        num_workers=3,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=6,
    )

    tj.cmd_benchmark(args)

    logs = holder["wb"].logs
    assert any(log.get("best_method") == "contrastive" for log in logs)
    assert any(log.get("benchmark_metric") == "rmse" for log in logs)

    config = holder["config"]
    assert config["hidden_dim"] == 16
    assert config["num_layers"] == 2
    assert config["gnn_type"] == "gcn"
    assert config["persistent_workers"] is False
    assert config["prefetch_factor"] == 6


def test_cmd_benchmark_modules_missing(monkeypatch):
    monkeypatch.setattr(tj, "load_directory_dataset", None)
    with pytest.raises(SystemExit) as ex:
        tj.cmd_benchmark(argparse.Namespace(test_dir=None))
    assert ex.value.code == 6


def test_cmd_benchmark_eval_only_uses_test_dir(tmp_path, monkeypatch):
    dataset = DummyDataset()

    def load_dataset_stub(path, label_col=None, add_3d=False, **kwargs):
        assert path == str(tmp_path / "test")
        return dataset

    monkeypatch.setattr(tj, "load_directory_dataset", load_dataset_stub)

    def eval_stub(ckpt_path, dataset, args, device):  # pragma: no cover - stub
        return {"roc_auc": 0.9}

    monkeypatch.setattr(tj, "evaluate_finetuned_head", eval_stub)

    calls = {"train_linear_head": 0}

    def train_head_stub(**kwargs):
        calls["train_linear_head"] += 1
        return {"roc_auc": 0.5}

    monkeypatch.setattr(tj, "train_linear_head", train_head_stub)

    ft_path = tmp_path / "ft.pt"
    ft_path.write_text("ckpt")

    args = argparse.Namespace(
        labeled_dir=str(tmp_path / "train"),
        test_dir=str(tmp_path / "test"),
        label_col="y",
        task_type="classification",
        epochs=1,
        batch_size=1,
        lr=0.01,
        patience=1,
        devices=1,
        seeds=[0],
        jepa_encoder="jepa.pt",
        contrastive_encoder=None,
        dataset="esol",
        gnn_type="gcn",
        hidden_dim=16,
        num_layers=2,
        device="cpu",
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        add_3d=False,
        ft_ckpt=str(ft_path),
    )

    tj.cmd_benchmark(args)
    assert calls["train_linear_head"] == 0


def test_cmd_benchmark_passes_loader_knobs(tmp_path, monkeypatch):
    dataset = DummyDataset()

    monkeypatch.setattr(tj, "load_directory_dataset", lambda *a, **k: dataset)

    class DummyEncoder:
        def load_state_dict(self, state):
            pass

    monkeypatch.setattr(tj, "build_encoder", lambda **kwargs: DummyEncoder())
    monkeypatch.setattr(tj.torch, "load", lambda *a, **k: {"encoder": {}})

    captured = {}

    def train_linear_head_stub(**kwargs):
        captured.update(
            {k: kwargs[k] for k in ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor", "bf16")}
        )
        return {"roc_auc": 0.5}

    monkeypatch.setattr(tj, "train_linear_head", train_linear_head_stub)

    args = argparse.Namespace(
        labeled_dir=str(tmp_path),
        test_dir=None,
        label_col="y",
        task_type="classification",
        epochs=1,
        batch_size=1,
        lr=0.01,
        patience=1,
        devices=1,
        seeds=[0],
        jepa_encoder="jepa.pt",
        contrastive_encoder=None,
        dataset="esol",
        gnn_type="gcn",
        hidden_dim=16,
        num_layers=2,
        device="cpu",
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        add_3d=False,
        num_workers=3,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=6,
        bf16=True,
    )

    tj.cmd_benchmark(args)

    assert captured == {
        "num_workers": 3,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 6,
        "bf16": True,
    }


# ---------------------------------------------------------------------------
# cmd_tox21 tests
# ---------------------------------------------------------------------------


def test_cmd_tox21_logs_metrics(tmp_path, monkeypatch):
    captures = {}

    def tox_stub(
        *,
        csv_path,
        task_name,
        pretrain_epochs,
        finetune_epochs,
        triage_pct=0.0,
        calibrate=True,
        device="cpu",
        **kwargs,
    ):
        captures.update(
            {
                k: kwargs[k]
                for k in (
                    "pretrain_lr",
                    "num_workers",
                    "pin_memory",
                    "persistent_workers",
                    "prefetch_factor",
                    "bf16",
                    "pretrain_time_budget_mins",
                    "finetune_time_budget_mins",
                )
            }
        )
        return 0.3, 0.1, 0.5, {"baseline": 0.2}

    monkeypatch.setattr(tj, "run_tox21_case_study", tox_stub)

    class DummyWB:
        def __init__(self):
            self.logs = []

        def log(self, data):
            self.logs.append(data)

        def finish(self):
            pass

    holder = {}

    def maybe_init_wandb_stub(*args, **kwargs):
        holder["config"] = kwargs.get("config")
        holder["wb"] = DummyWB()
        return holder["wb"]

    monkeypatch.setattr(tj, "maybe_init_wandb", maybe_init_wandb_stub)

    args = argparse.Namespace(
        csv=str(tmp_path / "tox.csv"),
        task="NR-AR",
        dataset="tox21",
        pretrain_epochs=1,
        finetune_epochs=1,
        pretrain_lr=5e-4,
        triage_pct=0.0,
        tox21_dir=str(tmp_path / "reports"),
        device="cpu",
        use_wandb=True,
        wandb_project="test",
        wandb_tags=[],
        num_workers=4,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=5,
        bf16=True,
        pretrain_time_budget_mins=7,
        finetune_time_budget_mins=3,
        gnn_type="edge_mpnn",
        hidden_dim=256,
        num_layers=3,
        add_3d=True,
    )

    tj.cmd_tox21(args)

    logs = holder["wb"].logs
    assert any(
        log.get("phase") == "tox21"
        and log.get("status") == "success"
        and log.get("mean_true") == 0.3
        for log in logs
    )
    assert any(log.get("benchmark_metric") == "roc_auc" for log in logs)

    assert captures == {
        "pretrain_lr": 5e-4,
        "num_workers": 4,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 5,
        "bf16": True,
        "pretrain_time_budget_mins": 7,
        "finetune_time_budget_mins": 3,
    }

    config = holder["config"]
    assert config["hidden_dim"] == 256
    assert config["num_layers"] == 3
    assert config["gnn_type"] == "edge_mpnn"
    assert config["pretrain_lr"] == 5e-4
    assert config["persistent_workers"] is False
    assert config["prefetch_factor"] == 5
    assert config["pin_memory"] is False
    assert config["bf16"] is True
    assert config["tasks"] == ["NR-AR"]
    assert config["task_count"] == 1
    assert config["dropout"] is None
    assert config["head_scheduler"] is None
    assert config["auto_pos_class_weight"] == {}
    assert "NR-AR" in config["class_balance"]

    summary_path = tmp_path / "reports" / "tox21_summary.json"
    assert summary_path.is_file()
    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["overall_gate_passed"] in {True, False}
    assert list(summary_payload["tasks"].keys()) == ["NR-AR"]

    manifest_path = tmp_path / "reports" / "run_manifest.json"
    assert manifest_path.is_file()


def test_cmd_tox21_inherits_best_config_loader_flags(tmp_path, monkeypatch):
    best_payload = {
        "devices": {"value": 2},
        "num_workers": {"value": 6},
        "prefetch_factor": {"value": 3},
        "pin_memory": {"value": 0},
        "persistent_workers": {"value": 0},
        "bf16": {"value": 0},
        "bf16_head": {"value": 1},
    }
    best_path = tmp_path / "best.json"
    best_path.write_text(json.dumps(best_payload))

    captures = {}

    def tox_stub(
        *,
        csv_path,
        task_name,
        pretrain_epochs,
        finetune_epochs,
        triage_pct=0.0,
        calibrate=True,
        device="cpu",
        **kwargs,
    ):
        keys = (
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "bf16",
            "bf16_head",
            "devices",
        )
        captures.update({k: kwargs[k] for k in keys})
        return 0.2, 0.1, 0.3, {"baseline": 0.15}

    monkeypatch.setattr(tj, "run_tox21_case_study", tox_stub)
    monkeypatch.setattr(tj, "maybe_init_wandb", lambda *a, **k: None)

    monkeypatch.setattr(tox_cmd.sys, "argv", ["train_jepa.py", "tox21"])

    args = argparse.Namespace(
        csv=str(tmp_path / "tox.csv"),
        task="NR-AR",
        dataset="tox21",
        pretrain_epochs=1,
        finetune_epochs=1,
        pretrain_lr=1e-4,
        triage_pct=0.0,
        tox21_dir=str(tmp_path / "reports"),
        device="cpu",
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        bf16=True,
        bf16_head=False,
        devices=1,
        hidden_dim=128,
        num_layers=3,
        gnn_type="edge_mpnn",
        add_3d=False,
        best_config_path=str(best_path),
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
    )

    tj.cmd_tox21(args)

    expected = {
        "num_workers": 6,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 3,
        "bf16": False,
        "bf16_head": True,
        "devices": 2,
    }
    assert {k: captures[k] for k in expected} == expected
    assert args.num_workers == 6
    assert args.pin_memory is False
    assert args.persistent_workers is False
    assert args.prefetch_factor == 3
    assert args.bf16 is False
    assert args.bf16_head is True
    assert args.devices == 2


def test_cmd_tox21_inherits_best_config_hidden_dim(tmp_path, monkeypatch):
    best_payload = {"hidden_dim": {"value": 384}}
    best_path = tmp_path / "best.json"
    best_path.write_text(json.dumps(best_payload))

    captures = {}

    def tox_stub(
        *,
        csv_path,
        task_name,
        pretrain_epochs,
        finetune_epochs,
        triage_pct=0.0,
        calibrate=True,
        device="cpu",
        **kwargs,
    ):
        captures["hidden_dim"] = kwargs["hidden_dim"]
        return 0.2, 0.1, 0.3, {"baseline": 0.15}

    monkeypatch.setattr(tj, "run_tox21_case_study", tox_stub)
    monkeypatch.setattr(tj, "maybe_init_wandb", lambda *a, **k: None)
    monkeypatch.setattr(tox_cmd.sys, "argv", ["train_jepa.py", "tox21"])

    args = argparse.Namespace(
        csv=str(tmp_path / "tox.csv"),
        task="NR-AR",
        dataset="tox21",
        pretrain_epochs=1,
        finetune_epochs=1,
        pretrain_lr=1e-4,
        triage_pct=0.0,
        tox21_dir=str(tmp_path / "reports"),
        device="cpu",
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        bf16=True,
        bf16_head=False,
        devices=1,
        hidden_dim=128,
        num_layers=3,
        gnn_type="edge_mpnn",
        add_3d=False,
        best_config_path=str(best_path),
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
    )

    tj.cmd_tox21(args)

    assert captures["hidden_dim"] == 384
    assert args.hidden_dim == 384


def test_cmd_tox21_auto_retries_allow_shape(monkeypatch, tmp_path):
    call_history: list[Optional[bool]] = []

    def tox_stub(**kwargs):
        call_history.append(kwargs.get("allow_shape_coercion"))
        if len(call_history) == 1:
            raise RuntimeError(
                "Encoder featurizer mismatch: checkpoint add_3d=True requested add_3d=False. "
                "Set allow_shape_coercion=true to override."
            )
        evaluation = SimpleNamespace(
            name="pretrain",
            encoder_source="pretrain_frozen",
            metrics={"roc_auc": 0.91},
            baseline_means={"baseline": 0.2},
            mean_true=0.3,
            mean_random=0.1,
            mean_pred=0.5,
            benchmark_metric="roc_auc",
            benchmark_threshold=0.6,
            met_benchmark=True,
            manifest_path="manifest.json",
        )
        result = SimpleNamespace(
            evaluations=[evaluation],
            threshold_rule=SimpleNamespace(metric="roc_auc", threshold=0.6),
            diagnostics={},
            encoder_hash="hash123",
            baseline_encoder_hash=None,
            encoder_load={},
            calibrator_state=None,
            split_summary={},
        )
        return result

    monkeypatch.setattr(tj, "run_tox21_case_study", tox_stub)

    class DummyWB:
        def __init__(self):
            self.logs = []

        def log(self, data):
            self.logs.append(data)

        def finish(self):
            pass

    holder = {}

    def maybe_init_wandb_stub(*args, **kwargs):
        holder["wb"] = DummyWB()
        return holder["wb"]

    monkeypatch.setattr(tj, "maybe_init_wandb", maybe_init_wandb_stub)

    csv_path = tmp_path / "tox.csv"
    csv_path.write_text("smiles,NR-AR\nC,1\n", encoding="utf-8")

    args = argparse.Namespace(
        csv=str(csv_path),
        task="NR-AR",
        dataset="tox21",
        pretrain_epochs=1,
        finetune_epochs=1,
        pretrain_lr=1e-4,
        triage_pct=0.0,
        tox21_dir=str(tmp_path / "reports"),
        device="cpu",
        use_wandb=True,
        wandb_project="proj",
        wandb_tags=[],
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        allow_shape_coercion=False,
        hidden_dim=128,
        num_layers=3,
        gnn_type="edge_mpnn",
        add_3d=False,
    )

    tj.cmd_tox21(args)

    assert call_history == [False, True]
    assert args.allow_shape_coercion is True

    logs = holder["wb"].logs
    assert any(log.get("allow_shape_coercion_retry") for log in logs)

    summary_path = tmp_path / "reports" / "tox21_summary.json"
    assert summary_path.is_file()
    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["overall_gate_passed"] in {True, False}

def test_cmd_tox21_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tj,
        "run_tox21_case_study",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    class DummyWB:
        def log(self, data):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(tj, "maybe_init_wandb", lambda *a, **k: DummyWB())
    monkeypatch.delenv("TOX21_DIR", raising=False)

    args = argparse.Namespace(
        csv="c",
        task="t",
        dataset="tox21",
        pretrain_epochs=1,
        finetune_epochs=1,
        pretrain_lr=1e-4,
        triage_pct=0.0,
        tox21_dir=str(tmp_path / "reports"),
        device="cpu",
        use_wandb=False,
        wandb_project="p",
        wandb_tags=[],
    )

    with pytest.raises(SystemExit) as ex:
        tj.cmd_tox21(args)
    assert ex.value.code == 5


def test_resolve_tox21_tasks_preserves_scalar_string():
    args = SimpleNamespace(tasks="NR-AR", task=None)
    assert tox_cmd._resolve_tox21_tasks(args) == ["NR-AR"]


def test_finalise_standalone_args_wraps_scalar_task():
    namespace = argparse.Namespace(tasks="NR-AR")
    finalised = tox_cmd._finalise_standalone_args(namespace)
    assert finalised.tasks == ["NR-AR"]


def test_resolve_task_encoder_checkpoint_switches_task(tmp_path):
    finetune_root = tmp_path / "finetune"
    source_dir = finetune_root / "NR-AR"
    target_dir = finetune_root / "NR-ER"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    base_checkpoint = source_dir / "encoder_ft_1909.pt"
    base_checkpoint.write_text("base", encoding="utf-8")
    target_checkpoint = target_dir / "encoder_ft_1909.pt"
    target_checkpoint.write_text("target", encoding="utf-8")

    resolved, info = tox_cmd._resolve_task_encoder_checkpoint(
        str(base_checkpoint),
        "NR-ER",
    )

    assert Path(resolved) == target_checkpoint
    assert info["source"] == "task_dir"
    assert info["task_dir"] == str(target_dir)


def test_resolve_task_encoder_checkpoint_handles_missing(tmp_path):
    finetune_root = tmp_path / "finetune"
    source_dir = finetune_root / "NR-AR"
    target_dir = finetune_root / "SR-ARE"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    base_checkpoint = source_dir / "encoder_ft.pt"
    base_checkpoint.write_text("base", encoding="utf-8")

    resolved, info = tox_cmd._resolve_task_encoder_checkpoint(
        str(base_checkpoint),
        "SR-ARE",
    )

    assert resolved == str(base_checkpoint)
    assert info["source"] == "task_dir_missing"
    assert info["task_dir"] == str(target_dir)


# ---------------------------------------------------------------------------
# cmd_grid_search tests
# ---------------------------------------------------------------------------


def test_cmd_grid_search_logs_and_csv(tmp_path, monkeypatch):
    def grid_stub(**kwargs):
        df = pd.DataFrame({"metric": [0.5]})
        out_csv = kwargs.get("out_csv")
        if out_csv:
            df.to_csv(out_csv, index=False)
        return df

    monkeypatch.setattr(tj, "run_grid_search", grid_stub)

    class DummyWB:
        def __init__(self):
            self.logs = []

        def log(self, data):
            self.logs.append(data)

        def finish(self):
            pass

    holder = {}

    def maybe_init_wandb_stub(*args, **kwargs):
        holder["wb"] = DummyWB()
        return holder["wb"]

    monkeypatch.setattr(tj, "maybe_init_wandb", maybe_init_wandb_stub)

    out_csv = tmp_path / "out.csv"
    args = argparse.Namespace(
        dataset_dir=str(tmp_path),
        unlabeled_dir=None,
        labeled_dir=None,
        label_col="y",
        task_type="classification",
        methods=["jepa"],
        mask_ratios=[0.1],
        contiguities=[0],
        hidden_dims=[16],
        num_layers_list=[2],
        gnn_types=["gcn"],
        ema_decays=[0.99],
        add_3d_options=[0],
        aug_rotate_options=[0],
        aug_mask_angle_options=[0],
        aug_dihedral_options=[0],
        pretrain_batch_sizes=[1],
        finetune_batch_sizes=[1],
        pretrain_epochs_options=[1],
        finetune_epochs_options=[1],
        learning_rates=[0.001],
        temperatures=[0.1],
        seeds=[0],
        device="cpu",
        use_wandb=True,
        wandb_project="test",
        wandb_tags=[],
        ckpt_dir=None,
        ckpt_every=0,
        use_scheduler=False,
        warmup_steps=0,
        out_csv=str(out_csv),
        best_config_out=None,
        force_refresh=False,
    )

    tj.cmd_grid_search(args)

    logs = holder["wb"].logs
    assert any(
        log.get("phase") == "grid_search" and log.get("status") == "start"
        for log in logs
    )
    assert any(
        log.get("phase") == "grid_search" and log.get("status") == "success"
        for log in logs
    )
    assert any("metric" in log for log in logs)
    assert out_csv.exists()


def test_cmd_grid_search_failure(monkeypatch, caplog, tmp_path):
    def failing(**kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(tj, "run_grid_search", failing)

    class DummyWB:
        def log(self, data):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(tj, "maybe_init_wandb", lambda *a, **k: DummyWB())

    args = argparse.Namespace(
        dataset_dir=str(tmp_path),
        unlabeled_dir=None,
        labeled_dir=None,
        label_col="y",
        task_type="classification",
        methods=["jepa"],
        mask_ratios=[0.1],
        contiguities=[0],
        hidden_dims=[16],
        num_layers_list=[2],
        gnn_types=["gcn"],
        ema_decays=[0.99],
        add_3d_options=[0],
        aug_rotate_options=[0],
        aug_mask_angle_options=[0],
        aug_dihedral_options=[0],
        pretrain_batch_sizes=[1],
        finetune_batch_sizes=[1],
        pretrain_epochs_options=[1],
        finetune_epochs_options=[1],
        learning_rates=[0.001],
        temperatures=[0.1],
        seeds=[0],
        device="cpu",
        use_wandb=False,
        wandb_project="p",
        wandb_tags=[],
        ckpt_dir=None,
        ckpt_every=0,
        use_scheduler=False,
        warmup_steps=0,
        out_csv=None,
        best_config_out=None,
        force_refresh=False,
    )

    # Suppress error logs from tj during this test
    caplog.set_level(logging.CRITICAL, logger="scripts.train_jepa")

    with pytest.raises(SystemExit) as ex:
        tj.cmd_grid_search(args)
    assert ex.value.code == 7
