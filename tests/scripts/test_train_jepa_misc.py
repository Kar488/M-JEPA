import argparse
import logging

import pandas as pd
import pytest

import models.encoder  # noqa: F401
import models.factory  # noqa: F401
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
        gnn_type="gcn",
        hidden_dim=16,
        num_layers=2,
        device="cpu",
        use_wandb=True,
        wandb_project="test",
        wandb_tags=[],
        add_3d=False,
    )

    tj.cmd_benchmark(args)

    logs = holder["wb"].logs
    assert any(log.get("best_method") == "contrastive" for log in logs)


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
        gnn_type="gcn",
        hidden_dim=16,
        num_layers=2,
        device="cpu",
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        add_3d=False,
        ft_ckpt="ft.pt",
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
        triage_pct=0.10,
        calibrate=True,
        device="cpu",
        **kwargs,
    ):
        captures.update(
            {k: kwargs[k] for k in ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor", "bf16")}
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
        holder["wb"] = DummyWB()
        return holder["wb"]

    monkeypatch.setattr(tj, "maybe_init_wandb", maybe_init_wandb_stub)

    args = argparse.Namespace(
        csv=str(tmp_path / "tox.csv"),
        task="NR-AR",
        pretrain_epochs=1,
        finetune_epochs=1,
        triage_pct=0.10,
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
    )

    tj.cmd_tox21(args)

    logs = holder["wb"].logs
    assert any(
        log.get("phase") == "tox21"
        and log.get("status") == "success"
        and log.get("mean_true") == 0.3
        for log in logs
    )

    assert captures == {
        "num_workers": 4,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 5,
        "bf16": True,
    }


def test_cmd_tox21_failure(tmp_path,monkeypatch):
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
        pretrain_epochs=1,
        finetune_epochs=1,
        triage_pct=0.10,
        tox21_dir=str(tmp_path / "reports"),
        device="cpu",
        use_wandb=False,
        wandb_project="p",
        wandb_tags=[],
    )

    with pytest.raises(SystemExit) as ex:
        tj.cmd_tox21(args)
    assert ex.value.code == 5


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
