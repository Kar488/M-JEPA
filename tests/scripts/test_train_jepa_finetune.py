import argparse

import numpy as np
import pytest

import models.encoder  # noqa: F401
import models.factory  # noqa: F401
from scripts import train_jepa as tj

torch = pytest.importorskip("torch")


def make_args(tmp_path, seeds=None):
    """Create argument namespace for finetuning/evaluation."""
    enc_path = tmp_path / "encoder.pt"
    enc_path.write_text("stub")
    return argparse.Namespace(
        labeled_dir=str(tmp_path),
        encoder=str(enc_path),
        gnn_type="gcn",
        hidden_dim=16,
        num_layers=2,
        task_type="classification",
        epochs=1,
        batch_size=1,
        lr=0.001,
        patience=1,
        devices=1,
        device="cpu",
        ema_decay=0.99,
        seeds=seeds or [0],
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        add_3d=False,
        label_col="y",
    )


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


def test_cmd_finetune_aggregates_metrics(tmp_path, monkeypatch):
    calls = {
        "load_directory_dataset": 0,
        "build_encoder": 0,
        "train_linear_head": 0,
        "maybe_init_wandb": 0,
    }

    dataset = DummyDataset()

    def load_dataset_stub(path, label_col=None, add_3d=False, **kwargs):
        calls["load_directory_dataset"] += 1
        return dataset

    monkeypatch.setattr(tj, "load_directory_dataset", load_dataset_stub)

    class DummyEncoder:
        def load_state_dict(self, state):
            pass

    def build_encoder_stub(**kwargs):
        calls["build_encoder"] += 1
        return DummyEncoder()

    monkeypatch.setattr(tj, "build_encoder", build_encoder_stub)

    metric_values = [1.0, 2.0, 3.0]
    idx = {"i": 0}

    def train_linear_head_stub(**kwargs):
        calls["train_linear_head"] += 1
        val = metric_values[idx["i"]]
        idx["i"] += 1
        return {"acc": val}

    monkeypatch.setattr(tj, "train_linear_head", train_linear_head_stub)

    class DummyWB:
        def __init__(self):
            self.logs = []

        def log(self, data):
            self.logs.append(data)

        def finish(self):
            pass

    def maybe_init_wandb_stub(*args, **kwargs):
        calls["maybe_init_wandb"] += 1
        return DummyWB()

    monkeypatch.setattr(tj, "maybe_init_wandb", maybe_init_wandb_stub)

    captured_metrics = {}
    orig_aggregate = tj.aggregate_metrics

    def aggregate_stub(metrics_list):
        captured_metrics["list"] = metrics_list
        captured_metrics["out"] = orig_aggregate(metrics_list)
        return captured_metrics["out"]

    monkeypatch.setattr(tj, "aggregate_metrics", aggregate_stub)
    # forces scripts.train_jepa to see a harmless checkpoint dict during the test,
    # so cmd_finetune won’t choke on the "stub" file.
    monkeypatch.setattr(tj.torch, "load", lambda *a, **k: {"encoder": {}}, raising=True)

    args = make_args(tmp_path, seeds=[0, 1, 2])
    tj.cmd_finetune(args)

    assert calls["load_directory_dataset"] == 1
    assert calls["build_encoder"] == 3
    assert calls["train_linear_head"] == 3
    assert calls["maybe_init_wandb"] == 1

    metrics_list = captured_metrics["list"]
    assert [m["acc"] for m in metrics_list] == metric_values
    agg = captured_metrics["out"]
    assert np.isclose(agg["acc_mean"], np.mean(metric_values))
    assert np.isclose(agg["acc_std"], np.std(metric_values))


def test_cmd_evaluate_delegates_to_finetune(tmp_path, monkeypatch):
    called = {"finetune": 0}

    def finetune_stub(args):
        called["finetune"] += 1
        assert args.seeds == [0, 1]

    monkeypatch.setattr(tj, "cmd_finetune", finetune_stub)

    args = make_args(tmp_path, seeds=[0, 1])
    tj.cmd_evaluate(args)
    assert called["finetune"] == 1
