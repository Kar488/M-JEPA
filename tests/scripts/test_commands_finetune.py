import sys
import types
from argparse import Namespace

import numpy as np
import pytest

import scripts.commands.finetune as finetune
from data.mdataset import GraphData, GraphDataset

torch = pytest.importorskip("torch")


def test_cmd_evaluate_delegates(monkeypatch):
    called = {}

    def fake(args):
        called["called"] = True

    monkeypatch.setattr(finetune, "cmd_finetune", fake)
    finetune.cmd_evaluate(Namespace())
    assert called.get("called")


def test_evaluate_finetuned_head_missing_keys(monkeypatch):
    state = {}
    from utils import checkpoint as ckpt_mod

    monkeypatch.setattr(ckpt_mod, "load_checkpoint", lambda p: state)
    args = Namespace(task_type="classification", batch_size=1, device="cpu")
    g = GraphData(
        x=np.ones((1, 1), dtype=np.float32), edge_index=np.zeros((2, 0), dtype=np.int64)
    )
    ds = GraphDataset([g], labels=[1.0])
    result = finetune.evaluate_finetuned_head("dummy", ds, args, device="cpu")
    assert result == {}


def test_evaluate_finetuned_head_classification(monkeypatch):
    g = GraphData(
        x=np.ones((2, 1), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    ds = GraphDataset([g], labels=[1.0])
    state = {"encoder": {}, "head": {}}

    def load_checkpoint(path):
        return state

    def load_state_dict_forgiving(model, state):
        return None

    def build_encoder(**cfg):
        class Dummy(torch.nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.hidden_dim = hidden_dim

            def state_dict(self, *a, **k):
                return {}

        return Dummy(cfg.get("hidden_dim", 4))

    def global_mean_pool(x, batch_ptr):
        return x.mean(dim=0, keepdim=True)

    def compute_classification_metrics(y_true, y_pred):
        return {"acc": float(y_pred.mean())}

    def compute_regression_metrics(y_true, y_pred):
        return {"mse": float(((y_true - y_pred) ** 2).mean())}

    monkeypatch.setitem(
        sys.modules,
        "utils.checkpoint",
        types.SimpleNamespace(
            load_checkpoint=load_checkpoint,
            load_state_dict_forgiving=load_state_dict_forgiving,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "models.factory",
        types.SimpleNamespace(build_encoder=build_encoder),
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.pooling",
        types.SimpleNamespace(global_mean_pool=global_mean_pool),
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.metrics",
        types.SimpleNamespace(
            compute_classification_metrics=compute_classification_metrics,
            compute_regression_metrics=compute_regression_metrics,
        ),
    )
    monkeypatch.setattr(
        finetune,
        "_encode_graph",
        lambda enc, g: torch.ones((g.x.shape[0], enc.hidden_dim)),
    )

    args = Namespace(
        task_type="classification",
        batch_size=1,
        device="cpu",
        hidden_dim=4,
        num_layers=1,
        gnn_type="mpnn",
        add_3d=False,
    )
    result = finetune.evaluate_finetuned_head("ckpt.pt", ds, args, device="cpu")
    assert "acc" in result


def test_maybe_enable_add_3d_sets_flag():
    args = Namespace(gnn_type="SchNet3D", add_3d=False)
    requires = finetune._maybe_enable_add_3d(args)
    assert requires is True
    assert args.add_3d is True


def test_finetune_epoch_warning_uses_nonzero_batches(monkeypatch):
    monkeypatch.setattr(finetune, "get_rank", lambda: 0)
    monkeypatch.setattr(finetune, "get_world_size", lambda: 4)
    msg = finetune._build_finetune_epoch_batch_warning(
        epoch_batches=0,
        train_batches=6,
        max_finetune_batches=0,
        seed=7,
        epoch=2,
    )
    assert msg is not None
    assert "only 0" not in msg
    assert "only 6 per-rank batches" in msg
    assert "rank=0/4" in msg


def test_finetune_epoch_warning_triggers_on_zero_batches(monkeypatch):
    monkeypatch.setattr(finetune, "get_rank", lambda: 1)
    monkeypatch.setattr(finetune, "get_world_size", lambda: 4)
    msg = finetune._build_finetune_epoch_batch_warning(
        epoch_batches=0,
        train_batches=0,
        max_finetune_batches=0,
        seed=3,
        epoch=1,
    )
    assert msg is not None
    assert "only 0 per-rank batches" in msg
    assert "rank=1/4" in msg


def test_ensure_dataset_has_pos_raises_for_missing_coords():
    graphs = [
        GraphData(
            x=np.ones((1, 1), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            pos=None,
        )
    ]

    class DummyDataset:
        def __init__(self, graphs):
            self.graphs = graphs

    with pytest.raises(ValueError):
        finetune._ensure_dataset_has_pos(DummyDataset(graphs))


def test_ensure_dataset_has_pos_skips_empty_graphs():
    graphs = [
        GraphData(
            x=np.zeros((0, 1), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            pos=None,
        ),
        GraphData(
            x=np.ones((1, 1), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            pos=np.zeros((1, 3), dtype=np.float32),
        ),
    ]

    class DummyDataset:
        def __init__(self, graphs):
            self.graphs = graphs

    # Should not raise because the first non-empty graph has coordinates.
    finetune._ensure_dataset_has_pos(DummyDataset(graphs))


def test_cmd_finetune_aggregates_metrics(monkeypatch, tmp_path):
    graph = GraphData(
        x=np.ones((2, 1), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    dataset = GraphDataset([graph] * 4, labels=[1.0, 0.0, 1.0, 0.0])

    monkeypatch.setattr(
        finetune,
        "load_directory_dataset",
        lambda *a, **k: dataset,
        raising=False,
    )
    monkeypatch.setattr(
        finetune,
        "_resolve_labeled_dataset_source",
        lambda *a, **k: str(tmp_path),
        raising=False,
    )
    monkeypatch.setattr(
        finetune,
        "_sanitize_dataset_labels",
        lambda ds: (ds, {}),
        raising=False,
    )
    monkeypatch.setattr(
        finetune,
        "resolve_device",
        lambda device: "cpu",
        raising=False,
    )
    monkeypatch.setattr(
        finetune,
        "_maybe_to",
        lambda model, device: model,
        raising=False,
    )
    monkeypatch.setattr(
        finetune,
        "_maybe_labels",
        lambda ds: getattr(ds, "labels", None),
        raising=False,
    )
    monkeypatch.setattr(
        finetune,
        "_infer_num_classes",
        lambda ds: 2,
        raising=False,
    )
    monkeypatch.setattr(
        finetune,
        "_maybe_state_dict",
        lambda obj: obj.state_dict() if hasattr(obj, "state_dict") else None,
        raising=False,
    )

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = 2
            self.layer = torch.nn.Linear(1, 2)

    monkeypatch.setattr(
        finetune,
        "build_encoder",
        lambda **k: DummyEncoder(),
        raising=False,
    )
    monkeypatch.setattr(
        finetune,
        "build_linear_head",
        lambda in_dim, num_classes, task_type: torch.nn.Linear(
            int(in_dim), int(num_classes)
        ),
        raising=False,
    )

    metric_values = [0.6, 0.8]
    calls = {"i": 0}

    def train_linear_head_stub(**kwargs):
        val = metric_values[calls["i"]]
        calls["i"] += 1
        return {
            "acc": val,
            "train/batches": 1.0,
            "train/epoch_batches": 1.0,
        }

    monkeypatch.setattr(
        finetune,
        "train_linear_head",
        train_linear_head_stub,
        raising=False,
    )

    captured = {}

    def aggregate_metrics_stub(metrics_list):
        captured["list"] = metrics_list
        return {"acc_mean": float(np.mean([m["acc"] for m in metrics_list]))}

    monkeypatch.setattr(
        finetune,
        "aggregate_metrics",
        aggregate_metrics_stub,
        raising=False,
    )

    wb_holder = {}

    class DummyWB:
        def __init__(self):
            self.logs = []
            self.config = {}

        def log(self, data):
            self.logs.append(data)

        def finish(self):
            pass

    def maybe_init_wandb_stub(*args, **kwargs):
        wb_holder["wb"] = DummyWB()
        return wb_holder["wb"]

    monkeypatch.setattr(
        finetune,
        "maybe_init_wandb",
        maybe_init_wandb_stub,
        raising=False,
    )

    import utils.checkpoint as ckpt_mod

    monkeypatch.setattr(
        ckpt_mod,
        "compute_state_dict_hash",
        lambda state: "hash",
        raising=False,
    )
    monkeypatch.setattr(
        ckpt_mod,
        "load_checkpoint",
        lambda *a, **k: {},
        raising=False,
    )
    monkeypatch.setattr(
        ckpt_mod,
        "save_checkpoint",
        lambda *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(
        ckpt_mod,
        "safe_load_checkpoint",
        lambda *a, **k: ({}, None),
        raising=False,
    )
    monkeypatch.setattr(
        ckpt_mod,
        "load_state_dict_forgiving",
        lambda *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(
        ckpt_mod,
        "safe_link_or_copy",
        lambda *a, **k: "copy",
        raising=False,
    )

    args = Namespace(
        labeled_dir=str(tmp_path),
        labeled_csv=None,
        gnn_type="mpnn",
        hidden_dim=2,
        num_layers=1,
        dropout=None,
        task_type="classification",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        encoder_lr=None,
        head_lr=None,
        ema_decay=0.0,
        seeds=[0, 1],
        add_3d=False,
        freeze_encoder=True,
        unfreeze_top_layers=0,
        max_finetune_batches=0,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        use_scaffold=False,
        use_focal_loss=False,
        dynamic_pos_weight=False,
        oversample_minority=False,
        pos_weight=None,
        layerwise_decay=None,
        calibrate_probabilities=False,
        threshold_metric=None,
        use_wandb=True,
        wandb_project="proj",
        wandb_tags=[],
        device="cpu",
        label_col="label",
        patience=1,
        devices=1,
        encoder=None,
        load_encoder_checkpoint=None,
        unfreeze_mode="none",
        save_every=1,
        time_budget_mins=0,
        cache_dir=None,
        max_pretrain_batches=0,
        class_weight=None,
        focal_gamma=2.0,
        calibration_method="temperature",
        per_task_hparams=None,
        resume_ckpt=None,
        metric="acc",
    )

    finetune._cmd_finetune_single(args)

    assert [m["acc"] for m in captured["list"]] == metric_values
    wb_logs = wb_holder["wb"].logs
    assert any("metric/acc_mean" in entry for entry in wb_logs)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 2)
        self.layer2 = torch.nn.Linear(2, 1)


def test_maybe_enable_add_3d_enables_flag(caplog):
    args = types.SimpleNamespace(gnn_type="SchNet3D", add_3d=False)
    requires = finetune._maybe_enable_add_3d(args)
    assert requires is True
    assert args.add_3d is True


def test_ensure_dataset_has_pos_validates_missing():
    graph = types.SimpleNamespace(x=torch.ones(2, 3), edge_index=torch.tensor([[0, 1], [1, 0]]), pos=None)
    dataset = types.SimpleNamespace(graphs=[graph])
    with pytest.raises(ValueError):
        finetune._ensure_dataset_has_pos(dataset)

    graph2 = types.SimpleNamespace(pos=torch.ones(2, 3))
    dataset2 = types.SimpleNamespace(graphs=[graph2])
    finetune._ensure_dataset_has_pos(dataset2)


def test_iter_trainable_params_and_configure_encoder():
    encoder = TinyEncoder()
    params = finetune._iter_trainable_params(encoder)
    assert len(params) == 4  # weights and biases

    # Freeze everything and unfreeze last layer
    trainable = finetune._configure_encoder_trainability(
        encoder,
        freeze_encoder=True,
        unfreeze_top_layers=1,
    )
    assert all(not p.requires_grad for p in encoder.parameters()) is False
    assert len(trainable) > 0

    # No freeze requested -> all params trainable
    encoder2 = TinyEncoder()
    all_params = finetune._configure_encoder_trainability(
        encoder2,
        freeze_encoder=False,
        unfreeze_top_layers=0,
    )
    assert all(p.requires_grad for p in all_params)
