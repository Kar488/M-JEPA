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

