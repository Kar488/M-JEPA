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
