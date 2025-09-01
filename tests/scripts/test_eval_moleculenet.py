import argparse
import sys
import types
from pathlib import Path

import pandas as pd

# Stub torch and dependent modules unconditionally
torch = types.ModuleType("torch")
torch.load = lambda *a, **k: {}
torch.device = lambda *a, **k: "cpu"
torch.cuda = types.SimpleNamespace(is_available=lambda: False)

nn_mod = types.ModuleType("torch.nn")
nn_mod.__path__ = []
nn_mod.Module = object
func_mod = types.ModuleType("torch.nn.functional")
nn_mod.functional = func_mod
torch.nn = nn_mod

sys.modules["torch"] = torch
sys.modules["torch.nn"] = nn_mod
sys.modules["torch.nn.functional"] = func_mod

stub_supervised = types.ModuleType("training.supervised")
stub_supervised.train_linear_head = lambda *a, **k: {}
sys.modules["training.supervised"] = stub_supervised

from scripts import eval_moleculenet as em  # noqa: E402


def test_evaluate_dataset_aggregates(monkeypatch):
    class DummyDS:
        graphs = [object()]
        smiles = ["C"]

    df = pd.DataFrame({"smiles": ["C"], "y": [1.0]})

    def fake_load(name, root):
        return DummyDS(), ["y"], df

    monkeypatch.setattr(em, "load_moleculenet_dataset", fake_load)
    monkeypatch.setattr(
        em, "train_linear_head", lambda *a, **k: {"roc_auc": 0.5, "head": object()}
    )
    monkeypatch.setattr(em, "set_seed", lambda *a, **k: None)

    encoder = object()
    res = em.evaluate_dataset("task", "classification", encoder, Path("."), "cpu", 1, 1)
    assert res["roc_auc_mean"] == 0.5
    assert res["roc_auc_std"] == 0.0


def test_main_writes_results(tmp_path, monkeypatch):
    args = argparse.Namespace(
        encoder_checkpoint="enc.pt",
        data_root=str(tmp_path),
        output=str(tmp_path / "out.csv"),
        device="cpu",
        input_dim=1,
        hidden_dim=2,
        num_layers=1,
        gnn_type="gcn",
        devices=1,
        patience=1,
    )
    monkeypatch.setattr(em, "parse_args", lambda: args)

    class DummyEncoder:
        def __init__(self, **kw):
            pass

        def load_state_dict(self, state):
            pass

        def to(self, device):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(em, "GNNEncoder", DummyEncoder)
    monkeypatch.setattr(em, "DATASETS", {"foo": "classification"})
    monkeypatch.setattr(em, "evaluate_dataset", lambda *a, **k: {"roc_auc_mean": 0.5})
    em.main()
    assert Path(args.output).exists()
