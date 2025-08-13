import argparse
import os
import sys
import types

# Minimal torch and model stubs for importing train_jepa without heavy deps
if "torch" not in sys.modules:
    torch_stub = types.SimpleNamespace(
        save=lambda obj, path: open(path, "wb").close(),
        load=lambda *args, **kwargs: {},
        manual_seed=lambda *args, **kwargs: None,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    sys.modules["torch"] = torch_stub

sys.modules.setdefault("models.factory", types.SimpleNamespace(build_encoder=lambda *a, **k: None))
sys.modules.setdefault("models.encoder", types.SimpleNamespace(GNNEncoder=object))

from scripts import train_jepa as tj


def make_args(tmp_path, contrastive=False):
    return argparse.Namespace(
        unlabeled_dir=str(tmp_path),
        gnn_type="gcn",
        hidden_dim=16,
        num_layers=2,
        mask_ratio=0.15,
        epochs=1,
        batch_size=1,
        lr=0.001,
        ema_decay=0.99,
        contrastive=contrastive,
        output=str(tmp_path / "encoder.pt"),
        contiguous=False,
        device="cpu",
        aug_rotate=False,
        aug_mask_angle=False,
        aug_dihedral=False,
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        add_3d=False,
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


def setup_stubs(monkeypatch, calls):
    dummy_dataset = DummyDataset()

    def load_dataset_stub(path, add_3d=False, **kwargs):
        calls["load_directory_dataset"] += 1
        return dummy_dataset

    monkeypatch.setattr(tj, "load_directory_dataset", load_dataset_stub)

    class DummyEncoder:
        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

    def build_encoder_stub(**kwargs):
        calls["build_encoder"] += 1
        return DummyEncoder()

    monkeypatch.setattr(tj, "build_encoder", build_encoder_stub)

    class DummyEMA:
        def __init__(self, model, decay):
            calls["EMA"] += 1

    monkeypatch.setattr(tj, "EMA", DummyEMA)

    class DummyPredictor:
        def __init__(self, embed_dim, hidden_dim):
            calls["MLPPredictor"] += 1

    monkeypatch.setattr(tj, "MLPPredictor", DummyPredictor)

    def train_jepa_stub(**kwargs):
        calls["train_jepa"] += 1

    monkeypatch.setattr(tj, "train_jepa", train_jepa_stub)

    def train_contrastive_stub(**kwargs):
        calls["train_contrastive"] += 1

    monkeypatch.setattr(tj, "train_contrastive", train_contrastive_stub)

    class DummyWB:
        def log(self, *args, **kwargs):
            pass

        def finish(self):
            pass

    def maybe_init_wandb_stub(*args, **kwargs):
        calls["maybe_init_wandb"] += 1
        return DummyWB()

    monkeypatch.setattr(tj, "maybe_init_wandb", maybe_init_wandb_stub)


def test_cmd_pretrain_creates_checkpoint_and_calls_training(tmp_path, monkeypatch):
    calls = {
        "load_directory_dataset": 0,
        "build_encoder": 0,
        "EMA": 0,
        "MLPPredictor": 0,
        "train_jepa": 0,
        "train_contrastive": 0,
        "maybe_init_wandb": 0,
    }
    setup_stubs(monkeypatch, calls)

    args = make_args(tmp_path, contrastive=False)
    tj.cmd_pretrain(args)

    assert calls["load_directory_dataset"] == 1
    assert calls["train_jepa"] == 1
    assert os.path.exists(args.output)


def test_cmd_pretrain_with_contrastive_branch(tmp_path, monkeypatch):
    calls = {
        "load_directory_dataset": 0,
        "build_encoder": 0,
        "EMA": 0,
        "MLPPredictor": 0,
        "train_jepa": 0,
        "train_contrastive": 0,
        "maybe_init_wandb": 0,
    }
    setup_stubs(monkeypatch, calls)

    args = make_args(tmp_path, contrastive=True)
    tj.cmd_pretrain(args)

    assert calls["load_directory_dataset"] == 1
    assert calls["train_jepa"] == 1
    assert calls["train_contrastive"] == 1
    assert os.path.exists(args.output)
    contrastive_path = tmp_path / "encoder_contrastive.pt"
    assert contrastive_path.exists()
