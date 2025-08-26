import sys
import types
import pandas as pd
import pytest
from dataclasses import dataclass


@pytest.fixture(autouse=True)
def _silence_tqdm(monkeypatch):
    """Make tqdm think we're not in a TTY so it doesn't draw a progress bar."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)


def test_run_ablation_without_from_smiles_list(monkeypatch):
    captured = {"jepa_dataset": None, "linear_datasets": []}

    # Stub required modules before importing experiments.ablation
    torch_stub = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", types.ModuleType("torch.nn"))
    monkeypatch.setitem(sys.modules, "torch.nn.functional", types.ModuleType("torch.nn.functional"))

    data_mod = types.ModuleType("data.mdataset")

    class DummyDataset:
        pass  # deliberately lacks from_smiles_list

    data_mod.GraphDataset = DummyDataset
    monkeypatch.setitem(sys.modules, "data.mdataset", data_mod)

    enc_mod = types.ModuleType("models.encoder")

    class DummyEncoder:
        def __init__(self, input_dim, hidden_dim, num_layers, gnn_type):
            pass

    enc_mod.GNNEncoder = DummyEncoder
    monkeypatch.setitem(sys.modules, "models.encoder", enc_mod)

    ema_mod = types.ModuleType("models.ema")

    class DummyEMA:
        def __init__(self, model, decay):
            pass

    ema_mod.EMA = DummyEMA
    monkeypatch.setitem(sys.modules, "models.ema", ema_mod)

    pred_mod = types.ModuleType("models.predictor")

    class DummyPredictor:
        def __init__(self, *, embed_dim, hidden_dim):
            pass

    pred_mod.MLPPredictor = DummyPredictor
    monkeypatch.setitem(sys.modules, "models.predictor", pred_mod)

    sup_mod = types.ModuleType("training.supervised")

    def stub_train_linear_head(*, dataset, **kwargs):
        captured["linear_datasets"].append(dataset)
        return {"roc_auc": 0.0, "pr_auc": 0.0, "rmse": 0.0, "mae": 0.0}

    sup_mod.train_linear_head = stub_train_linear_head
    monkeypatch.setitem(sys.modules, "training.supervised", sup_mod)

    unsup_mod = types.ModuleType("training.unsupervised")

    def stub_train_jepa(*, dataset, **kwargs):
        captured["jepa_dataset"] = dataset

    unsup_mod.train_jepa = stub_train_jepa
    monkeypatch.setitem(sys.modules, "training.unsupervised", unsup_mod)

    aug_mod = types.ModuleType("data.augment")

    @dataclass(frozen=True)
    class AugmentationConfig:
        random_rotate: bool = False
        mask_angle: bool = False
        perturb_dihedral: bool = False

    def iter_augmentation_options(*args, **kwargs):
        yield AugmentationConfig()

    aug_mod.AugmentationConfig = AugmentationConfig
    aug_mod.iter_augmentation_options = iter_augmentation_options
    monkeypatch.setitem(sys.modules, "data.augment", aug_mod)

    sys.modules.pop("experiments.ablation", None)
    import experiments.ablation as ablation

    df = ablation.run_ablation()
    assert isinstance(df, pd.DataFrame)
    assert captured["jepa_dataset"] is not None
    assert not isinstance(captured["jepa_dataset"], tuple)
    assert captured["linear_datasets"]
    assert all(not isinstance(ds, tuple) for ds in captured["linear_datasets"])

