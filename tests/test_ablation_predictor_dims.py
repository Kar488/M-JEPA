import sys
import types
import pandas as pd
import pytest

@pytest.fixture(autouse=True)
def _silence_tqdm(monkeypatch):
    """Make tqdm think we're not in a TTY so it doesn't draw a progress bar."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

def test_run_ablation_uses_int_dims(monkeypatch):
    captured = []

    # Stub modules required by experiments.ablation before import
    torch_stub = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", types.ModuleType("torch.nn"))
    monkeypatch.setitem(
        sys.modules, "torch.nn.functional", types.ModuleType("torch.nn.functional")
    )

    data_mod = types.ModuleType("data.mdataset")
    class DummyDataset:
        @classmethod
        def from_smiles_list(cls, smiles_list, labels):
            return cls()
    data_mod.GraphDataset = DummyDataset
    monkeypatch.setitem(sys.modules, "data.mdataset", data_mod)

    enc_mod = types.ModuleType("models.encoder")
    class DummyEncoder:
        def __init__(self, input_dim, hidden_dim, num_layers, gnn_type):
            self.hidden_dim = hidden_dim
    enc_mod.GNNEncoder = DummyEncoder
    monkeypatch.setitem(sys.modules, "models.encoder", enc_mod)

    ema_mod = types.ModuleType("models.ema")
    class DummyEMA:
        def __init__(self, model, decay):
            self.model = model
            self.decay = decay
    ema_mod.EMA = DummyEMA
    monkeypatch.setitem(sys.modules, "models.ema", ema_mod)

    pred_mod = types.ModuleType("models.predictor")
    class DummyPredictor:
        def __init__(self, *, embed_dim, hidden_dim):
            captured.append((embed_dim, hidden_dim))
    pred_mod.MLPPredictor = DummyPredictor
    monkeypatch.setitem(sys.modules, "models.predictor", pred_mod)

    sup_mod = types.ModuleType("training.supervised")
    sup_mod.train_linear_head = lambda *a, **k: {
        "roc_auc": 0.0,
        "pr_auc": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
    }
    monkeypatch.setitem(sys.modules, "training.supervised", sup_mod)

    unsup_mod = types.ModuleType("training.unsupervised")
    unsup_mod.train_jepa = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "training.unsupervised", unsup_mod)

    import experiments.ablation as ablation

    df = ablation.run_ablation()
    assert isinstance(df, pd.DataFrame)
    assert captured, "Predictor was never instantiated"
    assert all(isinstance(e, int) and isinstance(h, int) for e, h in captured)