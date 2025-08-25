import pytest
import sys
import types

# Ensure torch is available; otherwise skip
torch = pytest.importorskip("torch")

# Provide a minimal stub for RDKit if it's not installed
try:  # pragma: no cover - optional dependency
    import rdkit  # noqa: F401
except Exception:  # pragma: no cover
    rdkit_stub = types.ModuleType("rdkit")
    chem_stub = types.ModuleType("Chem")
    scaffolds_stub = types.ModuleType("Scaffolds")
    scaffolds_stub.MurckoScaffold = types.SimpleNamespace(
        GetScaffoldForMol=lambda mol: None
    )
    chem_stub.Scaffolds = scaffolds_stub
    rdkit_stub.Chem = chem_stub
    sys.modules["rdkit"] = rdkit_stub
    sys.modules["rdkit.Chem"] = chem_stub
    sys.modules["rdkit.Chem.Scaffolds"] = scaffolds_stub

# Minimal stub for torch_geometric to satisfy optional imports
try:  # pragma: no cover - optional dependency
    import torch_geometric  # noqa: F401
except Exception:  # pragma: no cover
    tg_stub = types.ModuleType("torch_geometric")
    tg_stub.__path__ = []
    tg_data = types.ModuleType("data")
    class _DummyData:  # minimal placeholder
        pass
    tg_data.Data = _DummyData
    tg_loader = types.ModuleType("loader")
    class _DummyLoader:  # placeholder for DataLoader
        pass
    tg_loader.DataLoader = _DummyLoader
    tg_stub.data = tg_data
    tg_stub.loader = tg_loader
    sys.modules["torch_geometric"] = tg_stub
    sys.modules["torch_geometric.data"] = tg_data
    sys.modules["torch_geometric.loader"] = tg_loader

# Minimal stub for sklearn metrics if missing
try:  # pragma: no cover - optional dependency
    import sklearn  # noqa: F401
except Exception:  # pragma: no cover
    sk_stub = types.ModuleType("sklearn")
    metrics_stub = types.ModuleType("metrics")
    def _dummy_metric(*args, **kwargs):
        return 0.0
    for name in [
        "roc_auc_score",
        "average_precision_score",
        "brier_score_loss",
        "mean_absolute_error",
        "mean_squared_error",
        "r2_score",
    ]:
        setattr(metrics_stub, name, _dummy_metric)
    metrics_stub.__getattr__ = lambda name: _dummy_metric
    sk_stub.metrics = metrics_stub
    sys.modules["sklearn"] = sk_stub
    sys.modules["sklearn.metrics"] = metrics_stub

import main


def test_demonstration_runs(monkeypatch):
    """Run the demo pipeline on a tiny dataset and ensure it completes."""
 
    def _fake_train_jepa(*args, **kwargs):
        return [0.0]

    def _fake_train_contrastive(*args, **kwargs):
        return [0.0]

    def _fake_linear_head(*args, **kwargs):
        return {"acc": 0.0}

    def _fake_case_study(*args, **kwargs):
        return 0.0, 0.0, 0.0
    
    monkeypatch.setattr(main, "train_jepa", _fake_train_jepa)
    monkeypatch.setattr(main, "train_contrastive", _fake_train_contrastive)
    monkeypatch.setattr(main, "train_linear_head", _fake_linear_head)
    monkeypatch.setattr(main, "run_tox21_case_study", _fake_case_study)

    # The function uses a built-in synthetic dataset; just ensure no exception
    main.demonstration(device="cpu", devices=1, use_scaffold=False)