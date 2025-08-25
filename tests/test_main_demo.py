import pytest
import sys
import types

# Ensure torch is available; otherwise skip
torch = pytest.importorskip("torch")

# Provide a minimal stub for RDKit if it's not installed
if "rdkit" not in sys.modules:
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
if "torch_geometric" not in sys.modules:
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

import main


def test_demonstration_runs(monkeypatch):
    """Run the demo pipeline on a tiny dataset and ensure it completes."""

    # Avoid heavy case study by stubbing it out
    def _fake_case_study(*args, **kwargs):
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(main, "run_tox21_case_study", _fake_case_study)

    # The function uses a built-in synthetic dataset; just ensure no exception
    main.demonstration(device="cpu", devices=1, use_scaffold=False)