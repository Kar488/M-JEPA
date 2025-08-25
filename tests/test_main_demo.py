import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")
pytest.importorskip("torch_geometric")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

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

    main.demonstration(device="cpu", devices=1, use_scaffold=False)
