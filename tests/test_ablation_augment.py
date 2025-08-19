import pandas as pd
import pytest
import sys

@pytest.fixture(autouse=True)
def _silence_tqdm(monkeypatch):
    """Make tqdm think we're not in a TTY so it doesn't draw a progress bar."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

def test_run_ablation_forwards_augment(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    from experiments import ablation
    from data.augment import AugmentationConfig

    calls = []

    def fake_train_jepa(*, random_rotate=False, mask_angle=False, perturb_dihedral=False, **kwargs):
        calls.append((random_rotate, mask_angle, perturb_dihedral))
        return []

    def fake_train_head(*args, **kwargs):
        return {"roc_auc": 0.0, "pr_auc": 0.0, "rmse": 0.0, "mae": 0.0}

    monkeypatch.setattr(ablation, "train_jepa", fake_train_jepa)
    monkeypatch.setattr(ablation, "train_linear_head", fake_train_head)

    df = ablation.run_ablation(
        augmentations=AugmentationConfig(rotate=True, mask_angle=True, dihedral=True)
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert calls and all(rr and ma and pd for rr, ma, pd in calls)