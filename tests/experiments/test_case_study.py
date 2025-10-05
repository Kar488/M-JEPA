import sys
import types

import torch


def test_tox21_case_study_smoke(monkeypatch):
    
    
    unsup = types.ModuleType("training.unsupervised")
    unsup.train_jepa = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "training.unsupervised", unsup)

    sup = types.ModuleType("training.supervised")

    def train_linear_head(*, dataset, encoder, task_type, epochs, lr, batch_size, device, patience, **kwargs):
        head = torch.nn.Linear(encoder.hidden_dim, 1)
        return {"head": head}

    sup.train_linear_head = train_linear_head
    monkeypatch.setitem(sys.modules, "training.supervised", sup)

    from experiments.case_study import run_tox21_case_study

    result = run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        pretrain_epochs=1,
        finetune_epochs=1,
        triage_pct=0.10,
    )
    assert result.evaluations, "Expected at least one evaluation"
    primary = result.evaluations[0]
    assert isinstance(primary.mean_true, float)
    assert isinstance(primary.mean_random, float)
    assert isinstance(primary.mean_pred, float)
    assert isinstance(primary.baseline_means, dict)
