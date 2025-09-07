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

    results = run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        pretrain_epochs=1,
        finetune_epochs=1,
        triage_pct=0.10,
    )
    assert all(isinstance(x, float) for x in results[:3])
    assert isinstance(results[3], dict)
