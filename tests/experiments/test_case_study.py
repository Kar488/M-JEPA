import sys
import types

import numpy as np
import pytest
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


def test_evaluate_case_study_handles_probability_mismatch(monkeypatch):
    import experiments.case_study as case_study

    dataset = types.SimpleNamespace(graphs=[types.SimpleNamespace(), types.SimpleNamespace()])
    labels = np.array([0.0, 1.0])

    def fake_predict(
        dataset,
        indices,
        encoder,
        head,
        device,
        edge_dim,
        batch_size=256,
        diag_hook=None,
    ):
        return torch.zeros((1, 1)), torch.zeros((1, 1))

    monkeypatch.setattr(case_study, "_predict_logits_probs_in_chunks", fake_predict)

    def fake_resize(arr, new_len):
        return np.array([], dtype=getattr(arr, "dtype", float))

    monkeypatch.setattr(case_study.np, "resize", fake_resize)

    mean_true, mean_rand, mean_pred, baselines, metrics = case_study._evaluate_case_study(
        dataset=dataset,
        encoder=None,
        head=None,
        all_labels=labels,
        train_idx=[0],
        val_idx=[0, 1],
        test_idx=[0, 1],
        triage_pct=0.1,
        calibrate=False,
        device="cpu",
        edge_dim=0,
        seed=0,
        baseline_embeddings=None,
    )

    assert mean_true == pytest.approx(0.5)
    assert mean_rand == pytest.approx(0.0)
    assert mean_pred == pytest.approx(1.0)
    assert baselines == {}


def test_evaluate_case_study_handles_resize_failure(monkeypatch):
    import experiments.case_study as case_study

    dataset = types.SimpleNamespace(graphs=[types.SimpleNamespace(), types.SimpleNamespace()])
    labels = np.array([0.0, 1.0])

    def fake_predict(
        dataset,
        indices,
        encoder,
        head,
        device,
        edge_dim,
        batch_size=256,
        diag_hook=None,
    ):
        # Return logits/probabilities shorter than requested to trigger the resize path.
        return torch.zeros((1, 1)), torch.zeros((1, 1))

    monkeypatch.setattr(case_study, "_predict_logits_probs_in_chunks", fake_predict)

    def failing_resize(arr, new_len):
        raise ValueError("boom")

    monkeypatch.setattr(case_study.np, "resize", failing_resize)

    mean_true, mean_rand, mean_pred, baselines, metrics = case_study._evaluate_case_study(
        dataset=dataset,
        encoder=None,
        head=None,
        all_labels=labels,
        train_idx=[0],
        val_idx=[0, 1],
        test_idx=[0, 1],
        triage_pct=0.1,
        calibrate=False,
        device="cpu",
        edge_dim=0,
        seed=0,
        baseline_embeddings=None,
    )

    assert mean_true == pytest.approx(0.5)
    assert mean_rand == pytest.approx(0.0)
    assert mean_pred == pytest.approx(1.0)
    assert baselines == {}
    assert metrics["roc_auc"] == pytest.approx(0.5)


def test_evaluate_case_study_handles_empty_predictions(monkeypatch):
    import experiments.case_study as case_study

    dataset = types.SimpleNamespace(graphs=[types.SimpleNamespace(), types.SimpleNamespace()])
    labels = np.array([0.0, 1.0])

    def fake_predict(
        dataset,
        indices,
        encoder,
        head,
        device,
        edge_dim,
        batch_size=256,
        diag_hook=None,
    ):
        return torch.empty((0, 1)), torch.empty((0, 1))

    monkeypatch.setattr(case_study, "_predict_logits_probs_in_chunks", fake_predict)

    mean_true, mean_rand, mean_pred, baselines, metrics = case_study._evaluate_case_study(
        dataset=dataset,
        encoder=None,
        head=None,
        all_labels=labels,
        train_idx=[0],
        val_idx=[0, 1],
        test_idx=[0, 1],
        triage_pct=0.5,
        calibrate=False,
        device="cpu",
        edge_dim=0,
        seed=0,
        baseline_embeddings=None,
    )

    assert mean_true == pytest.approx(0.5)
    assert mean_rand == pytest.approx(0.0)
    assert mean_pred == pytest.approx(1.0)
    assert baselines == {}
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(0.5)
    assert metrics["brier"] == pytest.approx(0.5)
    assert metrics["ece"] == pytest.approx(0.5)
