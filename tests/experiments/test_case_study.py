import logging
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

    mean_true, mean_rand, mean_pred, baselines, metrics, calibrator = case_study._evaluate_case_study(
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

    mean_true, mean_rand, mean_pred, baselines, metrics, calibrator = case_study._evaluate_case_study(
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

    mean_true, mean_rand, mean_pred, baselines, metrics, calibrator = case_study._evaluate_case_study(
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


def test_case_study_trains_head_when_missing(tmp_path, monkeypatch, caplog):
    import experiments.case_study as case_study

    dummy_ckpt = tmp_path / "ft_encoder.pt"
    dummy_ckpt.write_text("placeholder", encoding="utf-8")

    def fake_safe_load_checkpoint(primary, **_kwargs):
        return {"encoder": {}}, {}

    monkeypatch.setattr(case_study, "safe_load_checkpoint", fake_safe_load_checkpoint)
    monkeypatch.setattr(case_study, "load_state_dict_forgiving", lambda *args, **kwargs: None)

    train_calls: dict[str, object] = {}

    def fake_train_linear_head(*, dataset, encoder, epochs, freeze_encoder, patience, **kwargs):
        train_calls["called"] = True
        train_calls["epochs"] = epochs
        train_calls["freeze_encoder"] = freeze_encoder
        train_calls["patience"] = patience
        head = torch.nn.Linear(getattr(encoder, "hidden_dim", 32), 1)
        return {"head": head, "train/batches": 8.0}

    monkeypatch.setattr(case_study, "train_linear_head", fake_train_linear_head)

    caplog.set_level(logging.INFO, logger=case_study.logger.name)

    result = case_study.run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        finetune_epochs=3,
        encoder_checkpoint=str(dummy_ckpt),
        evaluation_mode="fine_tuned",
    )

    assert result.evaluations, "expected evaluations to be produced"
    assert train_calls.get("called") is True
    assert train_calls.get("epochs") == 3
    assert train_calls.get("freeze_encoder") is True
    assert train_calls.get("patience") == 10
    assert any("train_head=yes" in message for message in caplog.messages)
    assert any("head_trained=yes" in message for message in caplog.messages)


def test_case_study_frozen_finetuned_trains_linear_probe(monkeypatch, caplog):
    import experiments.case_study as case_study

    def fake_safe_load_checkpoint(primary, **_kwargs):
        return {"encoder": {}}, {}

    monkeypatch.setattr(case_study, "safe_load_checkpoint", fake_safe_load_checkpoint)
    monkeypatch.setattr(case_study, "load_state_dict_forgiving", lambda *args, **kwargs: None)

    calls: dict[str, object] = {}

    def fake_train_linear_head(*, dataset, encoder, freeze_encoder, **kwargs):
        calls["called"] = True
        calls["freeze_encoder"] = freeze_encoder
        params = list(getattr(encoder, "parameters", lambda: [])())
        calls["encoder_requires_grad"] = [getattr(p, "requires_grad", None) for p in params]
        head = torch.nn.Linear(getattr(encoder, "hidden_dim", 32), 1)
        return {"head": head, "train/batches": 4.0}

    monkeypatch.setattr(case_study, "train_linear_head", fake_train_linear_head)

    caplog.set_level(logging.INFO, logger=case_study.logger.name)

    case_study.run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        encoder_checkpoint="dummy_ft.pt",
        evaluation_mode="frozen_finetuned",
    )

    assert calls.get("called") is True
    assert calls.get("freeze_encoder") is True
    encoder_grad_flags = calls.get("encoder_requires_grad") or []
    assert encoder_grad_flags and all(flag is False for flag in encoder_grad_flags)
    assert any("train_head=yes" in message for message in caplog.messages)


def test_case_study_passes_head_lr_and_weight_decay(tmp_path, monkeypatch):
    import experiments.case_study as case_study

    dummy_ckpt = tmp_path / "encoder.pt"
    dummy_ckpt.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(case_study, "safe_load_checkpoint", lambda *args, **kwargs: ({"encoder": {}}, {}))
    monkeypatch.setattr(case_study, "_load_encoder_strict", lambda *args, **kwargs: {"hash": "stub", "matched_ratio": 1.0})
    monkeypatch.setattr(case_study, "load_state_dict_forgiving", lambda *args, **kwargs: None)

    captured: dict[str, object] = {}

    def fake_train_linear_head(*, dataset, encoder, lr, head_lr, encoder_lr, optimizer=None, head=None, **kwargs):
        captured["lr"] = lr
        captured["head_lr"] = head_lr
        captured["encoder_lr"] = encoder_lr
        captured["optimizer"] = optimizer
        captured["head"] = head
        head_module = head or torch.nn.Linear(getattr(encoder, "hidden_dim", 32), 1)
        return {"head": head_module, "train/batches": 1.0}

    monkeypatch.setattr(case_study, "train_linear_head", fake_train_linear_head)

    result = case_study.run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        encoder_checkpoint=str(dummy_ckpt),
        evaluation_mode="frozen_finetuned",
        head_lr=2e-3,
        encoder_lr=5e-4,
        weight_decay=1e-2,
    )

    assert result.evaluations, "expected evaluations to be produced"
    assert captured["lr"] == pytest.approx(2e-3)
    assert captured["head_lr"] == pytest.approx(2e-3)
    assert captured["encoder_lr"] == pytest.approx(5e-4)
    assert isinstance(captured["optimizer"], torch.optim.AdamW)
    assert isinstance(captured["head"], torch.nn.Module)
    opt = captured["optimizer"]
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-2)
