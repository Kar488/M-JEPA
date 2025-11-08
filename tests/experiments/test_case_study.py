import logging
import math
import sys
import types
from typing import Any, Dict

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


def test_evaluate_case_study_accepts_single_task_multilogit(monkeypatch):
    import experiments.case_study as case_study

    dataset = types.SimpleNamespace(
        graphs=[types.SimpleNamespace(), types.SimpleNamespace(), types.SimpleNamespace()]
    )
    labels = np.array([0.0, 1.0, 0.0])

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
        if len(indices) == 2:
            logits = torch.tensor(
                [[[0.2, -0.1]], [[-0.3, 0.4]]], dtype=torch.float32
            )  # (batch, 1, 2)
        else:
            logits = torch.tensor([[[0.1, -0.2]]], dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    monkeypatch.setattr(case_study, "_predict_logits_probs_in_chunks", fake_predict)

    mean_true, mean_rand, mean_pred, baselines, metrics, calibrator = case_study._evaluate_case_study(
        dataset=dataset,
        encoder=None,
        head=None,
        all_labels=labels,
        train_idx=[0],
        val_idx=[0, 1],
        test_idx=[2],
        triage_pct=0.1,
        calibrate=False,
        device="cpu",
        edge_dim=0,
        seed=0,
        baseline_embeddings=None,
    )

    assert mean_true == pytest.approx(0.0)
    assert baselines == {}
    assert np.isnan(metrics["roc_auc"])
    assert calibrator["feature_dim"] == 1
    assert calibrator["enabled"] is False


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


def test_evaluate_case_study_multitask_two_logits(monkeypatch):
    dataset = types.SimpleNamespace(graphs=[types.SimpleNamespace() for _ in range(3)])
    labels = np.array([0.0, 1.0, 0.0])

    training_pkg = types.ModuleType("training")
    unsup = types.ModuleType("training.unsupervised")
    unsup.train_jepa = lambda *args, **kwargs: {}

    sup = types.ModuleType("training.supervised")

    def _train_linear_head(**_kwargs):
        return {"head": torch.nn.Linear(1, 1)}

    sup.train_linear_head = _train_linear_head

    monkeypatch.setitem(sys.modules, "training", training_pkg)
    monkeypatch.setitem(sys.modules, "training.unsupervised", unsup)
    monkeypatch.setitem(sys.modules, "training.supervised", sup)

    import experiments.case_study as case_study

    val_logits = torch.tensor(
        [[[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]]],
        dtype=torch.float32,
    )
    val_probs = torch.tensor(
        [[[0.4, 0.6], [0.3, 0.7], [0.2, 0.8]]],
        dtype=torch.float32,
    )
    test_logits = torch.tensor(
        [[[0.0, 0.5], [0.5, 2.0], [1.0, 3.5]]],
        dtype=torch.float32,
    )
    test_probs = torch.tensor(
        [[[0.45, 0.55], [0.35, 0.65], [0.25, 0.75]]],
        dtype=torch.float32,
    )

    outputs = [(val_logits, val_probs), (test_logits, test_probs)]

    def fake_predict(*args, **kwargs):
        try:
            logits, probs = outputs.pop(0)
        except IndexError:  # pragma: no cover - defensive fallback
            logits = torch.empty((0, 2))
            probs = torch.empty((0, 2))
        return logits, probs

    monkeypatch.setattr(case_study, "_predict_logits_probs_in_chunks", fake_predict)

    class DummyLR:
        fit_args: dict[str, np.ndarray] = {}
        predict_args: dict[str, np.ndarray] = {}

        def __init__(self, *args, **kwargs):
            self.coef_ = np.zeros((1, 1))
            self.intercept_ = np.zeros(1)

        def fit(self, X, y):
            DummyLR.fit_args = {"X": X.copy(), "y": y.copy()}
            self.coef_ = np.ones((1, X.shape[1]))
            self.intercept_ = np.zeros(1)
            return self

        def predict_proba(self, X):
            DummyLR.predict_args = {"X": X.copy()}
            probs = np.full((X.shape[0], 2), 0.2, dtype=float)
            probs[:, 1] = 0.8
            probs[:, 0] = 0.2
            return probs

    monkeypatch.setattr(case_study, "LogisticRegression", DummyLR)

    mean_true, mean_rand, mean_pred, baselines, metrics, calibrator = case_study._evaluate_case_study(
        dataset=dataset,
        encoder=None,
        head=None,
        all_labels=labels,
        train_idx=[0],
        val_idx=[0, 1, 2],
        test_idx=[0, 1, 2],
        triage_pct=0.0,
        calibrate=True,
        device="cpu",
        edge_dim=0,
        seed=0,
        baseline_embeddings=None,
    )

    assert calibrator["feature_dim"] == 1
    assert calibrator["status"] == "fitted"
    assert calibrator["type"] == "platt"
    assert np.allclose(DummyLR.fit_args["X"].reshape(-1), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(DummyLR.fit_args["y"], np.array([0, 1, 0]))
    assert np.allclose(DummyLR.predict_args["X"].reshape(-1), np.array([0.5, 1.5, 2.5]))
    assert baselines == {}


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


def test_case_study_end_to_end_without_checkpoint_enables_full_finetune(monkeypatch):
    unsup = types.ModuleType("training.unsupervised")
    unsup.train_jepa = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "training.unsupervised", unsup)

    sup_module = types.ModuleType("training.supervised")
    sup_module.train_linear_head = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "training.supervised", sup_module)

    import importlib

    case_study = importlib.import_module("experiments.case_study")

    calls: dict[str, object] = {}

    def fake_train_linear_head(*, dataset, encoder, freeze_encoder, encoder_lr, **kwargs):
        calls["freeze_encoder"] = freeze_encoder
        calls["encoder_lr"] = encoder_lr
        head = torch.nn.Linear(getattr(encoder, "hidden_dim", 32), 1)
        return {"head": head, "train/batches": 4.0}

    monkeypatch.setattr(case_study, "train_linear_head", fake_train_linear_head)

    result = case_study.run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        finetune_epochs=2,
        evaluation_mode="fine_tuned",
    )

    assert calls.get("freeze_encoder") is False
    assert calls.get("encoder_lr") is not None
    assert isinstance(result.diagnostics, dict)
    assert result.diagnostics.get("auto_full_finetune") is True
    assert result.diagnostics.get("full_finetune") is True


def test_case_study_end_to_end_with_checkpoint_enables_full_finetune(monkeypatch):
    unsup = types.ModuleType("training.unsupervised")
    unsup.train_jepa = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "training.unsupervised", unsup)

    sup_module = types.ModuleType("training.supervised")
    calls: dict[str, Any] = {}

    def fake_train_linear_head(*, dataset, encoder, freeze_encoder, encoder_lr, **kwargs):
        calls["freeze_encoder"] = freeze_encoder
        calls["encoder_lr"] = encoder_lr
        head = torch.nn.Linear(getattr(encoder, "hidden_dim", 32), 1)
        return {"head": head, "train/batches": 4.0}

    sup_module.train_linear_head = fake_train_linear_head
    monkeypatch.setitem(sys.modules, "training.supervised", sup_module)

    import experiments.case_study as case_study

    monkeypatch.setattr(
        case_study,
        "safe_load_checkpoint",
        lambda *args, **kwargs: ({"encoder": {}}, {}),
        raising=False,
    )
    monkeypatch.setattr(
        case_study,
        "_load_encoder_strict",
        lambda *args, **kwargs: {"hash": "stub"},
        raising=False,
    )
    monkeypatch.setattr(case_study, "load_state_dict_forgiving", lambda *a, **k: None, raising=False)

    result = case_study.run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        finetune_epochs=2,
        encoder_checkpoint="dummy.pt",
        evaluation_mode="end_to_end",
    )

    assert calls.get("freeze_encoder") is False
    assert calls.get("encoder_lr") is not None
    assert result.diagnostics.get("auto_full_finetune") is True
    assert result.diagnostics.get("full_finetune") is True


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


def test_case_study_pos_class_weight_override(monkeypatch):
    import experiments.case_study as case_study

    monkeypatch.setattr(
        case_study,
        "safe_load_checkpoint",
        lambda *args, **kwargs: ({"encoder": {}}, {}),
    )
    monkeypatch.setattr(
        case_study,
        "_load_encoder_strict",
        lambda *args, **kwargs: {"hash": "stub", "matched_ratio": 1.0},
    )
    monkeypatch.setattr(case_study, "load_state_dict_forgiving", lambda *a, **k: None)

    captured: Dict[str, Any] = {}

    def fake_train_linear_head(*, class_weight=None, **kwargs):
        captured["class_weight"] = class_weight
        head = torch.nn.Linear(256, 1)
        return {"head": head, "train/batches": 1.0}

    monkeypatch.setattr(case_study, "train_linear_head", fake_train_linear_head)

    result = case_study.run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        encoder_checkpoint="stub.pt",
        evaluation_mode="frozen_finetuned",
        pos_class_weight=6.0,
        finetune_epochs=1,
    )

    weights = captured.get("class_weight")
    assert weights is not None
    assert weights[1] == pytest.approx(6.0)
    assert result.diagnostics["class_weight_manual"][1] == pytest.approx(6.0)


def test_case_study_freeze_encoder_overrides_full_finetune(monkeypatch):
    import experiments.case_study as case_study

    monkeypatch.setattr(
        case_study,
        "safe_load_checkpoint",
        lambda *args, **kwargs: ({"encoder": {}}, {}),
    )
    monkeypatch.setattr(
        case_study,
        "_load_encoder_strict",
        lambda *args, **kwargs: {"hash": "stub", "matched_ratio": 1.0},
    )
    monkeypatch.setattr(case_study, "load_state_dict_forgiving", lambda *a, **k: None)

    def fake_train_linear_head(*args, **kwargs):
        head = torch.nn.Linear(256, 1)
        return {"head": head, "train/batches": 1.0}

    monkeypatch.setattr(case_study, "train_linear_head", fake_train_linear_head)

    result = case_study.run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        encoder_checkpoint="stub.pt",
        evaluation_mode="fine_tuned",
        full_finetune=True,
        freeze_encoder=True,
        finetune_epochs=1,
    )

    assert result.diagnostics["full_finetune_requested"] is True
    assert result.diagnostics["full_finetune"] is False
    assert result.diagnostics["freeze_encoder_effective"] is True


def test_case_study_head_ensemble_averages_members(monkeypatch):
    import experiments.case_study as case_study

    monkeypatch.setattr(
        case_study,
        "safe_load_checkpoint",
        lambda *args, **kwargs: ({"encoder": {}}, {}),
    )
    monkeypatch.setattr(
        case_study,
        "_load_encoder_strict",
        lambda *args, **kwargs: {"hash": "stub", "matched_ratio": 1.0},
    )
    monkeypatch.setattr(case_study, "load_state_dict_forgiving", lambda *a, **k: None)

    logits = [math.log(0.2 / 0.8), math.log(0.8 / 0.2)]

    def fake_train_linear_head(*args, **kwargs):
        value = logits.pop(0)

        class _ConstantHead(torch.nn.Module):
            def __init__(self, logit: float) -> None:
                super().__init__()
                self.logit_value = float(logit)

            def forward(self, batch):  # pragma: no cover - exercised indirectly
                return torch.full((batch.shape[0], 1), self.logit_value, dtype=torch.float32)

        head = _ConstantHead(value)
        return {"head": head, "train/batches": 1.0}

    monkeypatch.setattr(case_study, "train_linear_head", fake_train_linear_head)

    def fake_predict(dataset, indices, encoder, head, device, edge_dim, diag_hook=None):
        value = getattr(head, "logit_value", 0.0)
        logits_tensor = torch.full((len(indices), 1), value, dtype=torch.float32)
        probs_tensor = torch.sigmoid(logits_tensor)
        if diag_hook is not None:
            diag_hook(1, len(indices))
        return logits_tensor, probs_tensor

    monkeypatch.setattr(case_study, "_predict_logits_probs_in_chunks", fake_predict)

    result = case_study.run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        encoder_checkpoint="stub.pt",
        evaluation_mode="frozen_finetuned",
        head_ensemble_size=2,
        finetune_epochs=1,
    )

    diag = result.diagnostics
    assert diag["head_ensemble_size"] == 2
    assert diag["head_ensemble_members_trained"] == 2
    assert len(diag.get("head_ensemble_metrics", [])) == 2
    metrics = result.evaluations[0].metrics
    assert not math.isnan(metrics["roc_auc"])


def test_auto_shape_coercion_normalises_metadata(tmp_path, monkeypatch):
    sklearn_stub = types.ModuleType("sklearn")
    sklearn_linear = types.ModuleType("sklearn.linear_model")
    sklearn_metrics = types.ModuleType("sklearn.metrics")

    class _DummyLogisticRegression:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):  # pragma: no cover - simple stub
            return self

        def predict(self, X):  # pragma: no cover - simple stub
            return np.zeros(len(X) if hasattr(X, "__len__") else 0)

    sklearn_linear.LogisticRegression = _DummyLogisticRegression
    sklearn_metrics.average_precision_score = lambda *args, **kwargs: 0.0
    sklearn_metrics.brier_score_loss = lambda *args, **kwargs: 0.0
    sklearn_metrics.roc_auc_score = lambda *args, **kwargs: 0.0
    sklearn_metrics.mean_absolute_error = lambda *args, **kwargs: 0.0
    sklearn_metrics.mean_squared_error = lambda *args, **kwargs: 0.0
    sklearn_metrics.r2_score = lambda *args, **kwargs: 0.0
    sklearn_stub.linear_model = sklearn_linear
    sklearn_stub.metrics = sklearn_metrics
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_stub)
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", sklearn_linear)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", sklearn_metrics)

    training_pkg = types.ModuleType("training")
    supervised_stub = types.ModuleType("training.supervised")
    unsupervised_stub = types.ModuleType("training.unsupervised")
    supervised_stub.train_linear_head = lambda **kwargs: {}
    unsupervised_stub.train_jepa = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "training", training_pkg)
    monkeypatch.setitem(sys.modules, "training.supervised", supervised_stub)
    monkeypatch.setitem(sys.modules, "training.unsupervised", unsupervised_stub)

    build_calls: Dict[str, Any] = {}

    class _StubEncoder(torch.nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim

        def to(self, *args, **kwargs):  # pragma: no cover - fluent helper
            return self

    def _stub_build_encoder(*, gnn_type, input_dim, hidden_dim, num_layers, edge_dim=None):
        build_calls.update(
            {
                "gnn_type": gnn_type,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "edge_dim": edge_dim,
            }
        )
        return _StubEncoder(hidden_dim)

    class _StubEMA:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):  # pragma: no cover - no-op stub
            return None

    class _StubPredictor(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):  # pragma: no cover - identity
            return x

    models_pkg = types.ModuleType("models")
    factory_stub = types.ModuleType("models.factory")
    encoder_stub = types.ModuleType("models.encoder")
    ema_stub = types.ModuleType("models.ema")
    predictor_stub = types.ModuleType("models.predictor")
    factory_stub.build_encoder = _stub_build_encoder
    encoder_stub.GNNEncoder = _StubEncoder
    ema_stub.EMA = _StubEMA
    predictor_stub.MLPPredictor = _StubPredictor
    models_pkg.factory = factory_stub
    models_pkg.encoder = encoder_stub
    models_pkg.ema = ema_stub
    models_pkg.predictor = predictor_stub
    monkeypatch.setitem(sys.modules, "models", models_pkg)
    monkeypatch.setitem(sys.modules, "models.factory", factory_stub)
    monkeypatch.setitem(sys.modules, "models.encoder", encoder_stub)
    monkeypatch.setitem(sys.modules, "models.ema", ema_stub)
    monkeypatch.setitem(sys.modules, "models.predictor", predictor_stub)

    import experiments.case_study as case_study

    dummy_ckpt = tmp_path / "encoder.pt"
    dummy_ckpt.write_text("stub", encoding="utf-8")

    class DummyGraph:
        def __init__(self, smiles: str):
            self.x = torch.zeros((4, 3))
            self.edge_attr = torch.zeros((4, 1))
            self.smiles = smiles

    add3d_calls: list[bool] = []

    class DummyDataset:
        def __init__(self, graphs, labels, smiles):
            self.graphs = graphs
            self.labels = np.asarray(labels, dtype=float)
            self.smiles = smiles

        @classmethod
        def from_smiles_list(cls, smiles_list, labels, add_3d=False):
            add3d_calls.append(bool(add_3d))
            graphs = [DummyGraph(smi) for smi in smiles_list]
            return cls(graphs, labels, list(smiles_list))

        def __len__(self):
            return len(self.graphs)

    monkeypatch.setattr(case_study, "_load_real_graphdataset", lambda: DummyDataset)
    monkeypatch.setattr(case_study, "attach_bond_features_from_smiles", lambda *args, **kwargs: None)

    state = {
        "encoder": {"weight": torch.zeros((2, 2))},
        "encoder_cfg": {"hidden_dim": 512, "num_layers": 7, "gnn_type": "gin"},
    }

    def fake_safe_load_checkpoint(primary, **_kwargs):
        assert primary == str(dummy_ckpt)
        return state, {}

    monkeypatch.setattr(case_study, "safe_load_checkpoint", fake_safe_load_checkpoint)

    load_calls: Dict[str, Any] = {}

    def fake_load_encoder(
        module,
        raw_state,
        *,
        allow_shape_coercion,
        verify_match_threshold,
        hidden_dim,
        edge_dim=None,
        checkpoint_hidden_dim=None,
        checkpoint_edge_dim=None,
        ckpt_path,
    ):
        load_calls.update(
            {
                "allow_shape_coercion": allow_shape_coercion,
                "hidden_dim": hidden_dim,
                "edge_dim": edge_dim,
                "ckpt_path": ckpt_path,
            }
        )
        return {"matched_ratio": 1.0}

    monkeypatch.setattr(case_study, "_load_encoder_strict", fake_load_encoder)

    def fake_train_linear_head(**kwargs):
        head = torch.nn.Linear(kwargs["encoder"].hidden_dim, 1)
        return {"head": head, "train/batches": 1.0}

    monkeypatch.setattr(case_study, "train_linear_head", fake_train_linear_head)
    monkeypatch.setattr(
        case_study,
        "_evaluate_case_study",
        lambda **kwargs: (0.5, 0.0, 0.5, {}, {"roc_auc": 0.5}, {"enabled": False, "status": "skip"}),
    )

    result = case_study.run_tox21_case_study(
        csv_path="samples/tox21_mini.csv",
        task_name="NR-AR",
        encoder_checkpoint=str(dummy_ckpt),
        hidden_dim=128,
        num_layers=2,
        gnn_type="mpnn",
        evaluation_mode="pretrain_frozen",
        add_3d=True,
    )

    assert result.evaluations, "expected evaluations to be produced"
    assert build_calls["hidden_dim"] == 512
    assert build_calls["num_layers"] == 7
    assert build_calls["gnn_type"] == "gin"
    assert load_calls["hidden_dim"] == 512
    assert load_calls["allow_shape_coercion"] is False
    assert load_calls["edge_dim"] == 1
    assert add3d_calls == [True]
    assert result.diagnostics.get("encoder_config", {}).get("hidden_dim") == 512
