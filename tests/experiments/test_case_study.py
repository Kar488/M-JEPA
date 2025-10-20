import logging
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

    class DummyDataset:
        def __init__(self, graphs, labels, smiles):
            self.graphs = graphs
            self.labels = np.asarray(labels, dtype=float)
            self.smiles = smiles

        @classmethod
        def from_smiles_list(cls, smiles_list, labels, add_3d=False):
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
        ckpt_path,
    ):
        load_calls.update(
            {
                "allow_shape_coercion": allow_shape_coercion,
                "hidden_dim": hidden_dim,
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
    )

    assert result.evaluations, "expected evaluations to be produced"
    assert build_calls["hidden_dim"] == 512
    assert build_calls["num_layers"] == 7
    assert build_calls["gnn_type"] == "gin"
    assert load_calls["hidden_dim"] == 512
    assert load_calls["allow_shape_coercion"] is False
