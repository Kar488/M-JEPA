import math
import numpy as np
import pytest

pytest.importorskip("sklearn")

from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    _normalize_probs,
    expected_calibration_error,
)


def test_compute_classification_metrics():
    y_true = np.array([0, 1] * 5)
    logits = np.linspace(-1, 1, 10)
    m = compute_classification_metrics(y_true, logits)
    assert set(m.keys()) == {"roc_auc", "pr_auc", "brier", "ece", "acc"}
    assert not math.isnan(m["roc_auc"]) and not math.isnan(m["pr_auc"])
    assert not math.isnan(m["brier"]) and not math.isnan(m["ece"])
    assert 0.0 <= m["acc"] <= 1.0


def test_compute_classification_metrics_single_class():
    y_true = np.zeros(5)
    logits = np.zeros(5)
    m = compute_classification_metrics(y_true, logits)
    for key in ("roc_auc", "pr_auc", "brier", "ece", "acc"):
        assert math.isnan(m[key])


def test_compute_classification_metrics_empty():
    m = compute_classification_metrics(np.array([]), np.array([]))
    for key in ("roc_auc", "pr_auc", "brier", "ece", "acc"):
        assert math.isnan(m[key])


def test_compute_regression_metrics():
    y_true = np.array([0.0, 1.0, 2.0])
    preds = np.array([0.0, 1.0, 2.0])
    m = compute_regression_metrics(y_true, preds)
    assert m["rmse"] == 0.0 and m["mae"] == 0.0 and m["r2"] == 1.0


def test_normalize_probs_handles_logits():
    logits = np.array([0.0, 2.0, -2.0])
    probs = _normalize_probs(logits)
    assert probs.shape == (3, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_normalize_probs_multiclass_logits():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    probs = _normalize_probs(logits)
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_expected_calibration_error():
    y_true = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    ece = expected_calibration_error(scores, y_true, n_bins=2, strategy="uniform")
    assert 0.0 <= ece <= 1.0
