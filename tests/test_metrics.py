import math
import numpy as np
from utils.metrics import compute_classification_metrics, compute_regression_metrics

def test_compute_classification_metrics():
    y_true = np.array([0, 1] * 5)
    logits = np.linspace(-1, 1, 10)
    m = compute_classification_metrics(y_true, logits)
    assert set(m.keys()) == {"roc_auc", "pr_auc"}
    assert not math.isnan(m["roc_auc"]) and not math.isnan(m["pr_auc"])


def test_compute_classification_metrics_single_class():
    y_true = np.zeros(5)
    logits = np.zeros(5)
    m = compute_classification_metrics(y_true, logits)
    assert math.isnan(m["roc_auc"]) and m["pr_auc"] == 0.0


def test_compute_regression_metrics():
    y_true = np.array([0.0, 1.0, 2.0])
    preds = np.array([0.0, 1.0, 2.0])
    m = compute_regression_metrics(y_true, preds)
    assert m["rmse"] == 0.0 and m["mae"] == 0.0