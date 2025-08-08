"""Metric computation for supervised tasks.

This module centralises the calculation of performance metrics for
classification and regression tasks. It wraps the sklearn metrics and
handles cases where the metrics are undefined (e.g. only one class in
the ground truth) by returning NaN values rather than raising an
exception. These utilities are used by the supervised training
routines.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
)


def compute_classification_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> Dict[str, float]:
    """Compute ROC‑AUC and PR‑AUC for binary classification.

    The logits are converted to probabilities using the sigmoid
    transformation. If the true labels contain only one class the
    metrics are undefined; in this case NaN values are returned.

    Args:
        y_true: Array of binary labels (0 or 1).
        y_pred_logits: Array of predicted logits.

    Returns:
        Dictionary with keys "roc_auc" and "pr_auc".
    """
    # Convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-y_pred_logits))
    metrics: Dict[str, float] = {}
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, probs)
    except ValueError:
        pr_auc = float("nan")
    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc
    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE and MAE for regression tasks."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}
