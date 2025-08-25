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
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred_logits: np.ndarray
) -> Dict[str, float]:
    """Compute ROC‑AUC, PR‑AUC and Brier score for binary classification.

    The logits are converted to probabilities using the sigmoid
    transformation. If the true labels contain only one class the
    AUC metrics are undefined; in this case NaN values are returned.

    Args:
        y_true: Array of binary labels (0 or 1).
        y_pred_logits: Array of predicted logits.

    Returns:
        Dictionary with keys "roc_auc", "pr_auc" and "brier".
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
    try:
        brier = brier_score_loss(y_true, probs)
    except ValueError:
        brier = float("nan")
        
    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc
    metrics["brier"] = brier
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute RMSE, MAE and R² for regression tasks."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}
