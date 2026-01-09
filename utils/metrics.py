"""Metric computation for supervised tasks.

This module centralises the calculation of performance metrics for
classification and regression tasks. It wraps the sklearn metrics and
handles cases where the metrics are undefined (e.g. only one class in
the ground truth) by returning NaN values rather than raising an
exception. These utilities are used by the supervised training
routines.
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Tuple

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
    """Compute classification metrics with robust guards for degenerate splits."""

    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred_logits, dtype=np.float64).reshape(-1)

    yt = np.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)
    yp = np.nan_to_num(
        yp,
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=-np.finfo(np.float64).max,
    )

    metrics: Dict[str, float] = {
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
        "acc": float("nan"),
    }

    if yt.size == 0 or yp.size == 0:
        return metrics

    y_int = yt.astype(np.int64, copy=False)
    if np.unique(y_int).size < 2:
        return metrics

    probs = _sigmoid(yp)

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_int, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_int, probs))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    try:
        metrics["brier"] = float(brier_score_loss(y_int, probs))
    except ValueError:
        metrics["brier"] = float("nan")

    try:
        metrics["ece"] = float(expected_calibration_error(probs, y_int, n_bins=15))
    except Exception:
        metrics["ece"] = float("nan")

    try:
        preds = (probs >= 0.5).astype(np.int64, copy=False)
        metrics["acc"] = float(np.mean(preds == y_int)) if y_int.size else float("nan")
    except Exception:
        metrics["acc"] = float("nan")

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute RMSE, MAE and R² for regression tasks."""
    # 1) sanitize shapes/types
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    # 2) drop non-finite rows (sklearn will error otherwise)
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() != yt.size:
        yt, yp = yt[m], yp[m]
    # 3) degenerate guard
    if yt.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    # 4) metrics
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae  = float(mean_absolute_error(yt, yp))
    r2   = float("nan") if np.std(yt) < 1e-12 else float(r2_score(yt, yp))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(yt.size)}


try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def _to_numpy(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z, dtype=np.float64)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask], dtype=np.float64))
    exp_z = np.exp(z[neg_mask], dtype=np.float64)
    out[neg_mask] = exp_z / (1.0 + exp_z)
    return out


def _normalize_probs(y_score: np.ndarray) -> np.ndarray:
    """
    Ensure inputs are probabilities. Accepts:
      - binary: shape (N,), values in [0,1]  OR logits (not in [0,1])
      - binary: shape (N, 2), class-ordered probabilities or logits
      - multi : shape (N, C), class-ordered probabilities or logits
    """
    y = np.asarray(y_score, dtype=np.float64)
    if y.ndim == 1:
        # Binary: (N,) → treat as P(pos)
        if (y < 0).any() or (y > 1).any():
            y = _sigmoid(y)
        y = np.stack([1 - y, y], axis=1)
    elif y.ndim == 2:
        # If any row sums outside [0.999,1.001], treat as logits and softmax
        rowsum = y.sum(axis=1, dtype=np.float64)
        if (rowsum < 0.999).any() or (rowsum > 1.001).any() or (y < 0).all():
            y = _softmax(y, axis=1)
    else:
        raise ValueError("y_score must be shape (N,), (N,2) or (N,C)")
    return y


def _expected_calibration_error(
    y_score,
    y_true,
    n_bins: int = 15,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> float:
    """
    Expected Calibration Error (ECE).
    - Multiclass: uses max-probability confidence and argmax prediction.
    - Binary   : either pass P(pos) as shape (N,) or class-probs/logits shape (N,2).

    ECE = sum_b (|acc_b - conf_b| * (n_b / N))
      where acc_b is mean correctness in bin b, conf_b is mean confidence in bin b.
    """
    p = _normalize_probs(_to_numpy(y_score))  # (N, C)
    y = _to_numpy(y_true).astype(np.int64).reshape(-1)
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"y_score and y_true have different lengths: {p.shape[0]} vs {y.shape[0]}")

    if p.size == 0 or y.size == 0:
        return float("nan")

    if p.shape[1] == 0:
        return float("nan")

    # predicted class + confidence
    pred = np.argmax(p, axis=1)
    conf = np.take_along_axis(p, pred[:, None], axis=1).squeeze(1)  # (N,)
    correct = (pred == y).astype(np.float64)

    N = conf.shape[0]
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    elif strategy == "quantile":
        # unique quantiles to avoid empty leading/trailing bins when conf==const
        q = np.linspace(0, 1, n_bins + 1, dtype=np.float64)
        edges = np.unique(np.quantile(conf, q))
        if edges.size < 2:  # all confidences identical → single bin at that value
            edges = np.array([conf.min(), conf.max()], dtype=np.float64)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    # Bin indices: include left edge, exclude right edge; include 1.0 in last bin
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc_b = float(correct[mask].mean())
        conf_b = float(conf[mask].mean())
        ece += (n_b / N) * abs(acc_b - conf_b)
    return float(ece)


# Optional public alias
expected_calibration_error = _expected_calibration_error
