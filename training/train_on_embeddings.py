"""Train linear models on learned embeddings using scikit-learn.

This module depends on :mod:`scikit-learn` for model implementations and
evaluation metrics.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def train_linear_on_embeddings_classification(
    X: np.ndarray, y: np.ndarray, max_iter: int = 500, C: float = 1.0
) -> dict:
    clf = LogisticRegression(max_iter=max_iter, n_jobs=1, C=C)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "acc": float((clf.predict(X) == y).mean()),
        "brier": float(brier_score_loss(y, proba)),
    }


def train_linear_on_embeddings_regression(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0
) -> dict:
    reg = Ridge(alpha=alpha, random_state=42)
    reg.fit(X, y)
    pred = reg.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    return {"rmse": rmse, "mae": mae, "r2": float(reg.score(X, y))}


def train_linear_on_embeddings_with_val(
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if task_type == "classification":
        clf = LogisticRegression(max_iter=500, n_jobs=1)
        clf.fit(X_train, y_train.astype(int))

        def _eval(X, y):
            proba = clf.predict_proba(X)[:, 1]
            return dict(
                roc_auc=float(roc_auc_score(y, proba)),
                pr_auc=float(average_precision_score(y, proba)),
                acc=float(accuracy_score(y, clf.predict(X))),
                brier=float(brier_score_loss(y, proba)),
            )

        out = {f"val_{k}": v for k, v in _eval(X_val, y_val.astype(int)).items()}
        if X_test is not None and y_test is not None:
            out.update(
                {f"test_{k}": v for k, v in _eval(X_test, y_test.astype(int)).items()}
            )
        return out
    else:
        reg = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train.astype(float))

        def _eval(X, y):
            pred = reg.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y, pred)))
            mae = float(mean_absolute_error(y, pred))
            return dict(rmse=rmse, mae=mae, r2=float(reg.score(X, y)))

        out = {f"val_{k}": v for k, v in _eval(X_val, y_val.astype(float)).items()}
        if X_test is not None and y_test is not None:
            out.update(
                {f"test_{k}": v for k, v in _eval(X_test, y_test.astype(float)).items()}
            )
        return out
