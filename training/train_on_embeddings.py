
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error
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
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Fit a simple linear model on embeddings and report metrics on val/test if provided."""
    out = {}
    if task_type == "classification":
        clf = LogisticRegression(max_iter=500, n_jobs=1)
        clf.fit(X_train, y_train)
        proba_tr = clf.predict_proba(X_train)[:, 1]
        out["train_roc_auc"] = float(roc_auc_score(y_train, proba_tr))
        out["train_pr_auc"] = float(average_precision_score(y_train, proba_tr))
        if X_val is not None and y_val is not None:
            proba_v = clf.predict_proba(X_val)[:, 1]
            out["val_roc_auc"] = float(roc_auc_score(y_val, proba_v))
            out["val_pr_auc"] = float(average_precision_score(y_val, proba_v))
        if X_test is not None and y_test is not None:
            proba_te = clf.predict_proba(X_test)[:, 1]
            out["test_roc_auc"] = float(roc_auc_score(y_test, proba_te))
            out["test_pr_auc"] = float(average_precision_score(y_test, proba_te))
    else:
        reg = Ridge(alpha=1.0, random_state=42)
        reg.fit(X_train, y_train)
        pred_tr = reg.predict(X_train)
        out["train_rmse"] = float(np.sqrt(mean_squared_error(y_train, pred_tr)))
        out["train_mae"] = float(mean_absolute_error(y_train, pred_tr))
        if X_val is not None and y_val is not None:
            pred_v = reg.predict(X_val)
            out["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, pred_v)))
            out["val_mae"] = float(mean_absolute_error(y_val, pred_v))
        if X_test is not None and y_test is not None:
            pred_te = reg.predict(X_test)
            out["test_rmse"] = float(np.sqrt(mean_squared_error(y_test, pred_te)))
            out["test_mae"] = float(mean_absolute_error(y_test, pred_te))
    return out
