from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, silhouette_score
)
from sklearn.cluster import KMeans


def linear_probing_classification(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(embeddings, labels)
    probs = clf.predict_proba(embeddings)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "acc": float((clf.predict(embeddings) == labels).mean()),
    }


def linear_probing_regression(embeddings: np.ndarray, targets: np.ndarray) -> dict:
    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(embeddings, targets)
    preds = reg.predict(embeddings)
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(reg.score(embeddings, targets)),
    }


def clustering_quality(embeddings: np.ndarray, n_clusters: int = 10) -> dict:
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings)
    return {"silhouette": float(silhouette_score(embeddings, labels))}
