from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression, Ridge

from data.dataset import GraphDataset

def compute_embeddings(dataset: GraphDataset, encoder, batch_size: int = 64, device: str = "cuda") -> np.ndarray:
    encoder.eval()
    embs = []
    G = dataset.graphs
    for i in range(0, len(G), batch_size):
        batch = G[i:i+batch_size]
        import torch
        with torch.no_grad():
            H = encoder(batch).detach().cpu().numpy()
        embs.append(H)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, getattr(encoder, "hidden_dim", 1)))

def linear_probe_classification(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    clf = LogisticRegression(max_iter=500, n_jobs=1)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    return {
        "probe_roc_auc": float(roc_auc_score(y, proba)),
        "probe_pr_auc": float(average_precision_score(y, proba)),
        "probe_acc": float(accuracy_score(y, clf.predict(X))),
    }

def linear_probe_regression(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(X, y)
    pred = reg.predict(X)
    return {
        "probe_rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "probe_mae": float(mean_absolute_error(y, pred)),
        "probe_r2": float(reg.score(X, y)),
    }

def clustering_quality(X: np.ndarray, n_clusters: int = 10) -> Dict[str, float]:
    if len(X) < max(2, n_clusters):
        return {"cluster_silhouette": 0.0}
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    lab = km.fit_predict(X)
    sil = float(silhouette_score(X, lab)) if len(np.unique(lab)) > 1 else 0.0
    return {"cluster_silhouette": sil}
