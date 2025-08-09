from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import silhouette_score

from data.dataset import GraphDataset
from data.scaffold_split import scaffold_split


def compute_embeddings(
    dataset: GraphDataset, encoder, batch_size: int = 64, device: str = "cuda"
) -> np.ndarray:
    encoder.eval()
    embs = []
    G = dataset.graphs
    for i in range(0, len(G), batch_size):
        batch = G[i : i + batch_size]
        import torch

        with torch.no_grad():
            H = encoder(batch).detach().cpu().numpy()
        embs.append(H)
    return (
        np.concatenate(embs, axis=0)
        if embs
        else np.zeros((0, getattr(encoder, "hidden_dim", 1)))
    )


def linear_probe_classification(
    X: np.ndarray,
    y: np.ndarray,
    smiles: Optional[List[str]] = None,
    use_scaffold: bool = False,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    n = len(y)
    if use_scaffold and smiles is not None:
        tr, _, te = scaffold_split(smiles)
        tr, te = tr.tolist(), te.tolist()
    else:
        idx = np.random.permutation(n)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        tr, te = idx[:train_end], idx[val_end:]

    clf = LogisticRegression(max_iter=500, n_jobs=1)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    pred = clf.predict(X[te])
    yt = y[te]
    return {
        "probe_roc_auc": float(roc_auc_score(yt, proba)),
        "probe_pr_auc": float(average_precision_score(yt, proba)),
        "probe_acc": float(accuracy_score(yt, pred)),
    }


def linear_probe_regression(
    X: np.ndarray,
    y: np.ndarray,
    smiles: Optional[List[str]] = None,
    use_scaffold: bool = False,
) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    n = len(y)
    if use_scaffold and smiles is not None:
        tr, _, te = scaffold_split(smiles)
        tr, te = tr.tolist(), te.tolist()
    else:
        idx = np.random.permutation(n)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        tr, te = idx[:train_end], idx[val_end:]

    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(X[tr], y[tr])
    pred = reg.predict(X[te])
    yt = y[te]
    return {
        "probe_rmse": float(np.sqrt(mean_squared_error(yt, pred))),
        "probe_mae": float(mean_absolute_error(yt, pred)),
        "probe_r2": float(reg.score(X[te], yt)),
    }


def clustering_quality(X: np.ndarray, n_clusters: int = 10) -> Dict[str, float]:
    if len(X) < max(2, n_clusters):
        return {"cluster_silhouette": 0.0}
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    lab = km.fit_predict(X)
    sil = float(silhouette_score(X, lab)) if len(np.unique(lab)) > 1 else 0.0
    return {"cluster_silhouette": sil}
