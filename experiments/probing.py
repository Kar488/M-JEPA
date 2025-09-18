from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import silhouette_score

from data.mdataset import GraphDataset, GraphData
from data.scaffold_split import scaffold_split_indices
from data.augment import generate_views

import torch
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.data import Data as PyGData

from utils.graph_ops import _encode_graph, _pool_graph_emb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

def _adj_to_edge_index(adj):
    import torch
    t = torch.as_tensor(adj)
    idx = (t > 0).nonzero(as_tuple=False).t().contiguous()  # [2, E]
    return idx

def _to_pyg(g) -> PyGData:
    """
    Convert your custom GraphData or assorted objects to PyG Data.
    Uses edge_index if present; otherwise derives it from dense adj.
    Carries x, edge_index, edge_attr, y when available.
    """
    if isinstance(g, PyGData):
        return g

    import torch
    x  = getattr(g, "x", None)
    ei = getattr(g, "edge_index", None)
    ea = getattr(g, "edge_attr", None)
    y  = getattr(g, "y", None)
    pos = getattr(g, "pos", None)
    if ei is None:
        adj = getattr(g, "adj", None)
        if adj is not None:
            ei = _adj_to_edge_index(adj)

    return PyGData(
        x=torch.as_tensor(x, dtype=torch.float32) if x is not None else None,
        edge_index=torch.as_tensor(ei, dtype=torch.long) if ei is not None else None,
        edge_attr=torch.as_tensor(ea, dtype=torch.float32) if ea is not None else None,
        y=(torch.as_tensor(y) if y is not None else None),
        pos=torch.as_tensor(pos, dtype=torch.float32) if pos is not None else None,
    )

# Safe AUC for binary or multi-class; returns NaN if undefined
def _safe_auc(y_true, y_proba):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 2):
            p1 = y_proba if y_proba.ndim == 1 else y_proba[:, -1]
            return float(roc_auc_score(y_true, p1))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")

def compute_embeddings(
    dataset,
    encoder,
    batch_size=32,
    device="cpu",
    structural_ops=None,
    geometric_ops=None,
):
    encoder = encoder.to(device).eval()

    # Get a sequence of graphs the DataLoader can handle
    if hasattr(dataset, "graphs"):
        graphs_raw = dataset.graphs
    elif hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        graphs_raw = dataset
    else:
        graphs_raw = [dataset]

    graphs = []
    for g in graphs_raw:
        if isinstance(g, GraphData):
            g = generate_views(g, structural_ops or [], geometric_ops or [])[0]
        graphs.append(_to_pyg(g))
    loader = GeoLoader(graphs, batch_size=batch_size, shuffle=False)

    outs = []
    with torch.no_grad():
        for batch in loader:
            # batch is a PyG Batch: has x, edge_index, batch (graph ids)
            h_nodes = _encode_graph(encoder, batch)      # [sum(N_i), D]
            h_graph = _pool_graph_emb(h_nodes, batch)    # [B, D]
            outs.append(h_graph.detach().cpu())
    if not outs:
        return np.zeros((0, getattr(encoder, "hidden_dim", 1)), dtype=np.float32)
    H = torch.cat(outs, dim=0).numpy()  # [num_graphs, D]
    return H


def linear_probe_classification(
    X: np.ndarray,
    y: np.ndarray,
    smiles: Optional[List[str]] = None,
    use_scaffold: bool = False,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    n = len(y)
    if use_scaffold and smiles is not None:
        tr, _, te = scaffold_split_indices(smiles)
        tr, te = tr.tolist(), te.tolist()
    else:
        idx = np.random.permutation(n)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        tr, te = idx[:train_end], idx[val_end:]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    pred = clf.predict(X[te])
    yt = y[te]
    return {
        "probe_roc_auc": _safe_auc(yt, proba),
        "probe_pr_auc": float(average_precision_score(yt, proba)),
        "probe_acc": float(accuracy_score(yt, pred)),
        "probe_brier": float(brier_score_loss(yt, proba)),
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
        tr, _, te = scaffold_split_indices(smiles)
        tr, te = tr.tolist(), te.tolist()
    else:
        idx = np.random.permutation(n)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        tr, te = idx[:train_end], idx[val_end:]

    X_train = np.asarray(X[tr], dtype=np.float64)
    X_test = np.asarray(X[te], dtype=np.float64)

    # Ridge does not handle NaNs, so impute with training-set means.
    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(X_train, axis=0)
    # Replace columns that are entirely NaN with zeros to avoid NaN means.
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_train = np.where(np.isnan(X_train), col_means, X_train)
    X_test = np.where(np.isnan(X_test), col_means, X_test)

    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(X_train, y[tr])
    pred = reg.predict(X_test)
    yt = y[te]
    return {
        "probe_rmse": float(np.sqrt(mean_squared_error(yt, pred))),
        "probe_mae": float(mean_absolute_error(yt, pred)),
        "probe_r2": float(reg.score(X_test, yt)),
    }


def clustering_quality(X: np.ndarray, n_clusters: int = 10) -> Dict[str, float]:
    n = len(X)
    # need at least 3 samples for a valid silhouette
    if n < 3:
        return {"cluster_silhouette": 0.0}

    # cap clusters to at most n-1 to satisfy silhouette constraints
    k = min(n_clusters, n - 1)

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    lab = km.fit_predict(X)

    # silhouette_score also needs at least 2 distinct labels
    if len(np.unique(lab)) < 2:
        return {"cluster_silhouette": 0.0}

    sil = float(silhouette_score(X, lab))
    return {"cluster_silhouette": sil}
