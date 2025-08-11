from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import silhouette_score

from data.mdataset import GraphDataset
from data.scaffold_split import scaffold_split_indices

import torch
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.data import Data as PyGData


def _to_tensor(x, dtype=None, device=None):
    if x is None:
        return None
    t = torch.as_tensor(x)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t

def _edge_index_to_dense(edge_index: torch.Tensor, num_nodes: int, device, add_self_loops: bool = True):
    if edge_index.numel() == 0:
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    else:
        values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=device).to_dense()
    if add_self_loops:
        adj = adj.clone()
        adj.fill_diagonal_(1.0)
    return adj

def _encode_graph(encoder: torch.nn.Module, g):
    device = next(encoder.parameters()).device if any(True for _ in encoder.parameters()) else torch.device("cpu")
    x = getattr(g, "x", None)
    x_t = _to_tensor(x, dtype=torch.float32, device=device)

    # Prefer GraphData.adj; otherwise build from PyG edge_index
    if hasattr(g, "adj") and getattr(g, "adj") is not None:
        adj_t = _to_tensor(getattr(g, "adj"), dtype=torch.float32, device=device)
        if adj_t.ndim > 2:
            adj_t = adj_t.squeeze()
        if adj_t.shape[0] == adj_t.shape[1]:
            adj_t = adj_t.clone()
            adj_t.fill_diagonal_(1.0)
    else:
        ei_t = _to_tensor(getattr(g, "edge_index"), dtype=torch.long, device=device)
        num_nodes = int(x_t.shape[0])
        adj_t = _edge_index_to_dense(ei_t, num_nodes, device=device, add_self_loops=True)

    # Try (x, adj, edge_attr) then fallback (x, adj)
    edge_t = _to_tensor(getattr(g, "edge_attr", None), dtype=torch.float32, device=device)
    try:
        return encoder(x_t, adj_t, edge_t)
    except TypeError:
        return encoder(x_t, adj_t)

def _pool_graph_emb(h: torch.Tensor, g) -> torch.Tensor:
    # Use PyG batch vector if present
    batch = getattr(g, "batch", None)
    if batch is not None:
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(h, batch)  # [B, D]
    # Single graph
    return h.mean(dim=0, keepdim=True) if h.dim() == 2 else h

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
    if ei is None:
        adj = getattr(g, "adj", None)
        if adj is not None:
            ei = _adj_to_edge_index(adj)

    return PyGData(
        x=torch.as_tensor(x, dtype=torch.float32) if x is not None else None,
        edge_index=torch.as_tensor(ei, dtype=torch.long) if ei is not None else None,
        edge_attr=torch.as_tensor(ea, dtype=torch.float32) if ea is not None else None,
        y=(torch.as_tensor(y) if y is not None else None),
    )

def compute_embeddings(dataset, encoder, batch_size=32, device="cpu"):
    encoder = encoder.to(device).eval()

    # Get a sequence of graphs the DataLoader can handle
    if hasattr(dataset, "graphs"):
        graphs_raw = dataset.graphs
    elif hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        graphs_raw = dataset
    else:
        graphs_raw = [dataset]

    graphs = [_to_pyg(g) for g in graphs_raw]   # <-- convert to PyG Data here
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
        tr, _, te = scaffold_split_indices(smiles)
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
