from __future__ import annotations

import inspect
from typing import Any

import torch


def _ref_device(module: torch.nn.Module) -> torch.device:
    """Return a reference device for a module."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_tensor(x: Any, dtype: torch.dtype | None = None, device: torch.device | None = None):
    """Safely convert ``x`` to a ``torch.Tensor`` with optional dtype and device."""
    if x is None:
        return None
    t = torch.as_tensor(x)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


def _edge_index_to_dense(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Build a dense adjacency matrix from a sparse ``edge_index`` representation."""
    if edge_index.numel() == 0:
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    else:
        values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=device).to_dense()
    if add_self_loops:
        adj = adj.clone()
        adj.fill_diagonal_(1.0)
    return adj


def _encode_graph(encoder: torch.nn.Module, g: Any):
    """
    Encode a graph object with a given encoder.

    Supports objects that expose ``x`` and either ``adj`` or ``edge_index``.
    Tensors are converted to the encoder's device and appropriate dtypes.
    The encoder is called in a flexible manner depending on its forward
    signature: it may accept the whole graph object, ``(x, adj)``,
    ``(x, adj, edge_attr)``, or just ``x``.
    """
    # unwrap common wrappers
    if isinstance(g, (tuple, list)) and len(g):
        g = g[0]

    # quick sanity
    if hasattr(encoder, "module"):
        mod = encoder.module
    else:
        mod = encoder

    sig = inspect.signature(mod.forward)
    params = [p.name for p in sig.parameters.values()
              if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    pset = set(params)

    # --- if the encoder wants a graph/batch object, give it the graph ---
    if (params and params[0] in {"batch", "g", "graph", "data"}) or hasattr(mod, "encode_graph"):
        return encoder(g)

    # otherwise prepare tensors as before
    device = _ref_device(encoder)
    x_t   = _to_tensor(getattr(g, "x", None), dtype=torch.float32, device=device)
    ei_t  = _to_tensor(getattr(g, "edge_index", None), dtype=torch.long, device=device)
    edge_t= _to_tensor(getattr(g, "edge_attr", None), dtype=torch.float32, device=device)
    adj_t = None
    if getattr(g, "adj", None) is not None:
        adj = getattr(g, "adj")
        adj_t = adj.to(device=device, dtype=torch.float32) if torch.is_tensor(adj) \
                else _to_tensor(adj, dtype=torch.float32, device=device)
        if adj_t.ndim > 2: adj_t = adj_t.squeeze()
        if adj_t.shape[0] == adj_t.shape[1]:
            adj_t = adj_t.clone(); adj_t.fill_diagonal_(1.0)

    # derive missing representation if needed
    if ei_t is None and adj_t is not None:
        idx = (adj_t > 0).nonzero(as_tuple=False).T
        ei_t = idx.to(dtype=torch.long, device=device)
    if adj_t is None and ei_t is not None:
        num_nodes = int(x_t.shape[0]) if x_t is not None else int(ei_t.max().item()+1)
        adj_t = _edge_index_to_dense(ei_t, num_nodes, device=device, add_self_loops=True)

    # --- call according to signature ---
    if {"x", "edge_index"}.issubset(pset) and ei_t is not None:
        return encoder(x_t, ei_t, edge_t) if "edge_attr" in pset and edge_t is not None else encoder(x_t, ei_t)
    if {"x", "adj"}.issubset(pset) and adj_t is not None:
        return encoder(x_t, adj_t, edge_t) if "edge_attr" in pset and edge_t is not None else encoder(x_t, adj_t)

    # final fallback: feature-only encoders
    return encoder(x_t)


def _pool_graph_emb(h: torch.Tensor, g: Any) -> torch.Tensor:
    """Aggregate node embeddings into a single graph embedding."""
    if not isinstance(h, torch.Tensor):
        h = torch.as_tensor(h)

    batch = getattr(g, "batch", None)
    if batch is not None:
        try:
            from torch_geometric.nn import global_mean_pool  # type: ignore

            return global_mean_pool(h, batch)
        except Exception:
            uniq = torch.unique(batch)
            return torch.stack([h[batch == i].mean(dim=0) for i in uniq], dim=0)

    if h.dim() == 2:
        return h.mean(dim=0)
    return h
