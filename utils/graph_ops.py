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
    params = [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    pset = set(params)

    first_arg = params[0] if params else None
    second_arg = params[1] if len(params) > 1 else None

    # --- if the encoder wants a graph/batch object, give it the graph ---
    if (
        (first_arg in {"batch", "g", "graph", "data"})
        or (second_arg in {"batch", "g", "graph", "data"})
        or hasattr(mod, "encode_graph")
    ):
        try:
            return encoder(g)
        except (TypeError, AttributeError) as exc:
            graphs = getattr(g, "graphs", None)
            if graphs is None:
                raise
            outputs = []
            for graph in graphs:
                out = encoder(graph)
                if not isinstance(out, torch.Tensor):
                    out = torch.as_tensor(out)
                if out.dim() == 0:
                    out = out.unsqueeze(0)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                outputs.append(out)
            try:
                return torch.cat(outputs, dim=0)
            except Exception:
                raise exc

    # otherwise prepare tensors as before
    device = _ref_device(encoder)
    x_t   = _to_tensor(getattr(g, "x", None), dtype=torch.float32, device=device)
    ei_t  = _to_tensor(getattr(g, "edge_index", None), dtype=torch.long, device=device)
    edge_t= _to_tensor(getattr(g, "edge_attr", None), dtype=torch.float32, device=device)
    pos_t = _to_tensor(getattr(g, "pos", None), dtype=torch.float32, device=device)
    batch_t = _to_tensor(getattr(g, "batch", None), dtype=torch.long, device=device)
    ptr_t = _to_tensor(getattr(g, "ptr", None), dtype=torch.long, device=device)
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
    if "batch" in pset and batch_t is not None:
        call_kwargs = {}
        if "x" in pset and x_t is not None:
            call_kwargs["x"] = x_t
        if "edge_index" in pset and ei_t is not None:
            call_kwargs["edge_index"] = ei_t
        if "edge_attr" in pset and edge_t is not None:
            call_kwargs["edge_attr"] = edge_t
        if "adj" in pset and "edge_index" not in pset and adj_t is not None:
            call_kwargs["adj"] = adj_t
        if "pos" in pset and pos_t is not None:
            call_kwargs["pos"] = pos_t
        if "ptr" in pset and ptr_t is not None:
            call_kwargs["ptr"] = ptr_t
        call_kwargs["batch"] = batch_t
        return encoder(**call_kwargs)

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
    if batch is not None and isinstance(batch, torch.Tensor):
        if batch.numel() == h.size(0):
            try:
                from torch_geometric.nn import global_mean_pool  # type: ignore

                return global_mean_pool(h, batch)
            except Exception:
                uniq = torch.unique(batch)
                return torch.stack([h[batch == i].mean(dim=0) for i in uniq], dim=0)
        else:
            return h

    if h.dim() == 2:
        return h.mean(dim=0)
    return h


def _ensure_edge_attr_np_or_torch(g, need_dim: int, device=None):
    """Ensure g.edge_attr exists and has width need_dim (numpy or torch)."""
    import numpy as np
    try:
        import torch as _t
        HAS_TORCH = True
    except Exception:
        HAS_TORCH = False

    # how many edges?
    E = 0
    ei = getattr(g, "edge_index", None)
    if ei is not None:
        if HAS_TORCH and isinstance(ei, _t.Tensor):
            E = int(ei.shape[1])
        else:
            E = int(np.asarray(ei).shape[1])
    else:
        adj = getattr(g, "adj", None)
        if adj is not None:
            if HAS_TORCH and isinstance(adj, _t.Tensor):
                E = int((adj > 0).sum().item())
            else:
                A = np.asarray(adj)
                E = int((A > 0).sum())
 
    # ---- helpers to coerce device/dtype -----------------------------------------
    def _to_torch_device(dev):
        import torch as _t
        if dev is None or isinstance(dev, _t.device):
            return dev
        return _t.device(dev)  # accepts "cpu", "cuda", "cuda:0"

    def _to_torch_dtype(dt):
        import torch as _t, numpy as np
        if isinstance(dt, _t.dtype):
            return dt
        # map common numpy dtypes → torch dtypes (fallback to float32)
        return {
            np.float32: _t.float32,
            np.float64: _t.float64,
            np.float16: _t.float16,
            np.int64:   _t.int64,
            np.int32:   _t.int32,
            np.int16:   _t.int16,
            np.int8:    _t.int8,
            np.uint8:   _t.uint8,
        }.get(dt, _t.float32)

    # ---- zeros like x (works for numpy or torch) --------------------------------
    def zeros_like_x(nr, nc):
        import numpy as np
        try:
            import torch as _t
            HAS_TORCH = True
        except Exception:
            HAS_TORCH = False

        x = getattr(g, "x", None)

        # A) x is TORCH tensor → allocate TORCH zeros that match it
        if HAS_TORCH and isinstance(x, _t.Tensor):
            dev_t = x.device if hasattr(x, "device") else None
            if dev_t is None and device is not None:
                dev_t = _to_torch_device(device)
            return _t.zeros((nr, nc), dtype=x.dtype, device=dev_t) if dev_t \
                else _t.zeros((nr, nc), dtype=x.dtype)

        # B) x is NUMPY but caller wants TORCH (factory encoders path)
        if HAS_TORCH and device is not None:
            dev_t = _to_torch_device(device)
            if isinstance(x, np.ndarray):
                # upgrade x to torch so everything stays consistent
                g.x = _t.from_numpy(x).to(dev_t)
                x = g.x
            dt = x.dtype if isinstance(x, _t.Tensor) else _t.float32
            return _t.zeros((nr, nc), dtype=dt, device=dev_t)

        # C) pure NUMPY fallback
        import numpy as np
        dt = getattr(x, "dtype", np.float32)
        return np.zeros((nr, nc), dtype=dt)
    # -----------------------------------------------------------------------------
    # ---- read/repair edge_attr ---------------------------------------------------
    e = getattr(g, "edge_attr", None)
    import numpy as np
    if e is None:
        w = 0
    elif HAS_TORCH and isinstance(e, _t.Tensor):
        w = int(e.shape[1])
    else:
        w = int(np.asarray(e).shape[1])

    if e is None or w == 0:
        g.edge_attr = zeros_like_x(E, need_dim)
        return g

    # if x is torch but edge_attr is numpy, upgrade edge_attr to torch first
    try:
        import torch as _t
        if isinstance(getattr(g, "x", None), _t.Tensor) and isinstance(e, np.ndarray):
            g.edge_attr = _t.from_numpy(e).to(g.x.dtype).to(g.x.device)
            e = g.edge_attr
    except Exception:
        pass

    # ---- pad/trim to need_dim (numpy or torch) ----------------------------------
    try:
        import torch as _t
        if HAS_TORCH and isinstance(e, _t.Tensor):
            if e.shape[1] < need_dim:
                pad = zeros_like_x(e.shape[0], need_dim - e.shape[1])
                g.edge_attr = _t.cat([e, pad], dim=1)
            elif e.shape[1] > need_dim:
                g.edge_attr = e[:, :need_dim]
        else:
            e_np = np.asarray(e)
            if e_np.shape[1] < need_dim:
                pad = zeros_like_x(e_np.shape[0], need_dim - e_np.shape[1])
                g.edge_attr = np.concatenate([e_np, pad], axis=1)
            elif e_np.shape[1] > need_dim:
                g.edge_attr = e_np[:, :need_dim]
    except Exception:
        # last-resort: leave as-is if anything unexpected happens
        g.edge_attr = e

    return g

def _encode_graph_flex(encoder, g, device=None):
    """
    Call either a factory-style encoder (encoder(g))
    or a legacy encoder that expects (x, adj).
    Handles numpy/torch inputs.
    """
    import numpy as np
    import torch

    # First try the single-arg (factory) path.
    try:
        return encoder(g)
    except TypeError:
        # Fall back to legacy (x, adj) signature.
        x = getattr(g, "x", None)
        adj = getattr(g, "adj", None)

        # If adj isn't present but edge_index is, build a quick adjacency.
        if adj is None and getattr(g, "edge_index", None) is not None:
            ei = g.edge_index
            if isinstance(ei, np.ndarray):
                N = int(x.shape[0])
                A = np.zeros((N, N), dtype=np.float32)
                A[ei[0], ei[1]] = 1
                A[ei[1], ei[0]] = 1  # assume undirected
                adj = A
            else:
                # torch
                N = int(x.shape[0])
                A = torch.zeros((N, N), dtype=torch.float32, device=device)
                A[ei[0], ei[1]] = 1
                A[ei[1], ei[0]] = 1
                adj = A

        # Convert numpy → torch as needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
        elif hasattr(x, "to"):
            x = x.to(device)

        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj).to(device)
        elif hasattr(adj, "to"):
            adj = adj.to(device)

        return encoder(x, adj)