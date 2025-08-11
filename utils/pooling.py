"""Pooling operations for graph embeddings.

Global mean pooling collapses node embeddings into a single vector per
graph by averaging embeddings across nodes. This operation requires
graph boundary pointers that indicate where each graph begins and ends in
the concatenated batch representation.
"""

from __future__ import annotations

import torch


def global_mean_pool(
    node_embeddings: torch.Tensor, graph_ptr: torch.Tensor
) -> torch.Tensor:
    """Compute graph embeddings by averaging node embeddings per graph.

    Args:
        node_embeddings: Tensor of shape (N_total, hidden_dim).
        graph_ptr: Tensor of shape (B,) where graph_ptr[i] gives the
            cumulative number of nodes up to graph i (inclusive). For
            example, if graph_ptr = [3, 7, 10], the first graph has 3
            nodes, the second has 4 nodes and the third has 3 nodes.

    Returns:
        Tensor of shape (B, hidden_dim) containing one embedding per graph.
    """
    #device = node_embeddings.device
    #B = graph_ptr.numel()
    ##hidden_dim = node_embeddings.size(1)
    #graph_embeddings = torch.zeros((B, hidden_dim), device=device)
    #start = 0
    #for i, end in enumerate(graph_ptr):
        #end_idx = end.item()
        #if end_idx > start:
            #graph_embeddings[i] = node_embeddings[start:end_idx].mean(dim=0)
        #start = end_idx
    #return graph_embeddings

    if not torch.is_tensor(node_embeddings):
            node_embeddings = torch.as_tensor(node_embeddings)
       
    if not torch.is_tensor(graph_ptr):
        graph_ptr = torch.as_tensor(graph_ptr)

    assert node_embeddings.dim() == 2, f"node_embeddings must be [N,F], got {list(node_embeddings.shape)}"
    N = node_embeddings.size(0)

    # --- Case A: CSR pointer ---
    if graph_ptr.dim() == 1 and graph_ptr.numel() >= 2 and graph_ptr.dtype in (torch.long, torch.int64):
        ptr = graph_ptr
        # ptr should be non-decreasing, start at 0, end at N
        if ptr[0].item() != 0 or ptr[-1].item() != N or torch.any(ptr[1:] < ptr[:-1]):
            raise ValueError(f"Invalid ptr: {ptr.tolist()} for N={N}")
        G = ptr.numel() - 1
        lengths = (ptr[1:] - ptr[:-1]).clamp_min(0)  # [G]
        # Build segment ids: 0 repeated lengths[0] times, etc.
        seg_ids = torch.repeat_interleave(torch.arange(G, device=node_embeddings.device), lengths) # [N]
        # Scatter-add and divide
        out_sum = torch.zeros(G, node_embeddings.size(1), dtype=node_embeddings.dtype, device=node_embeddings.device)
        if seg_ids.numel() > 0:
            out_sum.index_add_(0, seg_ids, node_embeddings)
        denom = lengths.clamp_min(1).to(node_embeddings.dtype).unsqueeze(1)  # avoid div-by-zero
        out = out_sum / denom
        # zero out truly empty graphs (length 0)
        if (lengths == 0).any():
            out[lengths == 0] = 0
        return out

    # --- Case B: batch vector ---
    if graph_ptr.shape == (N,):
        batch = graph_ptr.to(dtype=torch.long, device=node_embeddings.device)
        G = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if G == 0:
            return node_embeddings.new_zeros((0, node_embeddings.size(1)))
        out_sum = torch.zeros(G, node_embeddings.size(1), dtype=node_embeddings.dtype, device=node_embeddings.device)
        out_sum.index_add_(0, batch, node_embeddings)
        counts = torch.bincount(batch, minlength=G).clamp_min(1).to(node_embeddings.dtype).unsqueeze(1)
        out = out_sum / counts
        # zero rows for any id with zero count (paranoia; minlength ensures presence)
        zero_rows = (counts.squeeze(1) == 0)
        if zero_rows.any():
            out[zero_rows] = 0
        return out

    raise TypeError(
        "Second argument must be CSR ptr [G+1] (long) or batch ids [N] (long). "
        f"Got shape {list(graph_ptr.shape)}, dtype {graph_ptr.dtype}."
    )
