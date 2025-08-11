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
    # device = node_embeddings.device
    # B = graph_ptr.numel()-1
    # hidden_dim = node_embeddings.size(1)
    # graph_embeddings = torch.empty((B, hidden_dim), device=device)
    # for i in range(B):
    #     s = int(graph_ptr[i].item())
    #     e = int(graph_ptr[i + 1].item())
    #     graph_embeddings[i] = node_embeddings[s:e].mean(dim=0) if e > s else node_embeddings.new_zeros(hidden_dim)

    # return graph_embeddings

    x = node_embeddings
    p = graph_ptr.to(dtype=torch.long).view(-1)

    # Normalize to B+1 style ptr
    if p.numel() == 0:
        return x.new_zeros((0, x.size(1)))
    if p[0].item() != 0:
        # Looks like cumulative without leading 0 → prepend 0
        p = torch.cat([p.new_tensor([0]), p], dim=0)

    B = p.numel() - 1
    D = x.size(1)
    out = x.new_zeros((B, D))

    for i in range(B):
        s = int(p[i].item())
        e = int(p[i + 1].item())
        if e > s:
            out[i] = x[s:e].mean(dim=0)
        # else keep zeros for empty graphs

    return out