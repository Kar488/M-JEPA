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
    device = node_embeddings.device
    B = graph_ptr.numel()
    hidden_dim = node_embeddings.size(1)
    graph_embeddings = torch.zeros((B, hidden_dim), device=device)
    start = 0
    for i, end in enumerate(graph_ptr):
        end_idx = end.item()
        if end_idx > start:
            graph_embeddings[i] = node_embeddings[start:end_idx].mean(dim=0)
        start = end_idx
    return graph_embeddings
