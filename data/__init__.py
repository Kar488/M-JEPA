"""Data module for molecule graph processing.

This package contains classes and functions to parse SMILES strings into
graph representations, sample subgraphs for pretraining, and create
batches of graphs for both unsupervised and supervised learning.
"""

try:
    from .mdataset import GraphDataset, GraphData  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GraphDataset = GraphData = None  # type: ignore

__all__ = ["GraphDataset", "GraphData"]
