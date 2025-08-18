"""Data module for molecule graph processing.

This package contains classes and functions to parse SMILES strings into
graph representations, sample subgraphs for pretraining, and create
batches of graphs for both unsupervised and supervised learning.
"""

# Import GraphDataset and GraphData. ``mdataset`` gracefully handles the
# optional RDKit dependency internally so ``GraphDataset`` works whether or not
# RDKit is installed.
from .mdataset import GraphData, GraphDataset  # type: ignore

__all__ = ["GraphDataset", "GraphData"]
