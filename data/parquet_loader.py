from __future__ import annotations

"""Utilities for loading graph datasets stored in Parquet files.

This module provides a light‑weight loader that reads preprocessed graph
representations from Parquet files and converts them into
``torch_geometric.data.Data`` objects.  It is primarily intended for
cases where graph featurisation has been performed offline and the
resulting node/edge features are stored directly in the Parquet rows.
It requires :mod:`pandas` for reading the underlying Parquet data.

Expected schema for each row
----------------------------
Each row in the Parquet files should contain at least the following
columns:

``x``
    Nested list or array representing the node feature matrix with shape
    ``[num_nodes, feat_dim]``.
``edge_index``
    Nested list/array with shape ``[num_edges, 2]`` describing the graph
    connectivity.
``edge_attr`` (optional)
    Nested list/array with shape ``[num_edges, edge_feat_dim]`` holding
    edge features.
``y`` (optional)
    Any label or target associated with the graph.

All remaining columns are ignored.  The loader is intentionally simple –
it reads the entire Parquet files into memory which is sufficient for
the tiny test cases used in the exercises.  For large datasets one may
wish to implement streaming reads instead.
"""

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _row_to_data(row: dict) -> Data:
    """Turn a table row into a handy graph object.

    Convert a dictionary representation of one DataFrame row into a
    ``torch_geometric.data.Data`` instance, handling node features, edge
    indices, optional edge attributes and labels.
    """
    x = torch.as_tensor(row["x"], dtype=torch.float)
    edge_index = torch.as_tensor(row["edge_index"], dtype=torch.long)
    if edge_index.numel() > 0 and edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    edge_attr = row.get("edge_attr")
    if edge_attr is not None:
        edge_attr = torch.as_tensor(edge_attr, dtype=torch.float)
    y = row.get("y")
    if y is not None:
        y = torch.as_tensor(y)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class ParquetGraphDataset(torch.utils.data.Dataset):
    """Keep a bunch of graphs ready for quick grabs.

    Load graphs from one or more Parquet files into memory so they can be
    served through the standard PyTorch ``Dataset`` interface.
    """

    def __init__(self, files: Sequence[Path]):
        graphs: List[Data] = []
        for fp in files:
            df = pd.read_parquet(fp)
            for row in df.to_dict(orient="records"):
                graphs.append(_row_to_data(row))
        self.graphs = graphs

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:  # pragma: no cover - trivial
        return self.graphs[idx]


def _split_files(root: Path, split: str) -> List[Path]:
    """Find all the files that belong to one data split.

    Look for ``<split>.parquet`` or sharded files matching
    ``<split>_part_*.parquet`` under ``root`` and return them in sorted
    order.  Raise :class:`FileNotFoundError` if none exist.
    """

    pat_sharded = f"{split}_part_*.parquet"
    files = sorted(root.glob(pat_sharded))
    if not files:
        single = root / f"{split}.parquet"
        if single.exists():
            files = [single]
    if not files:
        raise FileNotFoundError(f"No parquet files found for split '{split}' under {root}")
    return files


def load_dataloaders(
    parquet_root: str,
    batch_size: int,
    *,
    num_workers: int = 0,
    **loader_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Hand out train, validation and test batches.

    Construct :class:`torch_geometric.loader.DataLoader` objects for each
    split by reading preprocessed graphs from Parquet files located under
    ``parquet_root``.  Additional ``DataLoader`` keyword arguments can be
    supplied via ``loader_kwargs``.
    """

    root = Path(parquet_root)
    train_ds = ParquetGraphDataset(_split_files(root, "train"))
    val_ds = ParquetGraphDataset(_split_files(root, "val"))
    test_ds = ParquetGraphDataset(_split_files(root, "test"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, **loader_kwargs
    )
    return train_loader, val_loader, test_loader
