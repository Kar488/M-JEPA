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

import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from utils.dataloader import autotune_worker_pool, ensure_open_file_limit


logger = logging.getLogger(__name__)


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

    def close(self) -> None:
        """Release references to loaded graphs to free memory early."""

        self.graphs.clear()


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
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
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

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    tuned_workers, tuned_persistent, tuned_prefetch = autotune_worker_pool(
        requested_workers=num_workers,
        dataset_size=len(train_ds),
        batch_size=batch_size,
        device_type=device_type,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        logger=logger,
        stage="parquet_loader",
    )

    if tuned_workers > 0:
        worker_count = int(tuned_workers)
        prefetch_budget = (
            int(tuned_prefetch) if isinstance(tuned_prefetch, (int, float)) else 0
        )
        if worker_count > 0:
            prefetch_budget = max(prefetch_budget, 2)
        min_fd_budget = max(
            4096, 1024 + 128 * max(worker_count, 1) * max(prefetch_budget, 1)
        )
        ensure_open_file_limit(min_fd_budget)

    common = dict(
        num_workers=tuned_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(tuned_workers) and tuned_persistent,
        **loader_kwargs,
    )

    def _prefetch_for(ds: Sequence) -> Optional[int]:
        if tuned_workers <= 0 or tuned_prefetch is None:
            return None
        batches = max(1, math.ceil(len(ds) / max(1, batch_size)))
        return max(1, min(tuned_prefetch, max(2, batches)))

    def _build_loader(ds, *, shuffle: bool) -> DataLoader:
        kwargs = dict(common)
        prefetch = _prefetch_for(ds)
        if prefetch is not None:
            kwargs["prefetch_factor"] = prefetch
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)

    train_loader = _build_loader(train_ds, shuffle=True)
    val_loader = _build_loader(val_ds, shuffle=False)
    test_loader = _build_loader(test_ds, shuffle=False)
    return train_loader, val_loader, test_loader
