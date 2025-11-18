from __future__ import annotations

import importlib
import os
from typing import Optional, Protocol, runtime_checkable

GraphDataset = None  # type: ignore[assignment]
_GRAPH_DATASET_IMPORT_ERROR: Optional[Exception] = None
_GRAPH_DATASET_MODULE: Optional[object] = None


@runtime_checkable
class SupportsTeardown(Protocol):
    """Protocol for datasets exposing an explicit resource teardown hook."""

    def close(self) -> None:
        """Release any cached readers, file handles or large buffers."""


def _ensure_graphdataset() -> "GraphDataset":  # type: ignore
    """Return GraphDataset or raise an informative ImportError."""

    global GraphDataset, _GRAPH_DATASET_MODULE, _GRAPH_DATASET_IMPORT_ERROR

    if GraphDataset is not None:
        return GraphDataset

    try:
        _GRAPH_DATASET_MODULE = importlib.import_module("data.mdataset")
        GraphDataset = getattr(_GRAPH_DATASET_MODULE, "GraphDataset")
    except Exception as exc:  # pragma: no cover - import failure path
        _GRAPH_DATASET_IMPORT_ERROR = exc
        raise ImportError(
            "GraphDataset is unavailable. Ensure `data.mdataset.GraphDataset` can be imported."
        ) from exc

    return GraphDataset


def load_dataset(path: str, **kwargs) -> "GraphDataset":  # type: ignore
    """Load a dataset from a file or directory using GraphDataset factory methods."""
    ds_cls = _ensure_graphdataset()
    if os.path.isdir(path):
        return ds_cls.from_directory(dirpath=path, **kwargs)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return ds_cls.from_parquet(filepath=path, **kwargs)
    if ext == ".csv":
        return ds_cls.from_csv(filepath=path, **kwargs)
    raise ValueError(f"Unsupported dataset path: {path}")


def load_directory_dataset(dirpath: str, **kwargs) -> "GraphDataset":  # type: ignore
    """Backward-compatible wrapper that delegates to :func:`load_dataset`."""
    return load_dataset(dirpath, **kwargs)


def load_parquet_dataset(filepath: str, **kwargs) -> "GraphDataset":  # type: ignore
    """Backward-compatible wrapper that delegates to :func:`load_dataset`."""
    return load_dataset(filepath, **kwargs)
