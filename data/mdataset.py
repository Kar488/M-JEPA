from __future__ import annotations

import importlib
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from itertools import repeat
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

_RUNNING_IN_CI = os.getenv("CI") == "true"  # run local vs remote

# RDKit imports
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms as MT
except Exception as e:
    if _RUNNING_IN_CI:
        raise ImportError("RDKit is required in CI for SMILES parsing.")
else:  # pragma: no cover - depends on rdkit availability
    try:
        RDLogger.logger().setLevel(RDLogger.ERROR)
    except Exception:
        pass

import pandas as pd

from ._graph_pickle import register_graph_class, rebuild_graph_data

logger = logging.getLogger(__name__)

EDGE_BASE_DIM = 7
EDGE_GEOM_DIM = 10
EDGE_FLAG_DIM = 1
EDGE_TOTAL_DIM = EDGE_BASE_DIM + EDGE_FLAG_DIM + EDGE_GEOM_DIM
GRAPH_CACHE_VERSION = "edgeflex_v2_250k"
GRAPH_SCHEMA_VERSION = "flex_v1"


def _cache_schema_suffix(add_3d: bool) -> str:
    edge_dim = EDGE_TOTAL_DIM if add_3d else EDGE_BASE_DIM
    return f"{GRAPH_CACHE_VERSION}_3d{int(add_3d)}_e{edge_dim}_{GRAPH_SCHEMA_VERSION}"


def _resolve_worker_count(num_workers: Optional[int]) -> int:
    """Map worker counts to an automatic CPU-friendly budget when unset."""

    if num_workers is None:
        num_workers = -1
    if num_workers <= 0:
        cpu_budget = max(1, (os.cpu_count() or 2) - 1)
        return max(1, cpu_budget)
    return int(num_workers)


@lru_cache(maxsize=None)
def _resolve_graphdataset_cls(module_name: str, qualname: str) -> type["GraphDataset"]:
    """Return a GraphDataset subclass given its module and qualified name."""

    module = importlib.import_module(module_name)
    attr = module
    for part in qualname.split("."):
        attr = getattr(attr, part)
    return attr


GraphDataState = Dict[str, Any]


def _graph_to_state(g: "GraphData") -> GraphDataState:
    """Convert a :class:`GraphData` instance into a picklable mapping."""

    return {
        "x": g.x,
        "edge_index": g.edge_index,
        "edge_attr": g.edge_attr,
        "pos": g.pos,
    }


def _graph_from_state(state: GraphDataState) -> "GraphData":
    """Recreate a :class:`GraphData` instance from a serialisable mapping."""

    return GraphData(
        x=state["x"],
        edge_index=state["edge_index"],
        edge_attr=state.get("edge_attr"),
        pos=state.get("pos"),
    )


def _coerce_cache_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        if "graphs" in payload:
            return payload
        # legacy caches stored as {"graphs": [...], "labels": ...}
        graphs = payload.get("graphs")
        if graphs is not None:
            return {"graphs": graphs, "labels": payload.get("labels"), "schema": payload.get("schema")}
    if isinstance(payload, tuple) and len(payload) == 2:
        graphs, labels = payload
        return {"graphs": graphs, "labels": labels, "schema": None}
    return None


def _load_graph_cache(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        logger.warning("Failed to load graph cache %s: %s", path, exc)
        return None
    coerced = _coerce_cache_payload(payload)
    if coerced is None:
        logger.warning("Graph cache %s has unexpected format; ignoring", path)
    return coerced


def _write_graph_cache(path: str, dataset: "GraphDataset") -> None:
    payload = {
        "graphs": dataset.graphs,
        "labels": dataset.labels,
        "schema": dataset.schema_metadata,
    }
    try:
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
    except Exception as exc:
        logger.warning("Failed to write graph cache %s: %s", path, exc)
    else:
        logger.info(
            "Wrote graph cache to %s (schema=%s)",
            path,
            payload["schema"].get("schema_token") if payload.get("schema") else "<none>",
        )


@dataclass
class GraphData:
    x: np.ndarray  # [num_nodes, feat_dim]
    edge_index: np.ndarray  # [2, num_edges] (directed; add reverse edges)
    edge_attr: Optional[np.ndarray] = None  # [num_edges, edge_feat_dim]
    pos: Optional[np.ndarray] = None  # [num_nodes, 3] 3D coordinates (optional)

    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return int(self.x.shape[0])

    @classmethod
    def _from_state(cls, state: GraphDataState) -> "GraphData":
        """Recreate an instance from its pickled state mapping."""

        return _graph_from_state(state)

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]: # type: ignore
        """Convert the graph to PyTorch tensors.

        Returns:
            node_features: Tensor of shape (N, F).
            adjacency: Dense adjacency tensor of shape (N, N).
        """
        if torch is None:  # pragma: no cover - optional dependency
            raise ImportError("PyTorch is required for tensor operations")

        def _ensure_cpu_tensor(value, *, dtype):
            if torch.is_tensor(value):
                tensor = value.detach()
                if tensor.device.type != "cpu":
                    tensor = tensor.to(device="cpu")
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
                return tensor
            return torch.as_tensor(value, dtype=dtype)

        x = _ensure_cpu_tensor(self.x, dtype=torch.float32)
        n = int(x.shape[0])
        adj = torch.zeros((n, n), dtype=torch.float32)

        edge_index = getattr(self, "edge_index", None)
        if edge_index is not None:
            idx = _ensure_cpu_tensor(edge_index, dtype=torch.long)
            if idx.numel() > 0:
                i = idx[0]
                j = idx[1]
                adj[i, j] = 1.0

        return x, adj

    # ``torch.utils.data`` uses ``ForkingPickler`` to serialise dataset items when
    # ``num_workers > 0`` (particularly under the ``spawn`` start method).
    # ``ForkingPickler`` first resolves ``GraphData`` from ``data.mdataset`` and
    # then compares it with the class referenced by each instance. Any module
    # reload or aliasing (common in tests) confuses that identity check and
    # raises ``PicklingError``. Implementing ``__reduce__`` sidesteps the lookup
    # by serialising to a lightweight mapping that ``_graph_from_state`` rebuilds.
    # Returning the module-level helper keeps the pickler from needing to
    # serialise the ``GraphData`` class itself, so stale class objects created
    # before a module reload remain picklable.

    def __getstate__(self) -> GraphDataState:
        return _graph_to_state(self)

    def __setstate__(self, state: GraphDataState) -> None:
        restored = _graph_from_state(state)
        self.x = restored.x
        self.edge_index = restored.edge_index
        self.edge_attr = restored.edge_attr
        self.pos = restored.pos

    def __reduce__(self):
        return (rebuild_graph_data, (self.__getstate__(),))

    def __reduce_ex__(self, protocol):
        return self.__reduce__()


# Explicitly anchor the public module path to avoid confusing ``spawn`` workers
# that import ``data`` via different aliases (e.g. after monkeypatching during
# tests). ``__qualname__`` already resolves to ``"GraphData"`` but setting the
# module guards against ``GraphData`` being re-exported through helper modules.
GraphData.__module__ = "data.mdataset"
register_graph_class(GraphData)

__all__ = [
    "GraphData",
    "GraphDataset",
    "_graph_from_state",
    "_graph_to_state",
]


class _SmilesGraphWorker:
    """Callable wrapper that survives pickling across ``spawn`` workers."""

    def __call__(
        self,
        smiles: str,
        add_3d: bool,
        random_seed: Optional[int],
        cls_module: str,
        cls_qualname: str,
    ) -> Optional[GraphDataState]:
        try:
            dataset_cls = _resolve_graphdataset_cls(cls_module, cls_qualname)
            graph = dataset_cls.smiles_to_graph(
                smiles, add_3d=add_3d, random_seed=random_seed
            )
        except Exception:
            return None

        return _graph_to_state(graph)

    def __reduce__(self):
        """Rebuild via the class constructor to keep pickling deterministic."""

        return (self.__class__, ())


_SMILES_GRAPH_WORKER = _SmilesGraphWorker()


def _recommended_chunksize(num_items: int, worker_budget: int) -> int:
    """Return a coarse-grained chunksize for ``ProcessPoolExecutor.map``."""

    if num_items <= 0 or worker_budget <= 0:
        return 1
    target = max(64, num_items // max(1, worker_budget * 4))
    return min(2048, target)


def _iter_graph_states(
    smiles: Sequence[str],
    *,
    add_3d: bool,
    random_seed: Optional[int],
    worker_budget: int,
    cls_module: str,
    cls_qualname: str,
) -> Iterator[Tuple[int, Optional[GraphDataState]]]:
    """Yield ``(index, GraphDataState | None)`` for each SMILES string."""

    if worker_budget > 0 and smiles:
        chunksize = _recommended_chunksize(len(smiles), worker_budget)
        worker = _SMILES_GRAPH_WORKER
        with ProcessPoolExecutor(max_workers=worker_budget) as ex:
            iterator = ex.map(
                worker,
                smiles,
                repeat(add_3d),
                repeat(random_seed),
                repeat(cls_module),
                repeat(cls_qualname),
                chunksize=chunksize,
            )
            for idx, g_state in enumerate(iterator):
                yield idx, g_state
    else:
        dataset_cls = _resolve_graphdataset_cls(cls_module, cls_qualname)
        for idx, sm in enumerate(smiles):
            try:
                graph = dataset_cls.smiles_to_graph(
                    sm, add_3d=add_3d, random_seed=random_seed
                )
            except Exception:
                yield idx, None
                continue
            yield idx, _graph_to_state(graph)

class GraphDataset:
    def __init__(
        self,
        graphs: List[GraphData],
        labels: Optional[Sequence] = None,
        smiles: Optional[List[str]] = None,
    ):
        self.graphs = graphs
        self.labels = np.asarray(labels) if labels is not None else None
        if self.labels is not None:
            if self.labels.ndim != 1 or self.labels.shape[0] != len(self.graphs):
                raise ValueError("labels must be 1D and the same length as graphs")

        self.smiles = smiles

        self._normalise_feature_dims()
        self._normalise_edge_dims()
        self._ensure_pos_consistency()

        self._schema_stats = self._compute_schema_stats()
        self._validate_schema_stats(self._schema_stats)
        self.node_dim = int(self._schema_stats.get("node_dim", 0))
        self.edge_dim = int(self._schema_stats.get("edge_dim", 0))
        self.schema_token = self._schema_stats.get("schema_token")
        logger.debug(
            "Initialized GraphDataset with %d graphs%s (node_dim=%s edge_dim=%s has_3d=%d)",
            len(graphs),
            " and labels" if labels is not None else "",
            self.node_dim,
            self.edge_dim,
            int(self._schema_stats.get("graphs_with_3d", 0)),
        )

    def _normalise_feature_dims(self) -> None:
        """Ensure all graphs expose the same node feature dimensionality."""

        if not self.graphs:
            return

        feature_dims: List[int] = []
        max_dim = 0
        for graph in self.graphs:
            x = getattr(graph, "x", None)
            if x is None:
                continue
            try:
                width = int(np.asarray(x).shape[1])
            except Exception:
                continue
            feature_dims.append(width)
            if width > max_dim:
                max_dim = width

        unique_dims = sorted({dim for dim in feature_dims if dim > 0})
        if max_dim <= 0 or len(unique_dims) <= 1:
            return

        logger.warning(
            "Normalising non-uniform node feature dims %s by padding to %d",
            unique_dims,
            max_dim,
        )

        for graph in self.graphs:
            x = getattr(graph, "x", None)
            if x is None:
                continue
            arr = np.asarray(x)
            if arr.ndim != 2:
                continue
            width = int(arr.shape[1])
            if width == max_dim:
                if not isinstance(x, np.ndarray):
                    graph.x = arr.astype(np.float32, copy=False)
                continue
            if width <= 0 or width > max_dim:
                continue

            pad_width = max_dim - width
            pad = np.zeros((arr.shape[0], pad_width), dtype=arr.dtype)
            padded = np.concatenate([arr, pad], axis=1)
            graph.x = padded.astype(np.float32, copy=False)

    def _ensure_pos_consistency(self) -> None:
        """Ensure graphs that expect coordinates expose ``pos`` arrays."""

        if not self.graphs:
            return

        pos_entries: List[Tuple[GraphData, np.ndarray]] = []
        max_width = 0
        any_pos = False

        for graph in self.graphs:
            pos_field = getattr(graph, "pos", None)
            if pos_field is None:
                continue

            try:
                if torch is not None and torch.is_tensor(pos_field):  # type: ignore[truthy-function]
                    arr = pos_field.detach().cpu().numpy()
                else:
                    arr = np.asarray(pos_field)
            except Exception:
                continue

            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                try:
                    arr = arr.reshape(arr.shape[0], -1)
                except Exception:
                    continue

            if arr.size == 0:
                continue

            any_pos = True
            width = int(arr.shape[1]) if arr.ndim == 2 else 0
            if width > max_width:
                max_width = width
            pos_entries.append((graph, arr))

        if not any_pos:
            return

        target_width = max(3, max_width)

        for graph, arr in pos_entries:
            width = int(arr.shape[1]) if arr.ndim == 2 else 0
            adjusted = arr
            if width > target_width:
                adjusted = arr[:, :target_width]
            elif width < target_width:
                pad = np.zeros((arr.shape[0], target_width - width), dtype=arr.dtype)
                adjusted = np.concatenate([arr, pad], axis=1)

            if adjusted.dtype != np.float32:
                adjusted = adjusted.astype(np.float32, copy=False)

            graph.pos = adjusted

        for graph in self.graphs:
            if getattr(graph, "pos", None) is not None:
                continue

            try:
                num_nodes = int(graph.num_nodes())
            except Exception:
                x_field = getattr(graph, "x", None)
                try:
                    arr = np.asarray(x_field)
                    num_nodes = int(arr.shape[0]) if arr.ndim >= 1 else 0
                except Exception:
                    num_nodes = 0

            graph.pos = np.zeros((num_nodes, target_width), dtype=np.float32)

    def _normalise_edge_dims(self) -> None:
        """Pad or truncate edge attributes so all graphs share a common width."""

        if not self.graphs:
            return

        edge_dims: List[int] = []
        arrays: List[Tuple[GraphData, np.ndarray]] = []

        for graph in self.graphs:
            edge_attr = getattr(graph, "edge_attr", None)
            if edge_attr is None:
                continue
            try:
                if torch is not None and torch.is_tensor(edge_attr):  # type: ignore[truthy-function]
                    arr = edge_attr.detach().cpu().numpy()
                else:
                    arr = np.asarray(edge_attr)
            except Exception:
                continue

            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim != 2 or arr.size == 0:
                continue

            arrays.append((graph, arr))
            edge_dims.append(int(arr.shape[1]))

        if not edge_dims:
            return

        max_dim = max(edge_dims)
        target_dim = max_dim
        if max_dim not in {EDGE_BASE_DIM, EDGE_TOTAL_DIM}:
            if max_dim < EDGE_BASE_DIM:
                target_dim = EDGE_BASE_DIM
            elif EDGE_BASE_DIM < max_dim <= EDGE_TOTAL_DIM:
                target_dim = EDGE_TOTAL_DIM
            else:
                logger.warning(
                    "Edge attributes wider than expected (max=%d); truncating to %d",
                    max_dim,
                    EDGE_TOTAL_DIM,
                )
                target_dim = EDGE_TOTAL_DIM

        adjusted = False
        for graph, arr in arrays:
            width = int(arr.shape[1])
            new_arr = arr
            if width > target_dim:
                new_arr = arr[:, :target_dim]
                adjusted = True
            elif width < target_dim:
                pad = np.zeros((arr.shape[0], target_dim - width), dtype=arr.dtype)
                new_arr = np.concatenate([arr, pad], axis=1)
                adjusted = True

            if new_arr.dtype != np.float32:
                new_arr = new_arr.astype(np.float32, copy=False)

            graph.edge_attr = new_arr

        if adjusted:
            logger.warning(
                "Normalised edge attribute dimensions to %d (original unique widths: %s)",
                target_dim,
                sorted(set(edge_dims)),
            )

    def _compute_schema_stats(self) -> Dict[str, Any]:
        """Collect dimensionality statistics for nodes and edges."""

        node_dims: List[int] = []
        edge_dims: List[int] = []
        node_examples: Dict[int, int] = {}
        edge_examples: Dict[int, int] = {}
        graphs_with_3d = 0
        edges_with_attr = 0
        edges_with_3d = 0

        for idx, graph in enumerate(self.graphs):
            x = getattr(graph, "x", None)
            node_dim = 0
            if x is not None:
                try:
                    node_dim = int(np.asarray(x).shape[1])
                except Exception:
                    node_dim = 0
            node_dims.append(node_dim)
            if node_dim > 0 and node_dim not in node_examples:
                node_examples[node_dim] = idx

            edge_attr = getattr(graph, "edge_attr", None)
            arr = None
            edge_dim = 0
            if edge_attr is not None:
                arr = np.asarray(edge_attr)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                if arr.ndim == 2 and arr.size > 0:
                    edge_dim = int(arr.shape[1])
            edge_dims.append(edge_dim)
            if edge_dim > 0 and edge_dim not in edge_examples:
                edge_examples[edge_dim] = idx

            if arr is not None and arr.ndim == 2 and arr.size > 0:
                edges_with_attr += int(arr.shape[0])
                flag_idx = EDGE_BASE_DIM
                if arr.shape[1] > flag_idx:
                    has_flag = bool(np.any(arr[:, flag_idx] > 0.5))
                    graphs_with_3d += int(has_flag)
                    edges_with_3d += int(np.count_nonzero(arr[:, flag_idx] > 0.5))

        node_dim = max((d for d in node_dims if d > 0), default=0)
        edge_dim = max((d for d in edge_dims if d > 0), default=0)

        schema_parts = [f"n{node_dim}", f"e{edge_dim}", GRAPH_SCHEMA_VERSION]
        token = "_".join(schema_parts)

        return {
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "node_dims": node_dims,
            "edge_dims": edge_dims,
            "node_examples": node_examples,
            "edge_examples": edge_examples,
            "graphs_with_3d": graphs_with_3d,
            "edges_with_attr": edges_with_attr,
            "edges_with_3d": edges_with_3d,
            "schema_token": token,
            "version": GRAPH_SCHEMA_VERSION,
        }

    def _validate_schema_stats(self, stats: Dict[str, Any]) -> None:
        """Raise descriptive errors when node/edge dims disagree."""

        node_dims = sorted({d for d in stats.get("node_dims", []) if d > 0})
        edge_dims = sorted({d for d in stats.get("edge_dims", []) if d > 0})

        if len(node_dims) > 1:
            samples = stats.get("node_examples", {})
            sample_msgs = []
            for dim in node_dims:
                idx = samples.get(dim)
                label = None
                if idx is not None and self.smiles is not None and idx < len(self.smiles):
                    label = self.smiles[idx]
                entry = f"{dim} (idx={idx}" + (f" smiles={label}" if label else "") + ")"
                sample_msgs.append(entry)
            raise ValueError(
                "Node feature dimension mismatch detected: "
                + ", ".join(sample_msgs)
            )

        if len(edge_dims) > 1:
            samples = stats.get("edge_examples", {})
            sample_msgs = []
            for dim in edge_dims:
                idx = samples.get(dim)
                label = None
                if idx is not None and self.smiles is not None and idx < len(self.smiles):
                    label = self.smiles[idx]
                entry = f"{dim} (idx={idx}" + (f" smiles={label}" if label else "") + ")"
                sample_msgs.append(entry)
            raise ValueError(
                "Edge feature dimension mismatch detected: "
                + ", ".join(sample_msgs)
            )

        if edge_dims:
            allowed_dims = {0, EDGE_BASE_DIM, EDGE_TOTAL_DIM}
            invalid = [dim for dim in edge_dims if dim not in allowed_dims]
            if invalid:
                raise ValueError(
                    "Unexpected edge_dim values: "
                    + ", ".join(str(dim) for dim in sorted(set(invalid)))
                    + f" (allowed: {sorted(allowed_dims)})"
                )

    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx):
        g = self.graphs[idx]
        return (g, self.labels[idx]) if self.labels is not None else g

    def close(self) -> None:
        """Release cached graph data and labels to help free memory."""

        self.graphs.clear()
        self.labels = None
        self.smiles = None

    def get_batch(
        self, indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: # type: ignore
        """Construct a mini-batch of graphs.

        Graphs are combined by block-diagonal stacking of their adjacency
        matrices and concatenation of node features.

        Args:
            indices: Indices of graphs to include in the batch.

        Returns:
            batch_x: Tensor of shape (sum_i N_i, F) containing all node features.
            batch_adj: Dense block-diagonal adjacency matrix.
            batch_ptr: Tensor marking graph boundaries within the batch.
            batch_labels: Labels tensor if dataset is labelled, else ``None``.
        """
        if torch is None:  # pragma: no cover - optional dependency
            raise ImportError("PyTorch is required for batching graphs")

        if not indices:
            logger.error("get_batch called with empty indices")
            raise ValueError("Empty batch indices")
        if max(indices) >= len(self.graphs):
            logger.error(
                "get_batch index out of bounds: max %d vs %d graphs",
                max(indices),
                len(self.graphs),
            )
            raise IndexError(
                f"Index out of bounds: max {max(indices)} vs {len(self.graphs)} graphs"
            )

        node_features: List[torch.Tensor] = [] # type: ignore
        adj_blocks: List[torch.Tensor] = [] # type: ignore
        sizes: List[int] = []
        for idx in indices:
            x_i, adj_i = self.graphs[idx].to_tensors()
            n_i = int(x_i.size(0))
            if n_i <= 0:
                logger.error("Graph at idx %d has 0 nodes", idx)
                raise ValueError(f"Graph at idx {idx} has 0 nodes")
            node_features.append(x_i)
            adj_blocks.append(adj_i)
            sizes.append(n_i)

        if node_features:
            feat_dims = {int(x_i.size(1)) for x_i in node_features}
            if len(feat_dims) > 1:
                max_feat_dim = max(feat_dims)
                logger.warning(
                    "Non-uniform node feature dims %s encountered; padding to %d",
                    sorted(feat_dims),
                    max_feat_dim,
                )
                padded_features: List[torch.Tensor] = []
                for idx, x_i in zip(indices, node_features):
                    feat_dim = int(x_i.size(1))
                    if feat_dim < max_feat_dim:
                        pad = torch.zeros(
                            (x_i.size(0), max_feat_dim - feat_dim),
                            dtype=x_i.dtype,
                            device=x_i.device,
                        )
                        x_i = torch.cat([x_i, pad], dim=1)
                    padded_features.append(x_i)
                node_features = padded_features
            batch_x = torch.cat(node_features, dim=0)
        else:
            batch_x = torch.zeros((0, self.graphs[0].x.shape[1]), dtype=torch.float32)
        batch_adj = (
            torch.block_diag(*adj_blocks)
            if adj_blocks
            else torch.zeros((0, 0), dtype=torch.float32)
        )
        ptr = np.cumsum([0] + sizes, dtype=np.int64)
        if np.any(np.diff(ptr) <= 0) or (len(ptr) != len(indices) + 1):
            raise AssertionError(
                f"Bad batch_ptr: {ptr.tolist()} for {len(indices)} graphs"
            )
        batch_ptr = torch.tensor(ptr, dtype=torch.long)

        batch_labels = (
            torch.tensor(self.labels[indices], dtype=torch.float32)
            if self.labels is not None
            else None
        )
        if batch_labels is not None:
            assert batch_labels.shape[0] == len(
                indices
            ), f"Labels length {batch_labels.shape[0]} != indices length {len(indices)}"

        return batch_x, batch_adj, batch_ptr, batch_labels

    @property
    def schema_metadata(self) -> Dict[str, Any]:
        return {
            "node_dim": int(self.node_dim),
            "edge_dim": int(self.edge_dim),
            "cache_version": GRAPH_CACHE_VERSION,
            "schema_token": self.schema_token,
            "version": GRAPH_SCHEMA_VERSION,
            "num_graphs": len(self.graphs),
            "graphs_with_3d": self._schema_stats.get("graphs_with_3d", 0),
            "edges_with_attr": self._schema_stats.get("edges_with_attr", 0),
            "edges_with_3d": self._schema_stats.get("edges_with_3d", 0),
        }

    def validate_cached_schema(
        self, schema_meta: Optional[Dict[str, Any]], *, source: str = "<cache>"
    ) -> None:
        if not schema_meta:
            raise ValueError(f"Cache {source} missing schema metadata")
        cache_version = schema_meta.get("cache_version")
        if cache_version != GRAPH_CACHE_VERSION:
            raise ValueError(
                f"Cache {source} cache version {cache_version} != {GRAPH_CACHE_VERSION}"
            )
        version = schema_meta.get("version")
        if version != GRAPH_SCHEMA_VERSION:
            raise ValueError(
                f"Cache {source} schema version {version} != {GRAPH_SCHEMA_VERSION}"
            )
        cached_node = int(schema_meta.get("node_dim", -1))
        cached_edge = int(schema_meta.get("edge_dim", -1))
        mismatches: List[str] = []
        if cached_node != int(self.node_dim):
            mismatches.append(f"node_dim cached={cached_node} actual={self.node_dim}")
        if cached_edge != int(self.edge_dim):
            mismatches.append(f"edge_dim cached={cached_edge} actual={self.edge_dim}")
        if mismatches:
            raise ValueError(
                f"Cache {source} schema mismatch: " + "; ".join(mismatches)
            )

    # ---------- Core featurisation ---------- #
    @staticmethod
    def smiles_to_graph(
        smiles: str, add_3d: bool = False, random_seed: Optional[int] = None
    ) -> GraphData:
        """
        Convert SMILES -> GraphData using RDKit when available.
        Falls back to a small synthetic graph if RDKit/embedding fails.
        """
        # --- graceful fallback if RDKit missing ---
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except Exception:
            logger.debug("RDKit not available; using fallback graph for %s", smiles)
            return _fallback_graph_from_string(smiles, add_pos=add_3d)

        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                logger.warning("Invalid SMILES %s", smiles)
                raise ValueError("Invalid SMILES string")

            # sanitization + explicit H (safer for 3D)
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                )
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception as e:
                # log and re‑raise; caller will decide whether to skip
                logger.warning(
                    "Sanitization/Kekulization failed for %s: %s",
                    smiles,
                    e,
                )
                raise ValueError("Sanitization/Kekulization failed")


            mol = Chem.AddHs(mol)

            # Node features: [Z, degree, aromatic, hybrid]
            feats = []
            for atom in mol.GetAtoms():
                z = atom.GetAtomicNum()
                deg = atom.GetDegree()
                aromatic = int(atom.GetIsAromatic())
                # store hybridization as small int code
                hybrid = int(atom.GetHybridization())
                feats.append([z, deg, aromatic, hybrid])

            X = (
                np.asarray(feats, dtype=np.float32)
                if feats
                else np.zeros((0, 4), dtype=np.float32)
            )

            coords = None
            has_3d = False
            if add_3d and mol.GetNumAtoms() > 0:
                coords, _ = _generate_conformer_coords(
                    mol,
                    smiles=smiles,
                    random_seed=random_seed,
                )
                if coords is not None and coords.shape[0] == X.shape[0]:
                    X = np.concatenate([X, coords], axis=1)
                    has_3d = True
                else:
                    coords = None

            # Edges + attrs
            edges: list[tuple[int, int]] = []
            base_attrs: list[list[float]] = []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                btype = b.GetBondType()
                onehot = [
                    int(btype == Chem.BondType.SINGLE),
                    int(btype == Chem.BondType.DOUBLE),
                    int(btype == Chem.BondType.TRIPLE),
                    int(btype == Chem.BondType.AROMATIC),
                ]
                conj = int(b.GetIsConjugated())
                ring = int(b.IsInRing())
                length = 0.0
                if coords is not None:
                    vi, vj = coords[i], coords[j]
                    length = float(np.linalg.norm(vi - vj))

                feat = onehot + [conj, ring, length]
                # undirected → add both directions
                edges.append((i, j))
                base_attrs.append(feat)
                edges.append((j, i))
                base_attrs.append(feat)

            E = (
                np.array(edges, dtype=np.int64).T
                if edges
                else np.zeros((2, 0), dtype=np.int64)
            )
            EA = None
            if base_attrs:
                base_arr = np.asarray(base_attrs, dtype=np.float32)
                if base_arr.ndim == 1:
                    base_arr = base_arr.reshape(-1, EDGE_BASE_DIM)
                if base_arr.shape[1] < EDGE_BASE_DIM:
                    pad = np.zeros(
                        (base_arr.shape[0], EDGE_BASE_DIM - base_arr.shape[1]),
                        dtype=base_arr.dtype,
                    )
                    base_arr = np.concatenate([base_arr, pad], axis=1)

                if not add_3d:
                    EA = base_arr
                else:
                    flag = np.full(
                        (base_arr.shape[0], 1), 1.0 if has_3d else 0.0, dtype=np.float32
                    )
                    geom = np.zeros(
                        (base_arr.shape[0], EDGE_GEOM_DIM), dtype=np.float32
                    )
                    if has_3d and E.shape[1] > 0:
                        try:
                            geom = _append_geom_edge_attr(mol, E, None)
                        except Exception:
                            geom = np.zeros(
                                (base_arr.shape[0], EDGE_GEOM_DIM), dtype=np.float32
                            )
                    if geom.ndim == 1 and base_arr.shape[0] > 0:
                        geom = geom.reshape(base_arr.shape[0], EDGE_GEOM_DIM)
                    if geom.shape[0] != base_arr.shape[0]:
                        gpad = np.zeros(
                            (base_arr.shape[0], EDGE_GEOM_DIM), dtype=np.float32
                        )
                        if geom.shape[0] > 0:
                            m = min(base_arr.shape[0], geom.shape[0])
                            gpad[:m] = geom[:m]
                        geom = gpad
                    EA = np.concatenate([base_arr, flag, geom], axis=1)

            pos: Optional[np.ndarray]
            if add_3d:
                if coords is None:
                    width = 3
                    num_nodes = int(X.shape[0]) if X.ndim == 2 else 0
                    pos = np.zeros((num_nodes, width), dtype=np.float32)
                else:
                    pos = coords
            else:
                pos = coords

            return GraphData(x=X, edge_index=E, edge_attr=EA, pos=pos)

        except Exception:
            # any unexpected RDKit error → robust fallback
            return _fallback_graph_from_string(smiles, add_pos=add_3d)

    # ---------- Builders ---------- #
    @classmethod
    def from_smiles_list(
        cls,
        smiles_list: List[str],
        labels: Optional[List[Any]] = None,
        add_3d: bool = False,
        random_seed: Optional[int] = None,
    ) -> "GraphDataset":
        graphs: List[GraphData] = []
        smiles_out: List[str] = []
        valid_indices: List[int] = []
        for i, sm in enumerate(smiles_list):
            try:
                g = cls.smiles_to_graph(sm, add_3d=add_3d, random_seed=random_seed)
            except Exception as e:
                # Dont raise exception as it may be a valid SMILES
                # if _RUNNING_IN_CI:
                # raise
                logger.warning("Skipping invalid SMILES %s: %s", sm, e)
                continue

            graphs.append(g)
            smiles_out.append(sm)
            valid_indices.append(i)

        y = None if labels is None else np.asarray(labels)
        # If labels are provided, filter them to match valid SMILES
        if y is not None:
            y = y[valid_indices]

        return cls(graphs, y, smiles_out)

    @classmethod
    def from_parquet(
        cls,
        filepath: str,
        smiles_col: str = "smiles",
        label_col: Optional[str] = None,
        cache_dir: Optional[str] = None,
        add_3d: bool = False,
        random_seed: Optional[int] = None,
        n_rows: Optional[int] = None,  # subset helper
        num_workers: int = 0,
    ) -> "GraphDataset":
        """Load a dataset from a Parquet file of SMILES.

        The file is read with pandas and each SMILES string is converted
        to GraphData via ``smiles_to_graph``. Optionally the featurisation
        runs in a process pool when ``num_workers > 0``.
        """
        cache_path = None
        if cache_dir and n_rows is None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = os.path.splitext(os.path.basename(filepath))[0]
            cache_name = f"{cache_name}_{_cache_schema_suffix(add_3d)}.pkl"
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.exists(cache_path):
                logger.info("Loading graphs from cache %s", cache_path)
                payload = _load_graph_cache(cache_path)
                if payload is not None:
                    try:
                        ds_cached = cls(
                            payload.get("graphs", []),
                            payload.get("labels"),
                            None,
                        )
                        ds_cached.validate_cached_schema(
                            payload.get("schema"), source=cache_path
                        )
                        return ds_cached
                    except Exception as exc:
                        logger.warning(
                            "Cached dataset %s invalid (%s); rebuilding",
                            cache_path,
                            exc,
                        )

        cols = [smiles_col] + ([label_col] if label_col else [])
        df = pd.read_parquet(filepath, columns=cols) 
        if n_rows is not None:
            df = df.head(int(n_rows))
        smiles = df[smiles_col].astype(str).tolist()
        labels = (
            df[label_col].to_numpy()
            if (label_col and label_col in df.columns)
            else None
        )

        graphs: List[GraphData] = []
        smiles_out: List[str] = []
        valid_indices: List[int] = []

        # Resolve the dataset class lazily in worker processes to avoid
        # pickling GraphData/GraphDataset objects when spawning the pool.
        cls_module = cls.__module__
        cls_qualname = cls.__qualname__

        worker_budget = _resolve_worker_count(num_workers)
        state_iter = _iter_graph_states(
            smiles,
            add_3d=add_3d,
            random_seed=random_seed,
            worker_budget=worker_budget,
            cls_module=cls_module,
            cls_qualname=cls_qualname,
        )
        for i, g_state in state_iter:
            if g_state is not None:
                graphs.append(_graph_from_state(g_state))
                smiles_out.append(smiles[i])
                valid_indices.append(i)
        # Filter labels to match valid graphs
        if labels is not None:
            labels = labels[valid_indices]
            if len(labels) != len(graphs):
                raise ValueError(
                    f"Mismatch: {len(graphs)} graphs vs {len(labels)} labels"
                )

        if add_3d:
            min_dim = 7  # base atom features (4) plus xyz
            for g in graphs:
                if g.x.shape[1] < min_dim:
                    pad = np.zeros((g.x.shape[0], min_dim - g.x.shape[1]), dtype=g.x.dtype)
                    g.x = np.concatenate([g.x, pad], axis=1)
        dataset = cls(graphs, labels, smiles_out)
        if cache_path:
            _write_graph_cache(cache_path, dataset)

        return dataset

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        smiles_col: str = "smiles",
        label_col: Optional[str] = None,
        sep: str = ",",
        cache_dir: Optional[str] = None,
        add_3d: bool = False,
        random_seed: Optional[int] = None,
        n_rows: Optional[int] = None,
        num_workers: Optional[int] = 0,
    ) -> "GraphDataset":
        cache_path = None
        if cache_dir and n_rows is None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = os.path.splitext(os.path.basename(filepath))[0]
            cache_name = f"{cache_name}_{_cache_schema_suffix(add_3d)}.pkl"
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.exists(cache_path):
                logger.info("Loading graphs from cache %s", cache_path)
                payload = _load_graph_cache(cache_path)
                if payload is not None:
                    try:
                        ds_cached = cls(
                            payload.get("graphs", []),
                            payload.get("labels"),
                            None,
                        )
                        ds_cached.validate_cached_schema(
                            payload.get("schema"), source=cache_path
                        )
                        return ds_cached
                    except Exception as exc:
                        logger.warning(
                            "Cached dataset %s invalid (%s); rebuilding",
                            cache_path,
                            exc,
                        )

        df = pd.read_csv(filepath, sep=sep)
        if n_rows is not None:
            df = df.head(int(n_rows))
        smiles = df[smiles_col].astype(str).tolist()
        labels_raw = (
            df[label_col].to_numpy()
            if (label_col and label_col in df.columns)
            else None
        )

        graphs: List[GraphData] = []
        smiles_out: List[str] = []
        valid_indices: List[int] = []

        cls_module = cls.__module__
        cls_qualname = cls.__qualname__

        worker_budget = _resolve_worker_count(num_workers if num_workers is not None else -1)
        state_iter = _iter_graph_states(
            smiles,
            add_3d=add_3d,
            random_seed=random_seed,
            worker_budget=worker_budget,
            cls_module=cls_module,
            cls_qualname=cls_qualname,
        )
        for idx, g_state in state_iter:
            if g_state is not None:
                graphs.append(_graph_from_state(g_state))
                smiles_out.append(smiles[idx])
                valid_indices.append(idx)

        labels = None
        if labels_raw is not None:
            labels_array = np.asarray(labels_raw)
            labels = labels_array[valid_indices]

        dataset = cls(graphs, labels, smiles_out)
        if cache_path:
            _write_graph_cache(cache_path, dataset)

        return dataset

    @classmethod
    def from_directory(
        cls,
        dirpath: str,
        ext: str = "parquet",
        smiles_col: str = "smiles",
        label_col: Optional[str] = None,
        cache_dir: Optional[str] = None,
        add_3d: bool = False,
        random_seed: Optional[int] = None,
        prefix_filter: Optional[str] = None,
        n_rows_per_file: Optional[int] = None,  # subset helper
        max_graphs: Optional[int] = None,
        num_workers: int = 0,
    ) -> "GraphDataset":
        graphs_all: List[GraphData] = []
        labels_all: List[Any] = []
        smiles_all: List[str] = []
        labels_present = False

        files = [
            f for f in os.listdir(dirpath) if f.lower().endswith(f".{ext.lower()}")
        ]
        files.sort()
        if prefix_filter:
            files = [f for f in files if os.path.basename(f).startswith(prefix_filter)]

        for fname in files:
            if max_graphs is not None and len(graphs_all) >= max_graphs:
                break
            path = os.path.join(dirpath, fname)
            remaining = None if max_graphs is None else max_graphs - len(graphs_all)
            n_rows = n_rows_per_file
            if remaining is not None:
                n_rows = remaining if n_rows is None else min(n_rows, remaining)
            if ext.lower() == "parquet":
                ds = cls.from_parquet(
                    filepath=path,
                    smiles_col=smiles_col,
                    label_col=label_col,
                    cache_dir=(
                        None
                        if n_rows_per_file
                        else (
                            None
                            if cache_dir is None
                            else os.path.join(
                                cache_dir,
                                f"{os.path.splitext(fname)[0]}_{_cache_schema_suffix(add_3d)}",
                            )
                        )
                    ),
                    add_3d=add_3d,
                    random_seed=random_seed,
                    n_rows=n_rows,
                    num_workers=num_workers,
                )
            elif ext.lower() == "csv":
                ds = cls.from_csv(
                    filepath=path,
                    smiles_col=smiles_col,
                    label_col=label_col,
                    cache_dir=(
                        None
                        if n_rows_per_file
                        else (
                            None
                            if cache_dir is None
                            else os.path.join(
                                cache_dir,
                                f"{os.path.splitext(fname)[0]}_{_cache_schema_suffix(add_3d)}",
                            )
                        )
                    ),
                    add_3d=add_3d,
                    random_seed=random_seed,
                    n_rows=n_rows,
                )
            else:
                raise ValueError(f"Unsupported ext: {ext}")

            graphs_all.extend(ds.graphs)
            smiles_all.extend(ds.smiles or [])
            if ds.labels is not None:
                labels_present = True
                labels_all.extend(ds.labels.tolist())

        if max_graphs is not None:
            graphs_all = graphs_all[:max_graphs]
            smiles_all = smiles_all[:max_graphs]
            labels_all = labels_all[:max_graphs]

        labels = np.asarray(labels_all) if labels_present else None
        smiles = smiles_all if smiles_all else None
        return cls(graphs_all, labels, smiles)


# ------------------ Geometry helpers (angles/dihedrals) ------------------ #


def _stable_smiles_seed(smiles: str) -> int:
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _generate_conformer_coords(
    mol: "Chem.Mol",
    *,
    smiles: str,
    random_seed: Optional[int],
    max_attempts: int = 4,
) -> Tuple[Optional[np.ndarray], bool]:
    if mol.GetNumAtoms() == 0:
        return None, False

    seeds: List[int] = []
    if random_seed is not None:
        seeds.append(int(random_seed))
    else:
        seeds.append(_stable_smiles_seed(smiles))
    seeds.extend([seeds[0] + 97, seeds[0] + 193, 0xF00D])

    attempts = 0
    for seed in seeds:
        if attempts >= max_attempts:
            break
        attempts += 1
        try:
            mol.RemoveAllConformers()
        except Exception:
            pass
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        params.randomSeed = int(seed) & 0xFFFFFFFF
        try:
            rc = AllChem.EmbedMolecule(mol, params)
        except Exception:
            rc = -1
        if rc != 0:
            continue
        optimised = False
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                optimised = True
        except Exception:
            optimised = False
        if not optimised:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
        try:
            conf = mol.GetConformer()
        except Exception:
            continue
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        if coords.shape[0] == mol.GetNumAtoms():
            return coords, True

    try:
        mol.RemoveAllConformers()
    except Exception:
        pass
    return None, False


def _embed_single_conformer(mol: Chem.Mol, max_attempts: int = 2) -> bool:
    if mol.GetNumConformers() > 0:
        return True
    base_seed = 0xF00D
    for attempt in range(max_attempts):
        seed = base_seed + attempt * 97
        try:
            mol.RemoveAllConformers()
        except Exception:
            pass
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        params.randomSeed = int(seed) & 0xFFFFFFFF
        try:
            rc = AllChem.EmbedMolecule(mol, params)
        except Exception:
            rc = -1
        if rc != 0:
            continue
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=100)
        except Exception:
            pass
        if mol.GetNumConformers() > 0:
            return True
    return False


def _pick_neighbor(mol: Chem.Mol, center: int, exclude: int) -> Optional[int]:
    ns = sorted(
        [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(center).GetNeighbors()
            if nbr.GetIdx() != exclude
        ]
    )
    return ns[0] if ns else None


def _geom_features_for_bond(
    mol: Chem.Mol, i: int, j: int, conf_id: int = 0
) -> np.ndarray:
    """
    10‑D per directed edge i->j:
    [d_ij,
     cos(∠k-i-j), sin(∠k-i-j), has_k,
     cos(∠i-j-l), sin(∠i-j-l), has_l,
     cos(φ k-i-j-l), sin(φ k-i-j-l), has_dihedral]
    """
    d = np.zeros(10, dtype=np.float32)
    if mol.GetNumConformers() == 0:
        return d
    conf = mol.GetConformer(conf_id)
    try:
        d[0] = float(MT.GetBondLength(conf, int(i), int(j)))
    except Exception:
        d[0] = 0.0

    k = _pick_neighbor(mol, i, j)
    l = _pick_neighbor(mol, j, i)
    if k is not None:
        try:
            ang = float(MT.GetAngleRad(conf, int(k), int(i), int(j)))
            d[1], d[2], d[3] = np.cos(ang), np.sin(ang), 1.0
        except Exception:
            pass
    if l is not None:
        try:
            ang = float(MT.GetAngleRad(conf, int(i), int(j), int(l)))
            d[4], d[5], d[6] = np.cos(ang), np.sin(ang), 1.0
        except Exception:
            pass
    if (k is not None) and (l is not None):
        try:
            dih = float(MT.GetDihedralRad(conf, int(k), int(i), int(j), int(l)))
            d[7], d[8], d[9] = np.cos(dih), np.sin(dih), 1.0
        except Exception:
            pass
    return d


def _append_geom_edge_attr(
    mol: Chem.Mol, edge_index: np.ndarray, edge_attr: Optional[np.ndarray]
) -> np.ndarray:
    _embed_single_conformer(mol)  # safe if it fails; zeros will be used
    feats = [_geom_features_for_bond(mol, int(i), int(j)) for i, j in edge_index.T]
    geom = np.stack(feats, axis=0) if feats else np.zeros((0, 10), dtype=np.float32)
    return geom if edge_attr is None else np.concatenate([edge_attr, geom], axis=1)


def _fallback_graph_from_string(s: str, add_pos: bool = False) -> "GraphData":
    """
    Deterministic tiny chain-graph from a string when RDKit isn't available.
    - N = max(2, min(10, len(s)))
    - Node feats: [position, position%3] as float32
    - Edges: linear chain i<->i+1 (both directions)
    """
    import numpy as _np

    n = max(2, min(10, len(s)))
    arange = _np.arange(n, dtype=_np.float32)
    x = _np.stack(
        [arange, (arange % 3).astype(_np.float32)],
        axis=1,
    )  # [N,2]
    rows = _np.concatenate([_np.arange(n - 1), _np.arange(1, n)], axis=0)
    cols = _np.concatenate([_np.arange(1, n), _np.arange(n - 1)], axis=0)
    edge_index = _np.stack([rows.astype(_np.int64), cols.astype(_np.int64)], axis=0)

    pos = None
    if add_pos:
        pos = _np.zeros((n, 3), dtype=_np.float32)
        pos[:, 0] = arange
        x = _np.concatenate([x, pos], axis=1)

    return GraphData(x=x, edge_index=edge_index, edge_attr=None, pos=pos)
