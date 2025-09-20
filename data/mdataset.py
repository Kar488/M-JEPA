from __future__ import annotations

import importlib
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import repeat
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

_RUNNING_IN_CI = os.getenv("CI") == "true"  # run local vs remote

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms as MT
except Exception as e:
    if _RUNNING_IN_CI:
        raise ImportError("RDKit is required in CI for SMILES parsing.")

import pandas as pd

logger = logging.getLogger(__name__)


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

        x = torch.as_tensor(self.x, dtype=torch.float32)
        n = x.shape[0]
        adj = torch.zeros((n, n), dtype=torch.float32)
        if self.edge_index.size > 0:
            i = torch.as_tensor(self.edge_index[0], dtype=torch.long)
            j = torch.as_tensor(self.edge_index[1], dtype=torch.long)
            adj[i, j] = 1.0
        return x, adj

    # ``torch.utils.data`` uses ``ForkingPickler`` to serialise dataset items when
    # ``num_workers > 0`` (particularly under the ``spawn`` start method).
    # ``ForkingPickler`` first resolves ``GraphData`` from ``data.mdataset`` and
    # then compares it with the class referenced by each instance. Any module
    # reload or aliasing (common in tests) confuses that identity check and
    # raises ``PicklingError``. Implementing ``__reduce__`` sidesteps the lookup
    # by serialising to a lightweight mapping that ``_graph_from_state`` rebuilds.
    # The classmethod ``_from_state`` simply delegates to this helper so cached
    # payloads from older versions remain compatible. ``__reduce__`` now returns
    # the classmethod directly instead of a module-level function so the
    # reloader only needs to resolve ``GraphData`` itself, avoiding fragile
    # attribute lookups under ``spawn`` workers.

    def __getstate__(self) -> GraphDataState:
        return _graph_to_state(self)

    def __setstate__(self, state: GraphDataState) -> None:
        restored = _graph_from_state(state)
        self.x = restored.x
        self.edge_index = restored.edge_index
        self.edge_attr = restored.edge_attr
        self.pos = restored.pos

    def __reduce__(self):
        return (self.__class__._from_state, (self.__getstate__(),))

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

__all__ = [
    "GraphData",
    "GraphDataset",
    "_graph_from_state",
    "_graph_to_state",
]


def _safe_smiles_to_graph(
    smiles: str,
    add_3d: bool,
    random_seed: Optional[int],
    cls_module: str,
    cls_qualname: str,
) -> Optional[GraphDataState]:
    """Helper for multiprocessing to convert a SMILES string into a graph."""

    try:
        dataset_cls = _resolve_graphdataset_cls(cls_module, cls_qualname)
        graph = dataset_cls.smiles_to_graph(
            smiles, add_3d=add_3d, random_seed=random_seed
        )
    except Exception:
        return None

    return _graph_to_state(graph)

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
        logger.debug(
            "Initialized GraphDataset with %d graphs%s",
            len(graphs),
            " and labels" if labels is not None else "",
        )

    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx):
        g = self.graphs[idx]
        return (g, self.labels[idx]) if self.labels is not None else g
    
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

        batch_x = (
            torch.cat(node_features, dim=0)
            if node_features
            else torch.zeros((0, self.graphs[0].x.shape[1]), dtype=torch.float32)
        )
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
            if add_3d and mol.GetNumAtoms() > 0:
                try:
                    params = AllChem.ETKDGv3()
                    if random_seed is not None:
                        params.randomSeed = int(random_seed)
                    # returns 0 on success; -1 on failure
                    rc = AllChem.EmbedMolecule(mol, params)
                    if rc == 0:
                        # geometry optimization is best-effort; failure is fine
                        try:
                            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                        except Exception:
                            pass
                        conf = mol.GetConformer()
                        coords = np.array(
                            [
                                list(conf.GetAtomPosition(i))
                                for i in range(mol.GetNumAtoms())
                            ],
                            dtype=np.float32,
                        )
                        # append (x,y,z)
                        if coords.shape[0] == X.shape[0]:
                            X = np.concatenate([X, coords], axis=1)
                except Exception:
                    logger.debug("3D embedding failed for %s", smiles)
                    coords = None
                if coords is None:
                    raise ValueError("3D embedding failed")

            # Edges + attrs
            edges: list[tuple[int, int]] = []
            eattr: list[list[float]] = []
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
                eattr.append(feat)
                edges.append((j, i))
                eattr.append(feat)

            E = (
                np.array(edges, dtype=np.int64).T
                if edges
                else np.zeros((2, 0), dtype=np.int64)
            )
            EA = np.asarray(eattr, dtype=np.float32) if eattr else None

            # optional geometric edge features
            if add_3d and E.shape[1] > 0:
                try:
                    EA = _append_geom_edge_attr(mol, E, EA)
                except Exception:
                    # if augmentation fails, keep existing EA (may be None)
                    pass

            return GraphData(x=X, edge_index=E, edge_attr=EA, pos=coords)

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
            cache_name = f"{cache_name}_3d{int(add_3d)}.pkl"  # clear old caches if needed
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.exists(cache_path):
                logger.info("Loading graphs from cache %s", cache_path)
                with open(cache_path, "rb") as f:
                    graphs, labels = pickle.load(f)
                return cls(graphs, labels, None)

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

        if num_workers > 0:
            with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
                iterator = ex.map(
                    _safe_smiles_to_graph,
                    smiles,
                    repeat(add_3d),
                    repeat(random_seed),
                    repeat(cls_module),
                    repeat(cls_qualname),
                )
                for i, g_state in enumerate(iterator):
                    if g_state is not None:
                        graphs.append(_graph_from_state(g_state))
                        smiles_out.append(smiles[i])
                        valid_indices.append(i)
        else:
            for i, sm in enumerate(smiles):
                try:
                    graph = cls.smiles_to_graph(
                        sm, add_3d=add_3d, random_seed=random_seed
                    )
                except Exception:
                    continue
                graphs.append(graph)
                smiles_out.append(sm)
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
        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump((graphs, labels), f)
            logger.info("Wrote graph cache to %s", cache_path)

        return cls(graphs, labels, smiles_out)

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
    ) -> "GraphDataset":
        cache_path = None
        if cache_dir and n_rows is None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = os.path.splitext(os.path.basename(filepath))[0]
            cache_name = f"{cache_name}_3d{int(add_3d)}.pkl"  # clear old caches if needed
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.exists(cache_path):
                logger.info("Loading graphs from cache %s", cache_path)
                with open(cache_path, "rb") as f:
                    graphs, labels = pickle.load(f)
                return cls(graphs, labels, None)

        df = pd.read_csv(filepath, sep=sep)
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
        for sm in smiles:
            try:
                g = cls.smiles_to_graph(sm, add_3d=add_3d, random_seed=random_seed)
                graphs.append(g)
                smiles_out.append(sm)
            except Exception:
                continue

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump((graphs, labels), f)
            logger.info("Wrote graph cache to %s", cache_path)

        return cls(graphs, labels, smiles_out)

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
                                cache_dir, f"{os.path.splitext(fname)[0]}_3d{int(add_3d)}"
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
                                cache_dir, f"{os.path.splitext(fname)[0]}_3d{int(add_3d)}"
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


def _embed_single_conformer(mol: Chem.Mol, max_attempts: int = 2) -> bool:
    if mol.GetNumConformers() > 0:
        return True
    p = AllChem.ETKDGv3()
    p.useSmallRingTorsions = True
    p.randomSeed = 0xF00D
    for _ in range(max_attempts):
        if AllChem.EmbedMolecule(mol, p) == 0:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=50)
            except Exception:
                pass
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
