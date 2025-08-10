from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

_RUNNING_IN_CI = os.getenv("CI") == "true" # run local vs remote

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms as MT
except Exception as e:
    if _RUNNING_IN_CI:
        raise ImportError("RDKit is required in CI for SMILES parsing.")

import pandas as pd



@dataclass
class GraphData:
    x: np.ndarray  # [num_nodes, feat_dim]
    edge_index: np.ndarray  # [2, num_edges] (directed; add reverse edges)
    edge_attr: Optional[np.ndarray] = None  # [num_edges, edge_feat_dim]

    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return int(self.x.shape[0])

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert the graph to PyTorch tensors.

        Returns:
            node_features: Tensor of shape (N, F).
            adjacency: Dense adjacency tensor of shape (N, N).
        """
        x = torch.as_tensor(self.x, dtype=torch.float32)
        n = x.shape[0]
        adj = torch.zeros((n, n), dtype=torch.float32)
        if self.edge_index.size > 0:
            i = torch.as_tensor(self.edge_index[0], dtype=torch.long)
            j = torch.as_tensor(self.edge_index[1], dtype=torch.long)
            adj[i, j] = 1.0
        return x, adj


class GraphDataset:
    def __init__(
        self,
        graphs: List[GraphData],
        labels: Optional[np.ndarray] = None,
        smiles: Optional[List[str]] = None,
    ):
        self.graphs = graphs
        self.labels = labels
        self.smiles = smiles

    def __len__(self) -> int:
        return len(self.graphs)

    def get_batch(
        self, indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
        node_features: List[torch.Tensor] = []
        adj_blocks: List[torch.Tensor] = []
        graph_ptr: List[int] = []
        offset = 0
        for idx in indices:
            x_i, adj_i = self.graphs[idx].to_tensors()
            node_features.append(x_i)
            adj_blocks.append(adj_i)
            offset += adj_i.shape[0]
            graph_ptr.append(offset)
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
        batch_ptr = torch.tensor(graph_ptr, dtype=torch.long)
        batch_labels = (
            torch.tensor(self.labels[indices], dtype=torch.float32)
            if self.labels is not None
            else None
        )
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
            return _fallback_graph_from_string(smiles)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # invalid SMILES -> fallback
                return _fallback_graph_from_string(smiles)

            # sanitization + explicit H (safer for 3D)
            Chem.SanitizeMol(mol)
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

            X = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, 4), dtype=np.float32)

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
                            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                            dtype=np.float32,
                        )
                        # append (x,y,z)
                        if coords.shape[0] == X.shape[0]:
                            X = np.concatenate([X, coords], axis=1)
                except Exception:
                    coords = None  # just proceed without 3D

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
                edges.append((i, j)); eattr.append(feat)
                edges.append((j, i)); eattr.append(feat)

            E = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
            EA = np.asarray(eattr, dtype=np.float32) if eattr else None

            # optional geometric edge features
            if add_3d and E.shape[1] > 0:
                try:
                    EA = _append_geom_edge_attr(mol, E, EA)
                except Exception:
                    # if augmentation fails, keep existing EA (may be None)
                    pass

            return GraphData(x=X, edge_index=E, edge_attr=EA)

        except Exception:
            # any unexpected RDKit error → robust fallback
            return _fallback_graph_from_string(smiles)




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
        for sm in smiles_list:
            try:
                g = cls.smiles_to_graph(
                    sm, add_3d=add_3d, random_seed=random_seed
                )
                
                graphs.append(g)
                smiles_out.append(sm)
                
            except Exception as e:
                
                if _RUNNING_IN_CI:
                    raise
                g = _fallback_graph_from_string(sm)
            
            graphs.append(g)
            smiles_out.append(sm)

        y = None if labels is None else np.asarray(labels)
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
    ) -> "GraphDataset":
        
        cache_path = None
        if cache_dir and n_rows is None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = os.path.basename(filepath).replace(".parquet", ".pkl")
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    graphs, labels = pickle.load(f)
                return cls(graphs, labels)

        df = pd.read_parquet(filepath)
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
                g = cls.smiles_to_graph(
                    sm, add_3d=add_3d, random_seed=random_seed
                )
                graphs.append(g)
                smiles_out.append(sm)
            except Exception:
                continue

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump((graphs, labels), f)

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
            cache_name = os.path.basename(filepath).replace(".csv", ".pkl")
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    graphs, labels = pickle.load(f)
                return cls(graphs, labels)

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
                g = cls.smiles_to_graph(
                    sm, add_3d=add_3d, random_seed=random_seed
                )
                graphs.append(g)
                smiles_out.append(sm)
            except Exception:
                continue

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump((graphs, labels), f)

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
            path = os.path.join(dirpath, fname)
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
                            else os.path.join(cache_dir, os.path.splitext(fname)[0])
                        )
                    ),
                    add_3d=add_3d,
                    random_seed=random_seed,
                    n_rows=n_rows_per_file,
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
                            else os.path.join(cache_dir, os.path.splitext(fname)[0])
                        )
                    ),
                    add_3d=add_3d,
                    random_seed=random_seed,
                    n_rows=n_rows_per_file,
                )
            else:
                raise ValueError(f"Unsupported ext: {ext}")

            graphs_all.extend(ds.graphs)
            smiles_all.extend(ds.smiles or [])
            if ds.labels is not None:
                labels_present = True
                labels_all.extend(ds.labels.tolist())

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

def _fallback_graph_from_string(s: str) -> "GraphData":
    """
    Deterministic tiny chain-graph from a string when RDKit isn't available.
    - N = max(2, min(10, len(s)))
    - Node feats: [position, position%3] as float32
    - Edges: linear chain i<->i+1 (both directions)
    """
    import numpy as _np

    n = max(2, min(10, len(s)))
    x = _np.stack(
        [_np.arange(n, dtype=_np.float32),
         (_np.arange(n) % 3).astype(_np.float32)],
        axis=1
    )  # [N,2]
    rows = _np.concatenate([_np.arange(n-1), _np.arange(1, n)], axis=0)
    cols = _np.concatenate([_np.arange(1, n), _np.arange(n-1)], axis=0)
    edge_index = _np.stack([rows.astype(_np.int64), cols.astype(_np.int64)], axis=0)
    
    return GraphData(x=x, edge_index=edge_index, edge_attr=None)