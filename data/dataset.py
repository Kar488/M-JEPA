from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional
import os
import pickle
import numpy as np

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception as e:
    raise ImportError("RDKit is required. Install via conda: conda install -c conda-forge rdkit") from e

import pandas as pd


@dataclass
class GraphData:
    x: np.ndarray               # [num_nodes, feat_dim]
    edge_index: np.ndarray      # [2, num_edges] (directed; add reverse edges)
    edge_attr: Optional[np.ndarray] = None  # [num_edges, edge_feat_dim]


class GraphDataset:
    def __init__(self, graphs: List[GraphData], labels: Optional[np.ndarray] = None):
        self.graphs = graphs
        self.labels = labels

    # ---------- Core featurisation ---------- #
    @staticmethod
    def smiles_to_graph(smiles: str, add_3d: bool = False, random_seed: Optional[int] = None) -> GraphData:
        """Convert SMILES -> RDKit Mol -> GraphData. Optionally append (x,y,z) to node features and compute edge features."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)

        # Node features (compact baseline; extend as you like)
        feats = []
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            deg = atom.GetDegree()
            aromatic = int(atom.GetIsAromatic())
            hybrid = int(atom.GetHybridization())
            feats.append([z, deg, aromatic, hybrid])
        X = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, 4), dtype=np.float32)

        # Optional 3D coords
        coords = None
        if add_3d and mol.GetNumAtoms() > 0:
            try:
                params = AllChem.ETKDGv3()
                if random_seed is not None:
                    params.randomSeed = int(random_seed)
                ok = AllChem.EmbedMolecule(mol, params)
                if ok == 0:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                    conf = mol.GetConformer()
                    coords = np.array(
                        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                        dtype=np.float32,
                    )
                    X = np.concatenate([X, coords], axis=1)
            except Exception:
                coords = None  # fall back

        # Edges + edge attributes (bond type one-hots, conjugation, ring, length if coords)
        edges = []
        eattr = []
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

            # add both directions
            edges.append((i, j)); eattr.append(feat)
            edges.append((j, i)); eattr.append(feat)

        E = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        EA = np.asarray(eattr, dtype=np.float32) if eattr else None

        return GraphData(x=X, edge_index=E, edge_attr=EA)

    # ---------- Builders ---------- #
    @classmethod
    def from_smiles_list(
        cls,
        smiles_list: List[str],
        labels: Optional[List[Any]] = None,
        add_3d_features: bool = False,
        random_seed: Optional[int] = None,
    ) -> "GraphDataset":
        graphs: List[GraphData] = []
        for sm in smiles_list:
            try:
                g = cls.smiles_to_graph(sm, add_3d=add_3d_features, random_seed=random_seed)
                graphs.append(g)
            except Exception:
                continue
        y = None if labels is None else np.asarray(labels)
        return cls(graphs, y)

    @classmethod
    def from_parquet(
        cls,
        filepath: str,
        smiles_col: str = "smiles",
        label_col: Optional[str] = None,
        cache_dir: Optional[str] = None,
        add_3d_features: bool = False,
        random_seed: Optional[int] = None,
        n_rows: Optional[int] = None,   # small subset helper
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
        labels = df[label_col].to_numpy() if (label_col and label_col in df.columns) else None

        graphs: List[GraphData] = []
        for sm in smiles:
            try:
                g = cls.smiles_to_graph(sm, add_3d=add_3d_features, random_seed=random_seed)
                graphs.append(g)
            except Exception:
                continue

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump((graphs, labels), f)

        return cls(graphs, labels)

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        smiles_col: str = "smiles",
        label_col: Optional[str] = None,
        sep: str = ",",
        cache_dir: Optional[str] = None,
        add_3d_features: bool = False,
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
        labels = df[label_col].to_numpy() if (label_col and label_col in df.columns) else None

        graphs: List[GraphData] = []
        for sm in smiles:
            try:
                g = cls.smiles_to_graph(sm, add_3d=add_3d_features, random_seed=random_seed)
                graphs.append(g)
            except Exception:
                continue

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump((graphs, labels), f)

        return cls(graphs, labels)

    @classmethod
    def from_directory(
        cls,
        dirpath: str,
        ext: str = "parquet",
        smiles_col: str = "smiles",
        label_col: Optional[str] = None,
        cache_dir: Optional[str] = None,
        add_3d_features: bool = False,
        random_seed: Optional[int] = None,
        prefix_filter: Optional[str] = None,
        n_rows_per_file: Optional[int] = None,  # subset helper
    ) -> "GraphDataset":
        graphs_all: List[GraphData] = []
        labels_all: List[Any] = []
        labels_present = False

        files = [f for f in os.listdir(dirpath) if f.lower().endswith(f".{ext.lower()}")]
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
                    cache_dir=None if n_rows_per_file else (None if cache_dir is None else os.path.join(cache_dir, os.path.splitext(fname)[0])),
                    add_3d_features=add_3d_features,
                    random_seed=random_seed,
                    n_rows=n_rows_per_file,
                )
            elif ext.lower() == "csv":
                ds = cls.from_csv(
                    filepath=path,
                    smiles_col=smiles_col,
                    label_col=label_col,
                    cache_dir=None if n_rows_per_file else (None if cache_dir is None else os.path.join(cache_dir, os.path.splitext(fname)[0])),
                    add_3d_features=add_3d_features,
                    random_seed=random_seed,
                    n_rows=n_rows_per_file,
                )
            else:
                raise ValueError(f"Unsupported ext: {ext}")

            graphs_all.extend(ds.graphs)
            if ds.labels is not None:
                labels_present = True
                labels_all.extend(ds.labels.tolist())

        labels = np.asarray(labels_all) if labels_present else None
        return cls(graphs_all, labels)
