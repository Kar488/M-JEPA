"""Dataset utilities for molecular graph representations.

This module defines classes and helper functions to convert SMILES
strings into graph objects with adjacency matrices and node features.
It also provides a batching mechanism that stacks multiple graphs into
a block‑diagonal adjacency matrix for efficient processing by GNNs.
Additionally, it includes a function to sample context and target
subgraphs for self‑supervised pretraining.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
except ImportError as e:
    raise ImportError(
        "RDKit is required for SMILES parsing. Please install rdkit-pypi or use conda install rdkit."
    ) from e


@dataclass
class GraphData:
    """Representation of a single molecular graph.

    Attributes:
        adj: A binary adjacency matrix of shape (N, N) where N is the number
            of atoms. A value of 1 indicates a bond between atoms i and j.
        x: Node feature matrix of shape (N, F) containing per‑atom features.
        smiles: The original SMILES string for reference.
    """
    adj: np.ndarray
    x: np.ndarray
    smiles: str

    def num_nodes(self) -> int:
        """Return the number of nodes (atoms) in the graph."""
        return self.x.shape[0]

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert the numpy arrays to float32 PyTorch tensors."""
        node_features = torch.from_numpy(self.x).float()
        adjacency = torch.from_numpy(self.adj).float()
        return node_features, adjacency


def smiles_to_graph(smiles: str) -> GraphData:
    """Parse a SMILES string into a rich GraphData object.

    This function extracts a variety of atomic features beyond the
    simplest ones. The node feature vector for each atom includes:
      0. Atomic number (integer)
      1. Atom degree (number of neighbours)
      2. Total valence (sum of formal valence and implicit valence)
      3. Hybridisation (encoded as an integer code)
      4. Aromaticity (1 if aromatic else 0)
      5. Ring membership (1 if atom is in a ring else 0)
      6. Formal charge

    The adjacency matrix is weighted by bond order: single bonds are
    weighted 1.0, double bonds 2.0, triple bonds 3.0 and aromatic bonds
    1.5. This weighting enables the GNN to distinguish bond types
    without explicit edge features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES string: {smiles}")
    N = mol.GetNumAtoms()
    # Construct weighted adjacency matrix
    adj = np.zeros((N, N), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        btype = bond.GetBondType()
        # Map bond type to numeric weight
        if btype == Chem.BondType.SINGLE:
            w = 1.0
        elif btype == Chem.BondType.DOUBLE:
            w = 2.0
        elif btype == Chem.BondType.TRIPLE:
            w = 3.0
        elif btype == Chem.BondType.AROMATIC:
            w = 1.5
        else:
            w = 1.0
        adj[i, j] = w
        adj[j, i] = w
    # Extract atomic features
    features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        degree = int(atom.GetDegree())
        valence = atom.GetTotalValence()  # includes implicit valence
        # Encode hybridisation as integer: use enumeration order
        hyb = atom.GetHybridization()
        hyb_mapping = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 1,
            Chem.rdchem.HybridizationType.SP3: 2,
            Chem.rdchem.HybridizationType.SP3D: 3,
            Chem.rdchem.HybridizationType.SP3D2: 4,
        }
        hybrid_code = hyb_mapping.get(hyb, -1)
        aromatic = 1 if atom.GetIsAromatic() else 0
        in_ring = 1 if atom.IsInRing() else 0
        formal_charge = atom.GetFormalCharge()
        features.append([
            atomic_num,
            degree,
            valence,
            hybrid_code,
            aromatic,
            in_ring,
            formal_charge,
        ])
    x = np.array(features, dtype=np.float32)
    return GraphData(adj=adj, x=x, smiles=smiles)


class GraphDataset:
    """Dataset wrapper for lists of molecular graphs.

    This class stores a list of GraphData objects and optional labels. It
    provides a batching function that concatenates node features and
    constructs a block‑diagonal adjacency matrix for mini‑batch training.
    """

    def __init__(self, graphs: List[GraphData], labels: Optional[np.ndarray] = None) -> None:
        self.graphs = graphs
        self.labels = labels

    @classmethod
    def from_smiles_list(cls, smiles_list: List[str], labels: Optional[List[Any]] = None) -> "GraphDataset":
        """Create a GraphDataset from a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.
            labels: Optional labels for each graph.

        Returns:
            A GraphDataset instance with GraphData objects.
        """
        graphs = [smiles_to_graph(s) for s in smiles_list]
        labels_array = None
        if labels is not None:
            labels_array = np.array(labels)
        return cls(graphs, labels_array)

    @classmethod
    def from_parquet(
        cls,
        filepath: str,
        smiles_col: str = "smiles",
        label_col: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> "GraphDataset":
        """Load a dataset from a Parquet file containing SMILES strings.

        This helper reads a Parquet file (such as those provided on
        HuggingFace) and constructs a GraphDataset. The file must
        contain a column with SMILES strings. Optionally, a label
        column can be specified for supervised tasks. To speed up
        repeated loading, processed graphs can be cached to disk.

        Args:
            filepath: Path to the Parquet file.
            smiles_col: Name of the column containing SMILES strings.
            label_col: Name of the column containing labels (for
                classification/regression). If None, the dataset will be
                treated as unlabeled.
            cache_dir: Directory in which to store cached GraphData
                objects. If provided, the loader will attempt to load
                pre‑processed graphs from this directory and will save
                newly processed graphs for future use.

        Returns:
            A GraphDataset instance.
        """
        import os
        import pickle
        import pandas as pd

        # Read the Parquet file using pandas
        df = pd.read_parquet(filepath)
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in {filepath}")
        smiles_list = df[smiles_col].astype(str).tolist()
        labels = None
        if label_col is not None:
            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found in {filepath}")
            labels = df[label_col].tolist()
        graphs: List[GraphData] = []
        # Create cache directory if caching is requested
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        for idx, sm in enumerate(smiles_list):
            cache_path = None
            if cache_dir is not None:
                # Use a simple hash of the SMILES string as filename
                import hashlib

                sm_hash = hashlib.md5(sm.encode("utf-8")).hexdigest()
                cache_path = os.path.join(cache_dir, f"{sm_hash}.pkl")
                if os.path.exists(cache_path):
                    # Load cached GraphData
                    with open(cache_path, "rb") as f:
                        g = pickle.load(f)
                    graphs.append(g)
                    continue
            # Parse SMILES to GraphData
            try:
                g = smiles_to_graph(sm)
            except Exception as e:
                # Skip malformed SMILES and warn
                print(f"Warning: skipping SMILES '{sm}' at index {idx} due to error: {e}")
                continue
            graphs.append(g)
            # Save to cache
            if cache_path is not None:
                with open(cache_path, "wb") as f:
                    pickle.dump(g, f)
        if labels is not None:
            # Filter labels to match successfully parsed graphs
            # Filter out indices where the graph could not be constructed
            if len(graphs) != len(smiles_list):
                # Identify indices of successfully processed SMILES
                processed_smiles_set = {g.smiles for g in graphs}
                filtered_labels = [lab for sm, lab in zip(smiles_list, labels) if sm in processed_smiles_set]
                labels_array = np.array(filtered_labels)
            else:
                labels_array = np.array(labels)
        else:
            labels_array = None
        return cls(graphs, labels_array)

    def __len__(self) -> int:
        return len(self.graphs)

    def get_batch(
        self, indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Construct a batch of graphs by stacking node features and adjacency matrices.

        Args:
            indices: Indices of graphs to include in the batch.

        Returns:
            batch_x: Concatenated node features (sum_i N_i, F).
            batch_adj: Block‑diagonal adjacency matrix (sum_i N_i, sum_i N_i).
            batch_ptr: Boundaries for each graph in the batch (len(indices),).
            batch_labels: Labels for graphs if available, else None.
        """
        node_features: List[torch.Tensor] = []
        adj_blocks: List[torch.Tensor] = []
        graph_ptr: List[int] = []
        node_offset = 0
        for idx in indices:
            g = self.graphs[idx]
            x_i, adj_i = g.to_tensors()
            node_features.append(x_i)
            if adj_blocks:
                existing = torch.block_diag(*adj_blocks)
                N_i = adj_i.shape[0]
                new_block = torch.zeros((existing.shape[0] + N_i, existing.shape[0] + N_i))
                new_block[: existing.shape[0], : existing.shape[0]] = existing
                new_block[existing.shape[0] :, existing.shape[0] :] = adj_i
                adj_blocks = [new_block]
            else:
                adj_blocks = [adj_i]
            node_offset += g.num_nodes()
            graph_ptr.append(node_offset)
        batch_x = torch.cat(node_features, dim=0)
        batch_adj = adj_blocks[0] if adj_blocks else torch.tensor([])
        batch_ptr = torch.tensor(graph_ptr, dtype=torch.long)
        if self.labels is not None:
            batch_labels = torch.tensor(self.labels[indices], dtype=torch.float32)
        else:
            batch_labels = None
        return batch_x, batch_adj, batch_ptr, batch_labels


def sample_subgraphs(
    graph: GraphData, mask_ratio: float = 0.15, contiguous: bool = False
) -> Tuple[GraphData, GraphData]:
    """Sample context and target subgraphs from a graph by masking nodes.

    A fraction of nodes specified by `mask_ratio` is removed to form the
    target subgraph; the remaining nodes form the context subgraph. If
    `contiguous` is True, nodes are removed in a contiguous block.

    Returns:
        (context_graph, target_graph): Two GraphData objects representing
        the context and target subgraphs.
    """
    N = graph.num_nodes()
    if N == 0:
        raise ValueError("Graph must contain at least one node.")
    num_mask = max(1, int(round(mask_ratio * N)))
    if contiguous:
        start = random.randint(0, max(0, N - num_mask))
        mask_indices = list(range(start, start + num_mask))
    else:
        mask_indices = random.sample(range(N), num_mask)
    mask_set = set(mask_indices)
    context_indices = [i for i in range(N) if i not in mask_set]
    target_indices = mask_indices.copy()
    if not context_indices:
        context_indices.append(target_indices.pop(0))
    if not target_indices:
        target_indices.append(context_indices.pop(0))
    def build_subgraph(indices: List[int]) -> GraphData:
        sub_adj = graph.adj[np.ix_(indices, indices)]
        sub_x = graph.x[indices]
        return GraphData(adj=sub_adj, x=sub_x, smiles=graph.smiles)
    context_graph = build_subgraph(context_indices)
    target_graph = build_subgraph(target_indices)
    return context_graph, target_graph
