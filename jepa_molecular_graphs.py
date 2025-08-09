"""
Joint‑Embedding Predictive Architectures (JEPA) and baseline models for
molecule representation learning.

This module relies on external packages: :mod:`matplotlib`, :mod:`pandas`,
and :mod:`scikit-learn` for plotting, data manipulation, and evaluation
metrics.

This module implements a simplified version of the JEPA algorithm adapted
for molecular graphs as described in the provided project specification.
The goal of JEPA is to learn semantically meaningful graph embeddings
without explicitly reconstructing raw molecular inputs. A separate baseline
contrastive model is also provided for comparison. After pretraining on
unlabelled data the models can be evaluated on downstream prediction tasks
using simple linear heads.

The code is intentionally modular and heavily documented in plain
English so that readers without a strong background can follow along.
Several test cases at the end of the file demonstrate the expected usage
patterns and produce clean tables and plots summarising the results.

Key components:
    * Graph representation – converts SMILES strings into adjacency
      matrices and node features.
    * Data loaders – provide batches of graphs for training.
    * A simple Graph Neural Network (GNN) encoder – learns node and
      graph level embeddings through message passing.
    * A momentum (EMA) copy of the encoder – updated using an exponential
      moving average of the online encoder’s weights.
    * A two‑layer MLP predictor – predicts target embeddings from context
      embeddings.
    * Training loops for JEPA and a contrastive baseline – update models
      using stochastic gradient descent on unsupervised objectives.
    * Downstream evaluation utilities – freeze encoders and train
      lightweight heads on labelled data for classification or regression.
    * Ablation experiments – explore the effect of masking strategy,
      architecture choices and hyper‑parameters.
    * A practical case study – ranks molecules by predicted toxicity
      (illustrative only since real toxicity labels are unavailable here).

Limitations: The full MoleculeNet datasets and baseline models such as
MolCLR, HiMol and GeomGCL are not included due to resource constraints and
licensing restrictions. Instead, we supply toy datasets and simplified
baselines suitable for demonstration and testing. The architecture is kept
small to ensure the provided test cases execute quickly on modest
hardware.

Author: OpenAI ChatGPT
Date: August 7, 2025
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
except ImportError as e:
    raise ImportError(
        "RDKit is required for SMILES parsing. Please install rdkit-pypi "
        "or ensure RDKit is available in your environment."
    ) from e

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.predictor import MLPPredictor
from utils.pooling import global_mean_pool
from utils.seed import set_seed


@dataclass
class GraphData:
    """Container for a single molecular graph.

    Attributes:
        adj (np.ndarray): A binary adjacency matrix of shape (N, N)
            where N is the number of atoms. adj[i, j] = 1 if there is an
            undirected bond between atoms i and j, else 0. Self‑loops are not
            included.
        x (np.ndarray): Node feature matrix of shape (N, F). Each row
            contains the features for a single atom. In this simplified
            implementation we use the atomic number and atomic degree as
            features. Additional features (e.g. hybridisation, aromaticity)
            could be appended here.
        smiles (str): The original SMILES string for reference and
            potential debugging.
    """

    adj: np.ndarray
    x: np.ndarray
    smiles: str

    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return self.x.shape[0]

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors.

        Returns:
            node_features: Tensor of shape (N, F) with dtype=torch.float32.
            adjacency: Tensor of shape (N, N) with dtype=torch.float32.
        """
        node_features = torch.from_numpy(self.x).float()
        adjacency = torch.from_numpy(self.adj).float()
        return node_features, adjacency


def smiles_to_graph(smiles: str) -> GraphData:
    """Convert a SMILES string into a GraphData object.

    Each molecule is parsed using RDKit. The adjacency matrix is obtained
    from RDKit’s built‑in routines. Node features are limited to two
    simple properties for demonstration: atomic number and degree. In a
    real application additional properties could be included such as
    formal charge, hybridisation state or aromaticity. Hydrogens are
    implicitly represented via RDKit and thus are not explicitly included
    as nodes.

    Args:
        smiles: A string representing the molecule using SMILES notation.

    Returns:
        A GraphData instance representing the molecular graph.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES string: {smiles}")
    N = mol.GetNumAtoms()
    # Obtain adjacency matrix (without self loops)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    # Extract two simple node features: atomic number and atomic degree
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    degrees = [int(atom.GetDegree()) for atom in mol.GetAtoms()]
    # Stack features into shape (N, 2)
    x = np.stack([atomic_numbers, degrees], axis=1).astype(np.float32)
    return GraphData(adj=adj.astype(np.float32), x=x, smiles=smiles)


class GraphDataset:
    """A dataset of molecular graphs for unsupervised pretraining or supervised tasks.

    The dataset can be initialised from a list of SMILES strings or from
    preconstructed GraphData objects. Batching support is provided for
    unsupervised pretraining: multiple graphs in a batch are combined by
    stacking their adjacency matrices and node feature matrices into a
    block‑diagonal representation. For supervised tasks we maintain a
    separate label array.
    """

    def __init__(
        self, graphs: List[GraphData], labels: Optional[np.ndarray] = None
    ) -> None:
        self.graphs = graphs
        self.labels = labels  # shape (num_graphs,) for classification/regression

    @classmethod
    def from_smiles_list(
        cls, smiles_list: List[str], labels: Optional[List[Any]] = None
    ) -> "GraphDataset":
        """Construct dataset from SMILES strings.

        Args:
            smiles_list: List of SMILES strings.
            labels: Optional list of labels corresponding to each SMILES. If
                provided, it must have the same length as smiles_list.

        Returns:
            An instance of GraphDataset containing GraphData objects for each SMILES.
        """
        graphs = [smiles_to_graph(s) for s in smiles_list]
        labels_array = None
        if labels is not None:
            labels_array = np.array(labels)
        return cls(graphs, labels_array)

    def __len__(self) -> int:
        return len(self.graphs)

    def get_batch(
        self, indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Create a batch of graphs by block‑diagonally stacking adjacency matrices.

        For unsupervised pretraining we simply need the combined node
        features and adjacency matrix. For supervised tasks we also
        return the batch labels.

        Args:
            indices: List of graph indices to include in the batch.

        Returns:
            batch_x: Tensor of shape (sum_i N_i, F), concatenation of node features.
            batch_adj: Tensor of shape (sum_i N_i, sum_i N_i), block‑diagonal adjacency matrix.
            batch_graph_ptr: Tensor marking the boundaries of graphs within the batch.
            batch_labels: Tensor of shape (len(indices),) with labels if available, otherwise None.
        """
        node_features = []
        adj_blocks = []
        graph_ptr = []
        node_offset = 0
        for idx in indices:
            g = self.graphs[idx]
            x_i, adj_i = g.to_tensors()
            node_features.append(x_i)
            N_i = adj_i.shape[0]
            # For adjacency, we build a block diagonal matrix by expanding existing block
            if adj_blocks:
                existing = torch.block_diag(*adj_blocks)
                new_block = torch.zeros(
                    (existing.shape[0] + N_i, existing.shape[0] + N_i)
                )
                new_block[: existing.shape[0], : existing.shape[0]] = existing
                new_block[existing.shape[0] :, existing.shape[0] :] = adj_i
                adj_blocks = [new_block]
            else:
                adj_blocks = [adj_i]
            node_offset += N_i
            graph_ptr.append(node_offset)
        batch_x = torch.cat(node_features, dim=0)
        batch_adj = adj_blocks[0] if adj_blocks else torch.tensor([])
        batch_graph_ptr = torch.tensor(graph_ptr, dtype=torch.long)
        if self.labels is not None:
            batch_labels = torch.tensor(self.labels[indices], dtype=torch.float32)
        else:
            batch_labels = None
        return batch_x, batch_adj, batch_graph_ptr, batch_labels


class GNNEncoder(nn.Module):
    """A simple Graph Neural Network (GNN) encoder for molecular graphs.

    This encoder performs message passing over the adjacency matrix to
    propagate information between atoms. It consists of a configurable
    number of layers. At each layer we compute a linear transformation of
    node features and aggregate messages from neighbours. A global mean
    pooling is applied at the end to obtain a fixed‑size graph
    representation. This implementation is intentionally simple for
    pedagogical clarity; more sophisticated architectures (e.g. GIN, GAT)
    can be substituted with minor modifications.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3
    ) -> None:
        """Initialise the GNN encoder.

        Args:
            input_dim: Dimensionality of input node features.
            hidden_dim: Dimensionality of hidden layers and output embedding.
            num_layers: Number of message passing layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # Linear layers
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.message_lin = nn.Linear(hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings for a batch of concatenated graphs.

        Args:
            x: Tensor of shape (N_total, F_in) representing node features.
            adj: Tensor of shape (N_total, N_total) representing the block‑diagonal adjacency matrix.

        Returns:
            h: Tensor of shape (N_total, hidden_dim) containing node embeddings.
        """
        h = F.relu(self.input_lin(x))
        for _ in range(self.num_layers):
            m = torch.matmul(adj, h)
            m = self.message_lin(m)
            h = F.relu(self.update_lin(h) + m)
        return h


class EMA:
    """Maintain an Exponential Moving Average (EMA) copy of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.99) -> None:
        self.decay = decay
        self.params = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.params, model.parameters()):
            ema_p.mul_(self.decay)
            ema_p.add_(p.detach() * (1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for p, ema_p in zip(model.parameters(), self.params):
            p.data.copy_(ema_p)


def sample_subgraphs(
    graph: GraphData, mask_ratio: float = 0.15, contiguous: bool = False
) -> Tuple[GraphData, GraphData]:
    """Sample context and target subgraphs from a graph."""
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


def regularisation_loss(model: nn.Module, lam: float = 1e-4) -> torch.Tensor:
    reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        reg_loss += torch.sum(p**2)
    return lam * reg_loss


def train_jepa(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    ema_encoder: GNNEncoder,
    predictor: MLPPredictor,
    ema: EMA,
    epochs: int = 10,
    batch_size: int = 32,
    mask_ratio: float = 0.15,
    contiguous: bool = False,
    lr: float = 1e-4,
    device: str = "cpu",
    reg_lambda: float = 1e-4,
) -> List[float]:
    encoder = encoder.to(device)
    ema_encoder = ema_encoder.to(device)
    predictor = predictor.to(device)
    optimiser = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )
    epoch_losses: List[float] = []
    all_indices = list(range(len(dataset)))
    for epoch in range(epochs):
        random.shuffle(all_indices)
        batch_losses = []
        for start in range(0, len(all_indices), batch_size):
            batch_indices = all_indices[start : start + batch_size]
            contexts: List[GraphData] = []
            targets: List[GraphData] = []
            for idx in batch_indices:
                g = dataset.graphs[idx]
                context_g, target_g = sample_subgraphs(
                    g, mask_ratio=mask_ratio, contiguous=contiguous
                )
                contexts.append(context_g)
                targets.append(target_g)
            context_dataset = GraphDataset(contexts)
            target_dataset = GraphDataset(targets)
            ctx_x, ctx_adj, ctx_ptr, _ = context_dataset.get_batch(
                list(range(len(contexts)))
            )
            tgt_x, tgt_adj, tgt_ptr, _ = target_dataset.get_batch(
                list(range(len(targets)))
            )
            ctx_x = ctx_x.to(device)
            ctx_adj = ctx_adj.to(device)
            tgt_x = tgt_x.to(device)
            tgt_adj = tgt_adj.to(device)
            ctx_node_emb = encoder(ctx_x, ctx_adj)
            ctx_graph_emb = global_mean_pool(ctx_node_emb, ctx_ptr.to(device))
            # Copy EMA parameters into ema_encoder
            ema.copy_to(ema_encoder)
            with torch.no_grad():
                tgt_node_emb = ema_encoder(tgt_x, tgt_adj)
                tgt_graph_emb = global_mean_pool(tgt_node_emb, tgt_ptr.to(device))
            pred_tgt_emb = predictor(ctx_graph_emb)
            mse_loss = F.mse_loss(pred_tgt_emb, tgt_graph_emb.detach())
            reg_loss = regularisation_loss(predictor, lam=reg_lambda)
            loss = mse_loss + reg_loss
            batch_losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            ema.update(encoder)
        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(mean_loss)
        print(f"Epoch {epoch+1}/{epochs} – JEPA training loss: {mean_loss:.4f}")
    return epoch_losses


def train_contrastive(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    projection_dim: int = 128,
    epochs: int = 10,
    batch_size: int = 32,
    mask_ratio: float = 0.15,
    lr: float = 1e-4,
    device: str = "cpu",
    temperature: float = 0.1,
) -> List[float]:
    encoder = encoder.to(device)
    proj_head = nn.Sequential(
        nn.Linear(encoder.hidden_dim, projection_dim),
        nn.ReLU(),
        nn.Linear(projection_dim, projection_dim),
    ).to(device)
    optimiser = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()), lr=lr
    )
    epoch_losses = []
    all_indices = list(range(len(dataset)))
    for epoch in range(epochs):
        random.shuffle(all_indices)
        batch_losses = []
        for start in range(0, len(all_indices), batch_size):
            batch_indices = all_indices[start : start + batch_size]
            view1_graphs: List[GraphData] = []
            view2_graphs: List[GraphData] = []
            for idx in batch_indices:
                g = dataset.graphs[idx]
                ctx1, _ = sample_subgraphs(g, mask_ratio=mask_ratio, contiguous=False)
                if random.random() < 0.5:
                    ctx2, _ = sample_subgraphs(
                        g, mask_ratio=mask_ratio, contiguous=True
                    )
                else:
                    ctx2, _ = sample_subgraphs(
                        g, mask_ratio=mask_ratio, contiguous=False
                    )
                view1_graphs.append(ctx1)
                view2_graphs.append(ctx2)
            view1_dataset = GraphDataset(view1_graphs)
            v1_x, v1_adj, v1_ptr, _ = view1_dataset.get_batch(
                list(range(len(view1_graphs)))
            )
            view2_dataset = GraphDataset(view2_graphs)
            v2_x, v2_adj, v2_ptr, _ = view2_dataset.get_batch(
                list(range(len(view2_graphs)))
            )
            v1_x, v1_adj, v2_x, v2_adj = (
                v1_x.to(device),
                v1_adj.to(device),
                v2_x.to(device),
                v2_adj.to(device),
            )
            h1_node = encoder(v1_x, v1_adj)
            h2_node = encoder(v2_x, v2_adj)
            h1_graph = global_mean_pool(h1_node, v1_ptr.to(device))
            h2_graph = global_mean_pool(h2_node, v2_ptr.to(device))
            z1 = proj_head(h1_graph)
            z2 = proj_head(h2_graph)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            sim_matrix = torch.mm(z1, torch.cat([z2, z1], dim=0).t()) / temperature
            labels = torch.arange(0, len(batch_indices), device=device)
            mask = torch.eye(len(batch_indices), device=device)
            logits = torch.cat(
                [
                    sim_matrix[:, : len(batch_indices)],
                    sim_matrix[:, len(batch_indices) :] * (1.0 - mask),
                ],
                dim=1,
            )
            targets = labels
            loss = F.cross_entropy(logits, targets)
            batch_losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(mean_loss)
        print(f"Epoch {epoch+1}/{epochs} – Contrastive loss: {mean_loss:.4f}")
    return epoch_losses


def train_linear_head(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    task_type: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict[str, float]:
    assert (
        dataset.labels is not None
    ), "Dataset must have labels for supervised training."
    assert task_type in {"classification", "regression"}, "Invalid task type."
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    num_graphs = len(dataset)
    indices = list(range(num_graphs))
    random.shuffle(indices)
    train_end = int(0.8 * num_graphs)
    val_end = int(0.9 * num_graphs)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    head = nn.Linear(encoder.hidden_dim, 1).to(device)
    if task_type == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    def evaluate(split_idx: List[int]) -> Tuple[float, Dict[str, float]]:
        batch_losses = []
        all_targets = []
        all_preds = []
        for start in range(0, len(split_idx), batch_size):
            batch_indices = split_idx[start : start + batch_size]
            batch_x, batch_adj, batch_ptr, batch_labels = dataset.get_batch(
                batch_indices
            )
            batch_x = batch_x.to(device)
            batch_adj = batch_adj.to(device)
            node_emb = encoder(batch_x, batch_adj)
            graph_emb = global_mean_pool(node_emb, batch_ptr.to(device))
            preds = head(graph_emb).squeeze(1)
            targets = torch.tensor(
                dataset.labels[batch_indices], dtype=torch.float32, device=device
            )
            loss = loss_fn(preds, targets)
            batch_losses.append(loss.item())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        y_true = np.concatenate(all_targets) if all_targets else np.array([])
        y_pred = np.concatenate(all_preds) if all_preds else np.array([])
        metrics: Dict[str, float] = {}
        if task_type == "classification" and y_true.size > 0:
            probs = 1 / (1 + np.exp(-y_pred))
            try:
                roc_auc = roc_auc_score(y_true, probs)
            except ValueError:
                roc_auc = float("nan")
            try:
                pr_auc = average_precision_score(y_true, probs)
            except ValueError:
                pr_auc = float("nan")
            metrics.update({"roc_auc": roc_auc, "pr_auc": pr_auc})
        elif task_type == "regression" and y_true.size > 0:
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            metrics.update({"rmse": rmse, "mae": mae})
        return mean_loss, metrics

    for epoch in range(epochs):
        encoder.eval()
        head.train()
        train_losses = []
        for start in range(0, len(train_idx), batch_size):
            batch_indices = train_idx[start : start + batch_size]
            batch_x, batch_adj, batch_ptr, _ = dataset.get_batch(batch_indices)
            batch_x = batch_x.to(device)
            batch_adj = batch_adj.to(device)
            node_emb = encoder(batch_x, batch_adj)
            graph_emb = global_mean_pool(node_emb, batch_ptr.to(device))
            preds = head(graph_emb).squeeze()
            targets = torch.tensor(
                dataset.labels[batch_indices], dtype=torch.float32, device=device
            )
            loss = loss_fn(preds, targets)
            train_losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        encoder.eval()
        head.eval()
        val_loss, val_metrics = evaluate(val_idx)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        print(
            f"Epoch {epoch+1}/{epochs} – Linear head train loss: {mean_train_loss:.4f}, val loss: {val_loss:.4f}"
        )
        if epoch >= 10 and val_loss > best_val_loss:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    _, test_metrics = evaluate(test_idx)
    return test_metrics


def run_ablation_experiments() -> pd.DataFrame:
    smiles_list = [
        "CCO",
        "CCN",
        "CCC",
        "c1ccccc1",
        "CC(=O)O",
        "CCOCC",
        "CNC",
        "CCCl",
        "COC",
        "CCN(CC)CC",
    ]
    classification_labels = np.random.randint(0, 2, size=len(smiles_list))
    regression_labels = np.random.rand(len(smiles_list)) * 10.0
    dataset_class = GraphDataset.from_smiles_list(
        smiles_list, labels=classification_labels.tolist()
    )
    dataset_reg = GraphDataset.from_smiles_list(
        smiles_list, labels=regression_labels.tolist()
    )
    configs = []
    results = []
    mask_ratios = [0.1, 0.15, 0.25]
    contiguities = [False, True]
    hidden_dims = [128, 256]
    layers_list = [2, 3]
    ema_decays = [0.95, 0.99]
    for mask_ratio in mask_ratios:
        for contiguous in contiguities:
            for hidden_dim in hidden_dims:
                for num_layers in layers_list:
                    for ema_decay in ema_decays:
                        encoder = GNNEncoder(
                            input_dim=2, hidden_dim=hidden_dim, num_layers=num_layers
                        )
                        ema_encoder = GNNEncoder(
                            input_dim=2, hidden_dim=hidden_dim, num_layers=num_layers
                        )
                        ema_helper = EMA(encoder, decay=ema_decay)
                        predictor = MLPPredictor(
                            embed_dim=hidden_dim, hidden_dim=hidden_dim * 2
                        )
                        print(
                            f"\nRunning JEPA pretraining for configuration: mask_ratio={mask_ratio}, contiguous={contiguous}, hidden_dim={hidden_dim}, layers={num_layers}, ema_decay={ema_decay}"
                        )
                        train_jepa(
                            dataset=dataset_class,
                            encoder=encoder,
                            ema_encoder=ema_encoder,
                            predictor=predictor,
                            ema=ema_helper,
                            epochs=2,
                            batch_size=4,
                            mask_ratio=mask_ratio,
                            contiguous=contiguous,
                            lr=5e-4,
                            device="cpu",
                            reg_lambda=1e-4,
                        )
                        class_metrics = train_linear_head(
                            dataset_class,
                            encoder,
                            task_type="classification",
                            epochs=5,
                            lr=5e-3,
                            batch_size=4,
                        )
                        reg_metrics = train_linear_head(
                            dataset_reg,
                            encoder,
                            task_type="regression",
                            epochs=5,
                            lr=5e-3,
                            batch_size=4,
                        )
                        config = {
                            "mask_ratio": mask_ratio,
                            "contiguous": contiguous,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "ema_decay": ema_decay,
                        }
                        result = {
                            "roc_auc": class_metrics.get("roc_auc", float("nan")),
                            "pr_auc": class_metrics.get("pr_auc", float("nan")),
                            "rmse": reg_metrics.get("rmse", float("nan")),
                            "mae": reg_metrics.get("mae", float("nan")),
                        }
                        configs.append(config)
                        results.append(result)
    df = pd.concat(
        [
            pd.DataFrame(configs),
            pd.DataFrame(results),
        ],
        axis=1,
    )
    return df


def case_study_tox21() -> pd.DataFrame:
    smiles_list = [
        "CCO",
        "CCN",
        "CCC",
        "c1ccccc1",
        "CC(=O)O",
        "CCOCC",
        "CNC",
        "CCCl",
        "COC",
        "CCN(CC)CC",
        "CCO",
        "CCN",
        "CCC",
        "c1ccccc1",
        "CC(=O)O",
        "CCOCC",
        "CNC",
        "CCCl",
        "COC",
        "CCN(CC)CC",
        "CCO",
        "CCN",
        "CCC",
        "c1ccccc1",
        "CC(=O)O",
        "CCOCC",
        "CNC",
        "CCCl",
        "COC",
        "CCN(CC)CC",
    ]
    true_toxicity = np.random.rand(len(smiles_list))
    dataset = GraphDataset.from_smiles_list(smiles_list, labels=true_toxicity.tolist())
    encoder = GNNEncoder(input_dim=2, hidden_dim=64, num_layers=2)
    ema_encoder = GNNEncoder(input_dim=2, hidden_dim=64, num_layers=2)
    ema_helper = EMA(encoder, decay=0.98)
    predictor = MLPPredictor(embed_dim=64, hidden_dim=128)
    train_jepa(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema_helper,
        epochs=3,
        batch_size=10,
        mask_ratio=0.15,
        contiguous=False,
        lr=1e-3,
        device="cpu",
        reg_lambda=1e-4,
    )
    test_metrics = train_linear_head(
        dataset, encoder, task_type="regression", epochs=10, lr=1e-2, batch_size=10
    )
    print("Case study regression metrics:", test_metrics)
    for p in encoder.parameters():
        p.requires_grad = False
    head = nn.Linear(encoder.hidden_dim, 1)
    optimiser = torch.optim.Adam(head.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    for epoch in range(20):
        batch_x, batch_adj, batch_ptr, _ = dataset.get_batch(list(range(len(dataset))))
        batch_x = batch_x.to("cpu")
        batch_adj = batch_adj.to("cpu")
        node_emb = encoder(batch_x, batch_adj)
        graph_emb = global_mean_pool(node_emb, batch_ptr)
        preds = head(graph_emb).squeeze(1)
        targets = torch.tensor(dataset.labels, dtype=torch.float32)
        loss = loss_fn(preds, targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    batch_x, batch_adj, batch_ptr, _ = dataset.get_batch(list(range(len(dataset))))
    node_emb = encoder(batch_x, batch_adj)
    graph_emb = global_mean_pool(node_emb, batch_ptr)
    preds = head(graph_emb).squeeze(1).detach().numpy()
    ranked_indices = np.argsort(-preds)
    median_toxicity = np.median(true_toxicity)
    benign_mask = true_toxicity < median_toxicity
    results = []
    for n_remove in [1, 3, 5, 10]:
        removed = ranked_indices[:n_remove]
        remaining_mask = np.ones(len(smiles_list), dtype=bool)
        remaining_mask[removed] = False
        remaining_benign = benign_mask[remaining_mask].sum()
        total_benign = benign_mask.sum()
        benign_fraction = (
            remaining_benign / total_benign if total_benign > 0 else float("nan")
        )
        results.append(
            {
                "top_removed": n_remove,
                "remaining_benign_fraction": benign_fraction,
            }
        )
    df = pd.DataFrame(results)
    return df


def plot_training_curves(
    curves: Dict[str, List[float]], title: str = "Training Loss Curves"
) -> None:
    plt.figure(figsize=(8, 4))
    for name, losses in curves.items():
        plt.plot(range(1, len(losses) + 1), losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    example_smiles = [
        "CCO",
        "CCN",
        "CCC",
        "c1ccccc1",
        "CC(=O)O",
        "CCOCC",
        "CNC",
        "CCCl",
        "COC",
        "CCN(CC)CC",
    ]
    unlabeled_dataset = GraphDataset.from_smiles_list(example_smiles)
    encoder = GNNEncoder(input_dim=2, hidden_dim=64, num_layers=2)
    ema_encoder = GNNEncoder(input_dim=2, hidden_dim=64, num_layers=2)
    ema_helper = EMA(encoder, decay=0.99)
    predictor = MLPPredictor(embed_dim=64, hidden_dim=128)
    print("\n--- Training JEPA on toy dataset ---")
    jepa_losses = train_jepa(
        dataset=unlabeled_dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema_helper,
        epochs=3,
        batch_size=5,
        mask_ratio=0.2,
        contiguous=False,
        lr=1e-3,
        device="cpu",
        reg_lambda=1e-4,
    )
    print("\n--- Training contrastive baseline on toy dataset ---")
    contrastive_encoder = GNNEncoder(input_dim=2, hidden_dim=64, num_layers=2)
    contrastive_losses = train_contrastive(
        dataset=unlabeled_dataset,
        encoder=contrastive_encoder,
        projection_dim=32,
        epochs=3,
        batch_size=5,
        mask_ratio=0.2,
        lr=1e-3,
        device="cpu",
        temperature=0.1,
    )
    plot_training_curves(
        {"JEPA": jepa_losses, "Contrastive": contrastive_losses},
        title="Toy Unsupervised Training Losses",
    )
    print("\n--- Running ablation experiments ---")
    ablation_df = run_ablation_experiments()
    print(ablation_df)
    print("\n--- Case study: toxicity ranking ---")
    case_df = case_study_tox21()
    print(case_df)
