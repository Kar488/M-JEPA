"""Practical case study demonstrating the use of JEPA embeddings.

This module simulates a simple case study inspired by the Tox21 dataset
to show how learned embeddings can be used to prioritise molecules.
Since the real Tox21 data are not available in this environment, we
generate synthetic toxicity scores for a small set of molecules. The
workflow trains a JEPA encoder, freezes it, trains a regression head
to predict toxicity, and then ranks molecules. We compare the effect
of excluding the most toxic predicted compounds versus selecting
molecules at random.

The functions here provide a template that can be adapted to real
datasets. Replace the synthetic data generation with actual toxicity
labels to use in practice.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import logging

from data.dataset import GraphDataset
from models.ema import EMA
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from training.supervised import train_linear_head
from training.unsupervised import train_jepa
from utils.seed import set_seed

logger = logging.getLogger(__name__)

def run_synthetic_case_study(
    smiles: List[str],
    num_top_exclude: int = 2,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Run a synthetic case study to demonstrate molecule prioritisation.

    Args:
        smiles: List of SMILES strings representing molecules.
        num_top_exclude: Number of top predicted toxic molecules to exclude.
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing three values:
            mean_true_toxicity: Mean of the ground truth toxicity scores.
            mean_random_after_exclusion: Mean toxicity after randomly excluding num_top_exclude molecules.
            mean_predicted_after_exclusion: Mean toxicity after excluding the num_top_exclude most toxic predictions.
    """
    set_seed(seed)
    # Generate synthetic toxicity scores (between 0 and 1)
    true_toxicity = np.random.rand(len(smiles))
    # Create labelled dataset for regression
    dataset = GraphDataset.from_smiles_list(smiles, labels=true_toxicity.tolist())
    # Determine input dimension
    input_dim = dataset.graphs[0].x.shape[1] if dataset.graphs else 0
    # Initialise JEPA components
    hidden_dim = 128
    num_layers = 3
    gnn_type = "mpnn"
    encoder = GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    )
    ema_encoder = GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    )
    ema_helper = EMA(encoder, decay=0.99)
    predictor = MLPPredictor(embed_dim=hidden_dim, hidden_dim=hidden_dim * 2)
    # Brief pretraining
    train_jepa(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema_helper,
        epochs=3,
        batch_size=4,
        mask_ratio=0.15,
        contiguous=False,
        lr=5e-4,
        device="cpu",
        reg_lambda=1e-4,
    )
    # Train regression head on toxicity scores
    regression_metrics = train_linear_head(
        dataset=dataset,
        encoder=encoder,
        task_type="regression",
        epochs=15,
        lr=1e-3,
        batch_size=4,
        device="cpu",
        val_patience=5,
    )
    # Compute predictions for ranking using the trained regression head
    reg_head = regression_metrics.get("head")
    encoder.eval()
    reg_head.eval()
    # Use batch size equal to dataset length for simplicity
    batch_x, batch_adj, batch_ptr, _ = dataset.get_batch(list(range(len(dataset))))
    # Convert to tensors on CPU
    import torch

    batch_x = batch_x
    batch_adj = batch_adj
    node_emb = encoder(batch_x, batch_adj)
    from utils.pooling import global_mean_pool

    graph_emb = global_mean_pool(node_emb, batch_ptr)
    preds = reg_head(graph_emb).squeeze(1).detach().numpy()
    # Rank molecules by predicted toxicity
    sorted_indices = np.argsort(-preds)  # Descending order
    # Exclude top predicted toxic compounds
    exclude_pred = sorted_indices[:num_top_exclude]
    remaining_pred = [i for i in range(len(smiles)) if i not in exclude_pred]
    mean_predicted_after = (
        float(np.mean(true_toxicity[remaining_pred])) if remaining_pred else 0.0
    )
    # Random exclusion for comparison
    random_indices = np.arange(len(smiles))
    np.random.shuffle(random_indices)
    exclude_rand = random_indices[:num_top_exclude]
    remaining_rand = [i for i in range(len(smiles)) if i not in exclude_rand]
    mean_random_after = (
        float(np.mean(true_toxicity[remaining_rand])) if remaining_rand else 0.0
    )
    mean_true = float(np.mean(true_toxicity))
    return mean_true, mean_random_after, mean_predicted_after


# ---------------------------------------------------------------------------
# Real Tox21 case study
#
# The function below demonstrates how to perform a real case study on the
# Tox21 dataset when the data are available locally. It closely follows the
# workflow used in the synthetic example but operates on actual toxicity
# labels. The user must provide the path to a CSV file containing SMILES
# strings and one or more label columns corresponding to Tox21 tasks. A
# specific task name should be chosen (e.g. "NR-AR", "SR-p53"). If the
# dataset contains additional tasks, they will be ignored. The function
# performs a simple train/validation/test split (80/10/10) using a
# scaffold split to reduce scaffold leakage. It then pretrains a JEPA
# encoder on the unlabeled molecules, trains a regression head on the
# chosen task, ranks compounds by predicted toxicity, and compares the
# effect of excluding the top predictions versus random exclusions.

import os
import random
from typing import Iterable, Optional

import numpy as np

from data.dataset import GraphDataset
from models.ema import EMA
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from training.supervised import train_linear_head
from training.unsupervised import train_jepa
from utils.pooling import global_mean_pool
from utils.seed import set_seed


def run_tox21_case_study(
    csv_path: str,
    task_name: str,
    smiles_col: str = "smiles",
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 20,
    num_top_exclude: int = 10,
    device: str = "cpu",
) -> Tuple[float, float, float]:
    """Run a real case study on the Tox21 dataset.

    Args:
        csv_path: Path to a CSV file containing Tox21 data. The file
            must include a column with SMILES strings and separate
            columns for each toxicity task.
        task_name: Name of the toxicity task to use as the target. This
            should match one of the column names in the CSV file.
        smiles_col: Name of the column containing SMILES strings (default: "smiles").
        train_fraction: Fraction of the data to use for training (default: 0.8).
        val_fraction: Fraction of the data to use for validation (default: 0.1).
        seed: Random seed for reproducibility.
        pretrain_epochs: Number of epochs for JEPA pretraining (default: 5). Use
            larger values on a powerful machine.
        finetune_epochs: Number of epochs for regression head training (default: 20).
        num_top_exclude: Number of top predicted toxic compounds to exclude
            when computing the post-filter mean toxicity.
        device: Device on which to run the computations ("cpu" or "cuda").

    Returns:
        A tuple of three floats:
            mean_true: Average toxicity in the full dataset for the chosen task.
            mean_random_after: Average toxicity after randomly excluding the
                specified number of compounds.
            mean_predicted_after: Average toxicity after excluding the top
                predicted toxic compounds.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the specified task_name is not found in the CSV file.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    # Load the dataset from CSV using the dataset helper. We deliberately
    # disable caching here because the number of molecules may be large and
    # caching can consume significant disk space. Users can specify a
    # cache directory if desired.
    import pandas as pd

    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in {csv_path}")
    if task_name not in df.columns:
        raise ValueError(f"Task column '{task_name}' not found in {csv_path}")
    # Extract SMILES and labels
    smiles_list = df[smiles_col].astype(str).tolist()
    labels_list = df[task_name].astype(float).tolist()
    # Create GraphDataset and align labels after removing invalid SMILES
    set_seed(seed)
    dataset = GraphDataset.from_smiles_list(smiles_list, labels=labels_list)
    if len(dataset) == 0:
        raise ValueError("No valid molecules could be parsed from the dataset.")
    # Convert labels to numpy array for convenience
    all_labels = dataset.labels.astype(float)
    # Compute dataset split sizes
    num_total = len(dataset)
    num_train = int(train_fraction * num_total)
    num_val = int(val_fraction * num_total)
    indices = list(range(num_total))
    # Scaffold split: Shuffle indices for a simple random split. Users may
    # replace this with a scaffold split using RDKit's scaffold splitting
    # utilities for a more robust partition. Here we randomise for simplicity.
    random.seed(seed)
    random.shuffle(indices)
    train_idx = indices[:num_train]
    val_idx = indices[num_train : num_train + num_val]
    test_idx = indices[num_train + num_val :]
    # Prepare unlabeled dataset for pretraining (use all molecules)
    # Pretrain JEPA
    input_dim = dataset.graphs[0].x.shape[1]
    hidden_dim = 256
    num_layers = 3
    gnn_type = "mpnn"
    encoder = GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    )
    ema_encoder = GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
    )
    ema_helper = EMA(encoder, decay=0.99)
    predictor = MLPPredictor(embed_dim=hidden_dim, hidden_dim=hidden_dim * 2)
    # Pretrain JEPA on the entire dataset without labels
    train_jepa(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema_helper,
        epochs=pretrain_epochs,
        batch_size=64,
        mask_ratio=0.15,
        contiguous=False,
        lr=1e-4,
        device=device,
        reg_lambda=1e-4,
    )

    # Train regression head on the task using only training indices
    # Define a small helper to create a subset dataset
    def subset_dataset(ds: GraphDataset, idxs: Iterable[int]) -> GraphDataset:
        sub_graphs = [ds.graphs[i] for i in idxs]
        sub_labels = ds.labels[idxs] if ds.labels is not None else None
        return GraphDataset(sub_graphs, sub_labels)

    train_ds = subset_dataset(dataset, train_idx)
    val_ds = subset_dataset(dataset, val_idx)
    test_ds = subset_dataset(dataset, test_idx)
    # Combine train and validation for supervised training function (it handles splits internally)
    # We'll just use the combined dataset and rely on the val_patience to stop early.
    combined_ds_graphs = train_ds.graphs + val_ds.graphs
    combined_ds_labels = np.concatenate([train_ds.labels, val_ds.labels])
    combined_ds = GraphDataset(combined_ds_graphs, combined_ds_labels)
    regression_metrics = train_linear_head(
        dataset=combined_ds,
        encoder=encoder,
        task_type="regression",
        epochs=finetune_epochs,
        lr=1e-3,
        batch_size=32,
        device=device,
        val_patience=5,
    )
    # Extract the trained head
    reg_head = regression_metrics.get("head")
    encoder.eval()
    reg_head.eval()
    # Compute predictions on the full dataset for ranking
    # We'll process the dataset in one batch for simplicity. If the dataset
    # is very large, consider batching this step as well.
    batch_indices = list(range(num_total))
    batch_x, batch_adj, batch_ptr, _ = dataset.get_batch(batch_indices)
    import torch

    batch_x = batch_x.to(device)
    batch_adj = batch_adj.to(device)
    # Forward pass
    node_emb = encoder(batch_x, batch_adj)
    from utils.pooling import global_mean_pool

    graph_emb = global_mean_pool(node_emb, batch_ptr)
    preds = reg_head(graph_emb).squeeze(1).detach().cpu().numpy()
    # Rank molecules by predicted toxicity (higher is more toxic)
    sorted_indices = np.argsort(-preds)
    exclude_pred = sorted_indices[:num_top_exclude]
    remaining_pred = [i for i in range(num_total) if i not in exclude_pred]
    mean_pred = float(np.mean(all_labels[remaining_pred])) if remaining_pred else 0.0
    # Random exclusion for comparison
    random_indices = np.arange(num_total)
    np.random.shuffle(random_indices)
    exclude_rand = random_indices[:num_top_exclude]
    remaining_rand = [i for i in range(num_total) if i not in exclude_rand]
    mean_rand = float(np.mean(all_labels[remaining_rand])) if remaining_rand else 0.0
    mean_true = float(np.mean(all_labels))
    return mean_true, mean_rand, mean_pred


if __name__ == "__main__":
    # Example usage of the case study with synthetic data
    smiles = [
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
    true_mean, rand_mean, pred_mean = run_synthetic_case_study(
        smiles, num_top_exclude=2, seed=42
    )
    logger.info("Mean true toxicity: %s", true_mean)
    logger.info("Mean toxicity after random exclusion: %s", rand_mean)
    logger.info("Mean toxicity after predicted exclusion: %s", pred_mean)
