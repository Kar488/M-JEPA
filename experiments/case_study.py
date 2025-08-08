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

from data.dataset import GraphDataset
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from models.ema import EMA
from training.unsupervised import train_jepa
from training.supervised import train_linear_head
from utils.seed import set_seed


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
    encoder = GNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type)
    ema_encoder = GNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type)
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
    mean_predicted_after = float(np.mean(true_toxicity[remaining_pred])) if remaining_pred else 0.0
    # Random exclusion for comparison
    random_indices = np.arange(len(smiles))
    np.random.shuffle(random_indices)
    exclude_rand = random_indices[:num_top_exclude]
    remaining_rand = [i for i in range(len(smiles)) if i not in exclude_rand]
    mean_random_after = float(np.mean(true_toxicity[remaining_rand])) if remaining_rand else 0.0
    mean_true = float(np.mean(true_toxicity))
    return mean_true, mean_random_after, mean_predicted_after


if __name__ == "__main__":
    # Example usage of the case study with synthetic data
    smiles = [
        "CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOCC", "CNC", "CCCl", "COC", "CCN(CC)CC"
    ]
    true_mean, rand_mean, pred_mean = run_synthetic_case_study(smiles, num_top_exclude=2, seed=42)
    print("Mean true toxicity:", true_mean)
    print("Mean toxicity after random exclusion:", rand_mean)
    print("Mean toxicity after predicted exclusion:", pred_mean)