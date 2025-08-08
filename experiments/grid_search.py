"""Hyper‑parameter grid search for JEPA.

This script runs a grid search over various hyper‑parameters for the
JEPA model on a toy dataset. For each configuration it performs a
short unsupervised pretraining, followed by training a linear head on a
simple downstream task. Metrics are recorded and returned as a
pandas DataFrame.

Note: For real experiments you should increase the number of epochs
and use larger datasets. This grid search is intentionally small to
serve as an example and to run quickly in constrained environments.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data.dataset import GraphDataset
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from models.ema import EMA
from training.unsupervised import train_jepa
from training.supervised import train_linear_head


def run_grid_search() -> pd.DataFrame:
    """Execute a comprehensive hyper‑parameter sweep and return averaged results.

    This function iterates over a range of JEPA hyper‑parameters and multiple
    random seeds. For each configuration, it performs a short unsupervised
    pretraining followed by a supervised linear head training. Classification
    metrics are averaged across seeds. The returned DataFrame contains the
    mean ROC‑AUC and PR‑AUC for each configuration.

    Returns:
        A DataFrame summarising configuration parameters and mean performance metrics.
    """
    # Define a small toy dataset with diverse molecules. In a real use case
    # you would replace this with a large unlabelled dataset and proper
    # labelled benchmarks. We deliberately keep it small here for
    # demonstration purposes and fast iteration.
    smiles_list = [
        "CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOCC", "CNC", "CCCl", "COC", "CCN(CC)CC"
    ]
    # Hyper‑parameter ranges. Feel free to extend these lists when running on
    # your own machine to explore a wider parameter space.
    mask_ratios = [0.1, 0.15, 0.25]
    contiguities = [False, True]
    hidden_dims = [128, 256]
    num_layers_list = [2, 3]
    gnn_types = ["mpnn", "gcn", "gat"]
    ema_decays = [0.95, 0.99]
    seeds = [42, 2025, 7]
    # Storage for results
    configs: List[Dict[str, float]] = []
    results: List[Dict[str, float]] = []
    # Loop over hyper‑parameter combinations
    for mask_ratio in mask_ratios:
        for contiguous in contiguities:
            for hidden_dim in hidden_dims:
                for num_layers in num_layers_list:
                    for gnn_type in gnn_types:
                        for ema_decay in ema_decays:
                            # Collect metrics for each seed
                            seed_metrics = []
                            for seed in seeds:
                                from utils.seed import set_seed

                                # Set random seed for reproducibility
                                set_seed(seed)
                                # Create dataset with random labels for this seed
                                labels = np.random.randint(0, 2, size=len(smiles_list))
                                dataset = GraphDataset.from_smiles_list(smiles_list, labels=labels.tolist())
                                # Determine input dimension from dataset
                                input_dim = dataset.graphs[0].x.shape[1] if dataset.graphs else 0
                                # Initialise models
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
                                ema_helper = EMA(encoder, decay=ema_decay)
                                predictor = MLPPredictor(embed_dim=hidden_dim, hidden_dim=hidden_dim * 2)
                                # Unsupervised pretraining (brief for demonstration)
                                train_jepa(
                                    dataset=dataset,
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
                                # Train linear head (classification)
                                metrics = train_linear_head(
                                    dataset=dataset,
                                    encoder=encoder,
                                    task_type="classification",
                                    epochs=10,
                                    lr=5e-3,
                                    batch_size=4,
                                    device="cpu",
                                    val_patience=3,
                                )
                                seed_metrics.append(metrics)
                            # Compute mean metrics across seeds
                            roc_aucs = [m.get("roc_auc", 0.0) for m in seed_metrics]
                            pr_aucs = [m.get("pr_auc", 0.0) for m in seed_metrics]
                            mean_roc = float(np.mean(roc_aucs)) if roc_aucs else 0.0
                            mean_pr = float(np.mean(pr_aucs)) if pr_aucs else 0.0
                            config = {
                                "mask_ratio": mask_ratio,
                                "contiguous": contiguous,
                                "hidden_dim": hidden_dim,
                                "num_layers": num_layers,
                                "gnn_type": gnn_type,
                                "ema_decay": ema_decay,
                            }
                            configs.append(config)
                            results.append({"roc_auc": mean_roc, "pr_auc": mean_pr})
    # Combine configuration parameters and metrics into a single DataFrame
    df = pd.concat([pd.DataFrame(configs), pd.DataFrame(results)], axis=1)
    return df


if __name__ == "__main__":
    df = run_grid_search()
    print(df)
