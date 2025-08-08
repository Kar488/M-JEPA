"""Entry point for running JEPA experiments and demonstrations.

This module illustrates how to use the modular codebase for both toy
examples and real datasets. It contains:

1. A simple demonstration using a small list of SMILES strings to
   pretrain JEPA, train a contrastive baseline, fine‑tune a linear
   head, run a synthetic case study, and display plots.
2. Helper functions to load datasets from Parquet files, set up the
   JEPA components, pretrain on unlabelled data, and fine‑tune on
   labelled tasks. These functions make it easy to swap in real
   datasets without duplicating code.
3. A function to run the hyper‑parameter grid search on a toy dataset
   for quick experimentation.

To run the toy demonstration, execute this script directly. To use
real datasets, download the appropriate Parquet files, then call
the helper functions from your own script or interactive session.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from data.dataset import GraphDataset
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from models.ema import EMA
from training.unsupervised import train_jepa, train_contrastive
from training.supervised import train_linear_head
from experiments.grid_search import run_grid_search
from experiments.case_study import run_synthetic_case_study
from utils.plotting import plot_training_curves, plot_hyperparameter_results


def load_parquet_dataset(
    filepath: str,
    smiles_col: str = "smiles",
    label_col: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> GraphDataset:
    """Load a dataset from a Parquet file using the GraphDataset loader.

    Args:
        filepath: Path to the Parquet file on disk.
        smiles_col: Name of the column containing SMILES strings.
        label_col: Name of the label column (for supervised tasks). If
            None, the dataset will be treated as unlabeled.
        cache_dir: Directory to store cached GraphData objects. If
            provided, preprocessed graphs will be saved and reused.

    Returns:
        A GraphDataset instance.
    """
    return GraphDataset.from_parquet(
        filepath=filepath,
        smiles_col=smiles_col,
        label_col=label_col,
        cache_dir=cache_dir,
    )


def setup_jepa(
    input_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    gnn_type: str = "mpnn",
    ema_decay: float = 0.99,
) -> Tuple[GNNEncoder, GNNEncoder, EMA, MLPPredictor]:
    """Initialise JEPA model components.

    Args:
        input_dim: Dimension of node features.
        hidden_dim: Dimension of hidden representations.
        num_layers: Number of GNN layers.
        gnn_type: Type of GNN ("mpnn", "gcn" or "gat").
        ema_decay: Decay rate for the exponential moving average.

    Returns:
        A tuple (encoder, ema_encoder, ema_helper, predictor).
    """
    encoder = GNNEncoder(input_dim, hidden_dim, num_layers, gnn_type)
    ema_encoder = GNNEncoder(input_dim, hidden_dim, num_layers, gnn_type)
    ema_helper = EMA(encoder, decay=ema_decay)
    predictor = MLPPredictor(embed_dim=hidden_dim, hidden_dim=hidden_dim * 2)
    return encoder, ema_encoder, ema_helper, predictor


def pretrain_jepa(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    ema_encoder: GNNEncoder,
    ema_helper: EMA,
    predictor: MLPPredictor,
    epochs: int = 100,
    batch_size: int = 256,
    mask_ratio: float = 0.15,
    contiguous: bool = False,
    lr: float = 1e-4,
    device: str = "cpu",
    reg_lambda: float = 1e-4,
) -> None:
    """Pretrain JEPA on an unlabelled dataset."""
    train_jepa(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema_helper,
        epochs=epochs,
        batch_size=batch_size,
        mask_ratio=mask_ratio,
        contiguous=contiguous,
        lr=lr,
        device=device,
        reg_lambda=reg_lambda,
    )


def fine_tune(
    dataset: GraphDataset,
    encoder: GNNEncoder,
    task_type: str = "regression",
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    val_patience: int = 5,
) -> Dict[str, float]:
    """Train a linear head on top of a frozen encoder for a supervised task."""
    metrics = train_linear_head(
        dataset=dataset,
        encoder=encoder,
        task_type=task_type,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        val_patience=val_patience,
    )
    return metrics


def demonstration() -> None:
    """Run a self‑contained demonstration using a toy dataset.

    This function pretrains JEPA and a contrastive baseline on a small
    set of SMILES strings, fine‑tunes a linear head on a mock
    classification task, and runs a synthetic case study. It also
    visualises the training losses and reports metrics.
    """
    # ------------------------------------------------------------------
    # Toy dataset (mock) – can be replaced with a real dataset via
    # load_parquet_dataset() if desired.
    smiles = [
        "CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOCC", "CNC", "CCCl", "COC", "CCN(CC)CC"
    ]
    dataset = GraphDataset.from_smiles_list(smiles)

    # ------------------------------------------------------------------
    # JEPA pretraining on toy data
    input_dim = dataset.graphs[0].x.shape[1]
    encoder, ema_encoder, ema_helper, predictor = setup_jepa(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        gnn_type="mpnn",
        ema_decay=0.99,
    )
    jepa_losses = train_jepa(
        dataset=dataset,
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
    # Contrastive baseline pretraining
    contrastive_encoder = GNNEncoder(input_dim=input_dim, hidden_dim=64, num_layers=2, gnn_type="mpnn")
    contrastive_losses = train_contrastive(
        dataset=dataset,
        encoder=contrastive_encoder,
        projection_dim=32,
        epochs=3,
        batch_size=5,
        mask_ratio=0.2,
        lr=1e-3,
        device="cpu",
        temperature=0.1,
    )
    # Plot training curves (normalised to the first epoch)
    plot_training_curves(
        {"JEPA": jepa_losses, "Contrastive": contrastive_losses},
        title="Toy Unsupervised Training Losses",
        normalize=True,
    )
    # ------------------------------------------------------------------
    # Supervised fine‑tuning on a toy classification task
    labels = np.random.randint(0, 2, size=len(smiles))
    labeled_dataset = GraphDataset.from_smiles_list(smiles, labels=labels.tolist())
    metrics = train_linear_head(
        dataset=labeled_dataset,
        encoder=encoder,
        task_type="classification",
        epochs=5,
        lr=1e-3,
        batch_size=5,
        device="cpu",
    )
    metrics_printable = {k: v for k, v in metrics.items() if k != "head"}
    print("Classification metrics after JEPA pretraining:", metrics_printable)
    # ------------------------------------------------------------------
    # Synthetic case study demonstration
    mean_true, mean_rand, mean_pred = run_synthetic_case_study(smiles, num_top_exclude=2, seed=42)
    print(
        f"Case study – mean true toxicity: {mean_true:.3f}, after random exclusion: {mean_rand:.3f}, after predicted exclusion: {mean_pred:.3f}"
    )


def grid_search_demo() -> None:
    """Run the hyper‑parameter grid search on a toy dataset and display results."""
    df = run_grid_search()
    print(df)
    if "roc_auc" in df.columns:
        # Create index labels for plotting
        df_plot = df.copy()
        df_plot.index = df.apply(
            lambda row: f"mask={row['mask_ratio']} contig={row['contiguous']} hid={row['hidden_dim']} lay={row['num_layers']} gnn={row['gnn_type']} ema={row['ema_decay']}",
            axis=1,
        )
        # Display only the top 15 configurations for readability
        plot_hyperparameter_results(
            df_plot,
            metric="roc_auc",
            title="ROC‑AUC Across Hyper‑parameters",
            top_n=15,
        )


if __name__ == "__main__":
    # Example usage: run the toy demonstration and grid search
    demonstration()
    grid_search_demo()

    # ------------------------------------------------------------------
    # To use real data, comment out the above calls and adapt the
    # following example to your file paths and tasks. This code is
    # provided as a template and will not run until you replace the
    # file paths with actual Parquet files on your machine.
    #
    # Example (pseudo‑code):
    #
    # # 1. Load unlabelled dataset (e.g. ZINC)
    # zinc_ds = load_parquet_dataset(
    #     filepath="path/to/ZINC_canonicalized.parquet",
    #     smiles_col="smiles",
    #     cache_dir="cache/zinc"
    # )
    #
    # # 2. Set up JEPA components based on input feature dimension
    # dim = zinc_ds.graphs[0].x.shape[1]
    # encoder, ema_encoder, ema_helper, predictor = setup_jepa(
    #     input_dim=dim,
    #     hidden_dim=256,
    #     num_layers=3,
    #     gnn_type="mpnn",
    #     ema_decay=0.99
    # )
    #
    # # 3. Pretrain JEPA on the unlabelled ZINC data
    # # If you want to capture the loss history for plotting, call
    # # train_jepa() directly and store the returned list. Otherwise,
    # # you can use pretrain_jepa() for a simple call that does not
    # # record losses.
    # jepa_losses = train_jepa(
    #     dataset=zinc_ds,
    #     encoder=encoder,
    #     ema_encoder=ema_encoder,
    #     predictor=predictor,
    #     ema=ema_helper,
    #     epochs=100,
    #     batch_size=256,
    #     mask_ratio=0.15,
    #     contiguous=False,
    #     lr=1e-4,
    #     device="cuda",
    #     reg_lambda=1e-4
    # )
    # # Optionally train a contrastive baseline on the same data
    # contrastive_encoder = GNNEncoder(
    #     input_dim=dim, hidden_dim=256, num_layers=3, gnn_type="mpnn"
    # )
    # contrastive_losses = train_contrastive(
    #     dataset=zinc_ds,
    #     encoder=contrastive_encoder,
    #     projection_dim=128,
    #     epochs=100,
    #     batch_size=256,
    #     mask_ratio=0.15,
    #     lr=1e-4,
    #     device="cuda",
    #     temperature=0.1
    # )
    # # Plot the JEPA and contrastive loss curves
    # plot_training_curves(
    #     {"JEPA": jepa_losses, "Contrastive": contrastive_losses},
    #     title="Real‑data Unsupervised Training Losses",
    #     normalize=True
    # )
    #
    # # 4. Load a labelled MoleculeNet task for fine‑tuning (e.g. ESOL)
    # esol_ds = load_parquet_dataset(
    #     filepath="path/to/esol.parquet",
    #     smiles_col="smiles",
    #     label_col="ESOL",
    #     cache_dir="cache/esol"
    # )
    #
    # # 5. Train a linear head for regression
    # regression_metrics = fine_tune(
    #     dataset=esol_ds,
    #     encoder=encoder,
    #     task_type="regression",
    #     epochs=50,
    #     lr=1e-3,
    #     batch_size=64,
    #     device="cuda",
    #     val_patience=5
    # )
    # print({k: v for k, v in regression_metrics.items() if k != "head"})
