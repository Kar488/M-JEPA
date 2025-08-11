"""Case study using real Tox21 toxicity labels.

This module demonstrates how JEPA embeddings can prioritise molecules by
ranking predictions on the Tox21 dataset. A small encoder is pretrained on
unlabelled molecules, a regression head is fitted on a chosen toxicity task
and the most toxic predictions are compared against a random exclusion
baseline.
"""

from __future__ import annotations
from typing import Iterable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # type-only import; not evaluated at runtime
    from data.mdataset import GraphDataset as GraphDatasetT

import logging
import os
import random
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from models.ema import EMA
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from training.supervised import train_linear_head
from training.unsupervised import train_jepa
from utils.pooling import global_mean_pool
from utils.seed import set_seed

logger = logging.getLogger(__name__)

import sys
import importlib, types
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.mdataset import GraphDataset as GraphDatasetT

def _load_real_graphdataset():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    mod_name = "data.mdataset"              # the real module name
    file_path = data_dir / "mdataset.py"

    # 1) Ensure 'data' package exists and points at your repo's data/ dir
    if "data" not in sys.modules:
        pkg = types.ModuleType("data")
        pkg.__path__ = [str(data_dir)]
        sys.modules["data"] = pkg
    else:
        # make sure its __path__ points to your repo
        sys.modules["data"].__path__ = [str(data_dir)]

    # 2) Build spec for the correct qualified name, create module, and
    #    register it in sys.modules BEFORE exec_module (needed for dataclasses)
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module.GraphDataset

def _import_graphdataset():
    from data.mdataset import GraphDataset
    return GraphDataset


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
    """Run a Tox21 ranking experiment using real labels.

    Args:
        csv_path: Path to a CSV file containing Tox21 data with SMILES and
            toxicity columns.
        task_name: Column name of the toxicity task to predict.
        smiles_col: Name of the column containing SMILES strings.
        train_fraction: Fraction of data used for training.
        val_fraction: Fraction of data used for validation.
        seed: Random seed for reproducibility.
        pretrain_epochs: Number of epochs for JEPA pretraining.
        finetune_epochs: Number of epochs for the regression head.
        num_top_exclude: Number of top predicted toxic compounds to exclude
            when computing the post-filter mean toxicity.
        device: Device on which to run computations.

    Returns:
        Tuple of (mean_true, mean_random_after, mean_predicted_after).

    Raises:
        FileNotFoundError: If the CSV file cannot be located.
        ValueError: If the required columns are missing.
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in {csv_path}")
    if task_name not in df.columns:
        raise ValueError(f"Task column '{task_name}' not found in {csv_path}")

    df = df[[smiles_col, task_name]].dropna(subset=[task_name])
    smiles_list = df[smiles_col].astype(str).tolist()
    labels_list = df[task_name].astype(float).tolist()

    set_seed(seed)

    GraphDatasetCls  = _load_real_graphdataset()

    def subset_dataset(ds: 'GraphDatasetT', idxs: Iterable[int]) -> 'GraphDatasetT':
        sub_graphs = [ds.graphs[i] for i in idxs]
        sub_labels = ds.labels[idxs] if ds.labels is not None else None
        return GraphDatasetCls(sub_graphs, sub_labels)


    dataset = GraphDatasetCls.from_smiles_list(smiles_list, labels=labels_list)
    if len(dataset) == 0:
        raise ValueError("No valid molecules could be parsed from the dataset.")

    all_labels = dataset.labels.astype(float)
    num_total = len(dataset)
    num_train = int(train_fraction * num_total)
    num_val = int(val_fraction * num_total)

    indices = list(range(num_total))
    random.seed(seed)
    random.shuffle(indices)
    train_idx = indices[:num_train]
    val_idx = indices[num_train : num_train + num_val]
    test_idx = indices[num_train + num_val :]

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


    train_ds = subset_dataset(dataset, train_idx)
    val_ds = subset_dataset(dataset, val_idx)
    _ = subset_dataset(dataset, test_idx)

    combined_ds_graphs = train_ds.graphs + val_ds.graphs
    combined_ds_labels = np.concatenate([train_ds.labels, val_ds.labels])
    combined_ds = GraphDatasetCls(combined_ds_graphs, combined_ds_labels)

    regression_metrics = train_linear_head(
        dataset=combined_ds,
        encoder=encoder,
        task_type="regression",
        epochs=finetune_epochs,
        lr=1e-3,
        batch_size=32,
        device=device,
        patience=5,
    )

    reg_head = regression_metrics.get("head")
    encoder.eval()
    reg_head.eval()

    batch_indices = list(range(num_total))
    batch_x, batch_adj, batch_ptr, _ = dataset.get_batch(batch_indices)
    import torch

    batch_x = batch_x.to(device)
    batch_adj = batch_adj.to(device)
    node_emb = encoder(batch_x, batch_adj)
    graph_emb = global_mean_pool(node_emb, batch_ptr)
    preds = reg_head(graph_emb).squeeze(1).detach().cpu().numpy()

    sorted_indices = np.argsort(-preds)
    exclude_pred = sorted_indices[:num_top_exclude]
    remaining_pred = [i for i in range(num_total) if i not in exclude_pred]
    mean_pred = float(np.mean(all_labels[remaining_pred])) if remaining_pred else 0.0

    random_indices = np.arange(num_total)
    np.random.shuffle(random_indices)
    exclude_rand = random_indices[:num_top_exclude]
    remaining_rand = [i for i in range(num_total) if i not in exclude_rand]
    mean_rand = float(np.mean(all_labels[remaining_rand])) if remaining_rand else 0.0

    mean_true = float(np.mean(all_labels))
    return mean_true, mean_rand, mean_pred


if __name__ == "__main__":
    csv = "samples/tox21_mini.csv"
    if os.path.exists(csv):
        true_mean, rand_mean, pred_mean = run_tox21_case_study(
            csv_path=csv,
            task_name="NR-AR",
            pretrain_epochs=1,
            finetune_epochs=1,
            num_top_exclude=1,
        )
        logger.info("Mean true toxicity: %s", true_mean)
        logger.info("Mean toxicity after random exclusion: %s", rand_mean)
        logger.info("Mean toxicity after predicted exclusion: %s", pred_mean)
    else:
        logger.error("Tox21 sample CSV not found: %s", csv)

