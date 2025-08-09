from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from data.dataset import GraphDataset
from models.ema import EMA
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from training.supervised import train_linear_head
from training.unsupervised import train_jepa


@dataclass(frozen=True)
class Config:
    """Configuration for a single ablation run."""

    mask_ratio: float
    contiguous: bool
    hidden_dim: int
    num_layers: int
    ema_decay: float
    gnn_type: str


def run_ablation() -> pd.DataFrame:
    """Run a small ablation study across GNN types.

    The experiment trains a tiny JEPA model and then evaluates a linear head
    for both classification and regression tasks. Metrics for each
    configuration are returned in a DataFrame.
    """

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

    mask_ratios = [0.1, 0.15, 0.25]
    contiguities = [False, True]
    hidden_dims = [128, 256]
    layers_list = [2, 3]
    ema_decays = [0.95, 0.99]
    gnn_types = ["mpnn", "gcn", "gat"]

    configs: List[Dict[str, object]] = []
    results: List[Dict[str, float]] = []

    for gnn_type in gnn_types:
        for mask_ratio in mask_ratios:
            for contiguous in contiguities:
                for hidden_dim in hidden_dims:
                    for num_layers in layers_list:
                        for ema_decay in ema_decays:
                            encoder = GNNEncoder(
                                input_dim=2,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                gnn_type=gnn_type,
                            )
                            ema_encoder = GNNEncoder(
                                input_dim=2,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                gnn_type=gnn_type,
                            )
                            ema_helper = EMA(encoder, decay=ema_decay)
                            predictor = MLPPredictor(
                                embed_dim=hidden_dim, hidden_dim=hidden_dim * 2
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
                            cfg = Config(
                                mask_ratio=mask_ratio,
                                contiguous=contiguous,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                ema_decay=ema_decay,
                                gnn_type=gnn_type,
                            )
                            res = {
                                "roc_auc": class_metrics.get("roc_auc", float("nan")),
                                "pr_auc": class_metrics.get("pr_auc", float("nan")),
                                "rmse": reg_metrics.get("rmse", float("nan")),
                                "mae": reg_metrics.get("mae", float("nan")),
                            }
                            configs.append(asdict(cfg))
                            results.append(res)

    df = pd.concat([pd.DataFrame(configs), pd.DataFrame(results)], axis=1)
    return df


__all__ = ["Config", "run_ablation"]
