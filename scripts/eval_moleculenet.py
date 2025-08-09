"""Evaluate frozen encoder on MoleculeNet tasks with linear probes.

This script iterates over a selection of MoleculeNet datasets and trains a
simple linear head on top of a frozen encoder. For each dataset we run the
linear head training for multiple random seeds and report aggregate metrics.

Metrics
-------
* Classification: ROC-AUC and PR-AUC
* Regression: RMSE and MAE

Results for all datasets and seeds are aggregated and written to a CSV file.

The datasets are expected to reside in ``<data_root>/<dataset>/<split>/0000.parquet``
where ``<dataset>`` is one of the supported MoleculeNet tasks and ``<split>`` is
``train``, ``val`` or ``test``. Each parquet file must contain at least two
columns: ``smiles`` and one or more label columns. If multiple label columns
are present (e.g. multi-task datasets) we evaluate a separate linear probe for
each label and average the resulting metrics.

Example
-------

```
python scripts/eval_moleculenet.py \
    --encoder-checkpoint checkpoints/encoder_final.pt \
    --data-root data/moleculenet \
    --output results.csv
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import logging

from data.dataset import GraphDataset
from models.encoder import GNNEncoder
from training.supervised import train_linear_head
from utils.seed import set_seed

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Structured logging
logger = logging.getLogger(__name__)


DATASETS: Dict[str, str] = {
    # Regression tasks
    "qm7": "regression",
    "qm8": "regression",
    "qm9": "regression",
    "esol": "regression",
    "freesolv": "regression",
    "lipophilicity": "regression",
    # Classification tasks
    "bace": "classification",
    "bbbp": "classification",
    "clintox": "classification",
    "sider": "classification",
}

SEEDS: Iterable[int] = range(3)  # 0, 1, 2


# -----------------------------------------------------------------------------
# Dataset loading utilities
# -----------------------------------------------------------------------------

def _read_dataset_files(path: Path) -> pd.DataFrame:
    """Read train/val/test parquet files and concatenate into a single DataFrame."""
    dfs: List[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        file = path / split / "0000.parquet"
        if file.exists():
            dfs.append(pd.read_parquet(file))
    if not dfs:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return pd.concat(dfs, ignore_index=True)


def load_moleculenet_dataset(name: str, root: Path) -> Tuple[GraphDataset, List[str], pd.DataFrame]:
    """Load MoleculeNet dataset and return base dataset plus label columns.

    Returns
    -------
    dataset : GraphDataset
        Dataset without labels (graphs + smiles).
    label_cols : List[str]
        Names of label columns present in the underlying table.
    df : pandas.DataFrame
        The concatenated dataframe with SMILES and labels.
    """
    df = _read_dataset_files(root / name)
    smiles = df["smiles"].astype(str).tolist()
    # Convert SMILES to graph structures once
    base_ds = GraphDataset.from_smiles_list(smiles, labels=None)
    label_cols = [c for c in df.columns if c != "smiles"]
    return GraphDataset(base_ds.graphs, None, base_ds.smiles), label_cols, df


# -----------------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------------

def evaluate_dataset(
    name: str,
    task_type: str,
    encoder: GNNEncoder,
    data_root: Path,
    device: torch.device,
    devices: int,
    patience: int,
) -> Dict[str, float]:
    """Evaluate a dataset over multiple seeds and aggregate metrics."""
    dataset, label_cols, df = load_moleculenet_dataset(name, data_root)
    metrics_accum: Dict[str, List[float]] = {}
    for seed in SEEDS:
        set_seed(seed)
        # For each label column run a separate probe
        for col in label_cols:
            labels = df[col].to_numpy()
            ds = GraphDataset(dataset.graphs, labels, dataset.smiles)
            res = train_linear_head(
                ds,
                encoder,
                task_type,
                device=str(device),
                devices=devices,
                patience=patience,
            )
            # Exclude the trained head from aggregation
            res.pop("head", None)
            for k, v in res.items():
                metrics_accum.setdefault(k, []).append(float(v))
    # Aggregate
    summary: Dict[str, float] = {}
    for k, vals in metrics_accum.items():
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate linear probes on MoleculeNet datasets")
    p.add_argument("--encoder-checkpoint", type=str, required=True, help="Path to encoder checkpoint")
    p.add_argument("--data-root", type=str, default="data/moleculenet", help="Root directory with MoleculeNet datasets")
    p.add_argument("--output", type=str, default="moleculenet_results.csv", help="Where to save aggregated CSV results")
    p.add_argument("--device", type=str, default="cpu", help="Computation device")
    p.add_argument("--input-dim", type=int, default=4, help="Input node feature dimension")
    p.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension of the encoder")
    p.add_argument("--num-layers", type=int, default=3, help="Number of GNN layers")
    p.add_argument("--gnn-type", type=str, default="mpnn", help="Type of GNN encoder (mpnn/gcn/gat)")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs for DDP training")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Load encoder
    encoder = GNNEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
    )
    state = torch.load(args.encoder_checkpoint, map_location=device)
    # Support checkpoints saved either directly or under 'encoder'
    if isinstance(state, dict) and "encoder" in state:
        encoder.load_state_dict(state["encoder"])
    else:
        encoder.load_state_dict(state)
    encoder.to(device)
    encoder.eval()

    data_root = Path(args.data_root)
    results = []
    for name, task_type in DATASETS.items():
        try:
            metrics = evaluate_dataset(
                name, task_type, encoder, data_root, device, args.devices, args.patience
            )
            row = {"dataset": name, **metrics}
            results.append(row)
            logger.info("%s: %s", name, metrics)
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", name, e)

    if results:
        pd.DataFrame(results).to_csv(args.output, index=False)
        logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
