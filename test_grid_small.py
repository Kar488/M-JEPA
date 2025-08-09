"""Tiny, fast grid search that runs end-to-end on a small subset."""

from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import GraphDataset
from experiments.grid_search import run_grid_search

# Configure a source parquet (adjust path to one of your shards)
SOURCE = Path("data/ZINC_canonicalized/train-0000.parquet")  # change if needed
TMP = Path("data/tmp_small.parquet")
TMP.parent.mkdir(parents=True, exist_ok=True)

if SOURCE.exists():
    df = pd.read_parquet(SOURCE).head(80)  # keep small
    # ensure smiles col exists
    use_cols = [c for c in df.columns if c.lower() == "smiles"]
    if use_cols:
        smiles_col = use_cols[0]
    else:
        raise SystemExit("Could not find a 'smiles' column in the Parquet subset.")
    df.to_parquet(TMP, index=False)

    def small_dataset_fn(add_3d: bool):
        return GraphDataset.from_parquet(
            filepath=str(TMP),
            smiles_col=smiles_col,
            cache_dir="cache/tmp_small",
            add_3d_features=add_3d,
        )

else:
    # Fallback: small toy dataset
    def small_dataset_fn(add_3d: bool):
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
        labels = np.random.randint(0, 2, size=len(smiles)).tolist()
        return GraphDataset.from_smiles_list(
            smiles, labels=labels, add_3d_features=add_3d
        )


df_res = run_grid_search(
    dataset_fn=small_dataset_fn,
    task_type="classification",
    seeds=(42, 123),
    add_3d_options=(False, True),
    mask_ratios=(0.10, 0.20),
    contiguities=(False, True),
    hidden_dims=(64,),
    num_layers_list=(2,),
    gnn_types=("mpnn", "gcn"),
    ema_decays=(0.95, 0.99),
    pretrain_batch_sizes=(8,),
    finetune_batch_sizes=(4,),
    pretrain_epochs_options=(2,),
    finetune_epochs_options=(2,),
    lrs=(1e-3,),
    device="cpu",
    n_jobs=0,
)

out = Path("outputs/small_grid_results.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df_res.to_csv(out, index=False)
print("Wrote:", out)
print(df_res.head())
