"""Tiny, fast grid search that runs end-to-end on a small subset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Configure a source parquet (adjust path to one of your shards)
SOURCE = Path("data/ZINC-canonicalized/train-00000-of-00003-1dd8e62fc2556455.parquet")  # change if needed

import sys, types, importlib.util 

# Monkeypatch tqdm to use a dummy class that does nothing
class DummyTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable or []
    def __iter__(self):
        return iter(self.iterable)
    def update(self, n=1):
        pass
    def set_description(self, desc=None):
        pass
    def close(self):
        pass

@pytest.fixture(autouse=True)
def _silence_tqdm(monkeypatch):
    """Make tqdm think we're not in a TTY so it doesn't draw a progress bar."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

@pytest.fixture
def tmp_parquet(tmp_path):
    path = tmp_path / "tmp_small.parquet"
    yield path

@pytest.fixture(autouse=True)
def _ddp_sane_env(monkeypatch):
    # either disable entirely:
    monkeypatch.setenv("DISABLE_DDP", "1")
    # or, if you prefer to allow single-process DDP, use loopback:
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_port()))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


import socket, contextlib, pytest, os

def _free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
    
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

GraphDataset = _load_real_graphdataset()

def test_grid_search_small(wb,tmp_parquet):

    pytest.importorskip("rdkit")
    from experiments.grid_search import run_grid_search

    if SOURCE.exists():
        try:
            df = pd.read_parquet(SOURCE).head(20)  # keep small
            source_valid = True
        except Exception:
            df = None
            # force fallback to toy dataset below
            source_valid = False
    else:
        df = None
        source_valid = False

    if source_valid and df is not None:
        # ensure smiles col exists
        use_cols = [c for c in df.columns if c.lower() == "smiles"]
        if use_cols:
            smiles_col = use_cols[0]
        else:
            raise SystemExit("Could not find a 'smiles' column in the Parquet subset.")
        
        df["label"] = np.random.randint(0, 2, size=len(df)) # fake labels since its not in dataset
        df.to_parquet(tmp_parquet, index=False) 

        def small_dataset_fn(add_3d: bool):
            
            ds =  GraphDataset.from_parquet(
                filepath=str(tmp_parquet),
                smiles_col=smiles_col,
                label_col="label", # label column is somehow ignored by loader
                cache_dir=None,
                add_3d=add_3d,
            )
            # hard‑set labels from the Parquet file
            # import numpy as np, pandas as pd
            # y = pd.read_parquet(TMP, columns=["label"])["label"].to_numpy()
            # ds.labels = y.astype(int)  # or float for regression

            return ds

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
            labels = np.random.randint(0, 2, size=len(smiles)).tolist() # fake labels since its not in dataset
            return GraphDataset.from_smiles_list(
                smiles, labels=labels, add_3d=add_3d
            )
    # Run the grid search with the small dataset function
    # Note: this will run on CPU only, so adjust params accordingly
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
    wb.log({"output_csv": str(out), "rows": len(df_res)})
