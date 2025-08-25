from pathlib import Path

import contextlib
import pandas as pd
import pytest

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logging import maybe_init_wandb
import shutil

# --- debug: show which torch is imported under pytest ---
import importlib, sys
try:
    torch = importlib.import_module("torch")
    print("[pytest] torch ->", torch, getattr(torch, "__file__", None))
except Exception as e:
    print("[pytest] torch import failed:", repr(e))
# --------------------------------------------------------

# adding this to support MurckoScaffold failures  in moleculenet_dc tests
def pytest_sessionstart(session):
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold as MS
        if not hasattr(MS, "MurckoScaffoldSmiles"):
            def MurckoScaffoldSmiles(*, smiles=None, mol=None, includeChirality=False):
                if mol is None:
                    if smiles is None:
                        raise ValueError("Provide SMILES or mol")
                    mol = Chem.MolFromSmiles(smiles)
                scaf = MS.GetScaffoldForMol(mol)
                return Chem.MolToSmiles(scaf, isomericSmiles=includeChirality)
            MS.MurckoScaffoldSmiles = MurckoScaffoldSmiles
    except Exception:
        # If RDKit isn't present at all, relevant tests will skip/fail as usual.
        pass

@pytest.fixture(scope="session")
def tiny_parquet(tmp_path_factory):
    # Create a tiny parquet with just a few SMILES
    p = tmp_path_factory.mktemp("data") / "tiny.parquet"
    df = pd.DataFrame(
        {"smiles": ["CCO", "CCC", "CCN", "c1ccccc1", "COC", "CCCl", "CNC"]}
    )
    try:
        df.to_parquet(p, index=False)
    except Exception:
        # If pyarrow/fastparquet not installed, fall back to csv; tests will handle gracefully.
        p = p.with_suffix(".csv")
        df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def toy_smiles():
    return ["CCO", "CCC", "CCN", "c1ccccc1", "COC", "CCCl", "CNC"]


@pytest.fixture(scope="session")
def wb():
    import os, contextlib, wandb

    project = os.getenv("WANDB_PROJECT", "m-jepa")
    mode = "online" if os.getenv("WANDB_API_KEY") else "offline"

    # self-heal if another test has finished our run
    if wandb.run is None:
        wandb.init(
                project=project,
                mode=mode,
                reinit="return_previous",     
                settings=wandb.Settings(start_method="thread"),
            )
   

    # Optional: belt-and-suspenders assert so failures point here
    assert wandb.run is not None, "wandb.init() did not create a run"

    yield wandb
    with contextlib.suppress(Exception):
        wandb.finish()

@pytest.fixture(scope="session", autouse=True)
def clean_artifacts():
    yield
    for d in ("outputs", "reports", "ckpts", "analysis", "cache", "wandb"):
        shutil.rmtree(Path(d), ignore_errors=True)

@pytest.fixture(autouse=True)
def _suite_default_ddp_off(monkeypatch):
    monkeypatch.setenv("DISABLE_DDP", "1")