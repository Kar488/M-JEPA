import importlib
import os
import shutil
import sys
from pathlib import Path
  
import pytest
sys.path.insert(0, os.path.abspath(Path(__file__).resolve().parent.parent))


# Ensure project 'training' beats tests/training shadow
tests_root = Path(__file__).resolve().parent
repo_root = tests_root.parent
for mod in ("training", "training.supervised"):
    if mod in sys.modules:
        m = sys.modules[mod]
        if getattr(m, "__file__", "") and str(tests_root) in m.__file__:
            del sys.modules[mod]

# --------------------------------------------------------
# adding this to support MurckoScaffold failures  in moleculenet_dc tests
def pytest_sessionstart(session):

    import sys, torch
    print(f"\n[debug] python: {sys.executable}")
    print(f"[debug] torch: {getattr(torch,'__file__',None)}  v={getattr(torch,'__version__',None)}")
    import torch.nn as nn
    print(f"[debug] torch.nn: {getattr(nn,'__file__',None)}  has Linear? {'Linear' in dir(nn)}")

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
    pd = pytest.importorskip("pandas")
    p = tmp_path_factory.mktemp("data") / "tiny.parquet"
    df = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCC",
                "CCN",
                "c1ccccc1",
                "COC",
                "CCCl",
                "CNC",
            ]
        }
    )
    try:
        df.to_parquet(p, index=False)
    except Exception:
        # If pyarrow/fastparquet not installed, fall back to csv;
        # tests will handle this gracefully.
        p = p.with_suffix(".csv")
        df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def toy_smiles():
    return ["CCO", "CCC", "CCN", "c1ccccc1", "COC", "CCCl", "CNC"]


@pytest.fixture(scope="session")
def wb():
    import contextlib
    import os

    wandb = pytest.importorskip("wandb")

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