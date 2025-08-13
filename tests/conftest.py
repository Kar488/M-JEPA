from pathlib import Path

import contextlib
import pandas as pd
import pytest

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logging import maybe_init_wandb

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