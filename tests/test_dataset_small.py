import os

import numpy as np
import pytest

import pandas as pd

import sys
import importlib
pytest.importorskip("rdkit")
from data import GraphDataset

def test_dataset_from_smiles_list(toy_smiles):
    ds = GraphDataset.from_smiles_list(toy_smiles, add_3d=False)

    if len(ds.graphs) != len(toy_smiles):
        import pprint, pytest
        pytest.fail(
            "len mismatch | inputs=%d graphs=%d\nsample graph repr:\n%s"
            % (len(toy_smiles), len(ds.graphs), pprint.pformat(ds.graphs[:2]))
        )


    assert len(ds.graphs) == len(toy_smiles)
    assert ds.graphs[0].x.ndim == 2
    assert ds.graphs[0].edge_index.shape[0] == 2


@pytest.mark.parametrize("add_3d", [False, True])
def test_dataset_from_file_smoke(tiny_parquet, add_3d):
    path = str(tiny_parquet)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".parquet":
        ds = GraphDataset.from_parquet(
                path, smiles_col="smiles", cache_dir=None, add_3d=add_3d, n_rows=10
            )        
    else:
        ds = GraphDataset.from_csv(
            path, smiles_col="smiles", cache_dir=None, add_3d=add_3d, n_rows=10
        )
    assert len(ds.graphs) > 0
    # if 3D enabled, feature dim should be >= base (4) + 3
    base_dim = 4
    if add_3d and ds.graphs[0].x.shape[0] > 0:
        assert ds.graphs[0].x.shape[1] >= base_dim + 3
