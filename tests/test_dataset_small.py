import os

import numpy as np
import pytest

import pandas as pd


import sys


def test_dataset_from_smiles_list(toy_smiles):
    from data.mdataset import GraphDataset

    ds = GraphDataset.from_smiles_list(toy_smiles, add_3d=False)
    assert len(ds.graphs) == len(toy_smiles)
    assert ds.graphs[0].x.ndim == 2
    assert ds.graphs[0].edge_index.shape[0] == 2


@pytest.mark.parametrize("add_3d", [False, True])
def test_dataset_from_file_smoke(tiny_parquet, add_3d):
    #from data.mdataset import GraphDataset

    path = str(tiny_parquet)
    ext = os.path.splitext(path)[1].lower()

    #print("sys.path:", sys.path)
    print("Imported mdataset.py from (expected):", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mdataset.py')))
    try:
        from data.mdataset import GraphDataset
        print("Actual module path:", sys.modules['data.mdataset'].__file__)
        print("GraphDataset has from_smiles_list:", hasattr(GraphDataset, 'from_smiles_list'))
        print("GraphDataset has from_parquet:", hasattr(GraphDataset, 'from_parquet'))
    except Exception as e:
        print("Error importing GraphDataset:", str(e))


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
