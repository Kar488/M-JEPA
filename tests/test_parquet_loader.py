import pandas as pd
import torch
import pytest

pytest.importorskip("torch_geometric")

from data.parquet_loader import load_dataloaders


def _write_split(path, rows):
    df = pd.DataFrame(rows)
    # ensure nested lists are preserved on readback
    pd.options.io.parquet.engine = "fastparquet"
    df.to_parquet(path)


def test_load_dataloaders(tmp_path):
    rows = [
        {
            "x": [[1.0], [2.0]],
            "edge_index": [[0, 1], [1, 0]],
            "edge_attr": [[1.0], [1.0]],
            "y": 0,
        },
        {
            "x": [[3.0], [4.0]],
            "edge_index": [[0, 1], [1, 0]],
            "edge_attr": [[1.0], [1.0]],
            "y": 1,
        },
    ]
    for split in ("train", "val", "test"):
        _write_split(tmp_path / f"{split}.parquet", rows)

    train_loader, val_loader, test_loader = load_dataloaders(tmp_path, batch_size=2)

    assert len(train_loader.dataset) == 2
    batch = next(iter(train_loader))
    assert batch.x.shape == (4, 1)
    assert batch.edge_index.shape[1] == 4
    assert sorted(batch.y.tolist()) == [0, 1]

def test_load_dataloaders_batching_and_order(tmp_path):
    rows = [
        {
            "x": [[float(i)], [float(i + 0.1)]],
            "edge_index": [[0, 1], [1, 0]],
            "edge_attr": [[1.0], [1.0]],
            "y": i,
        }
        for i in range(3)
    ]
    for split in ("train", "val", "test"):
        _write_split(tmp_path / f"{split}.parquet", rows)

    train_loader, val_loader, test_loader = load_dataloaders(tmp_path, batch_size=2)

    train_batches = list(train_loader)
    assert [b.num_graphs for b in train_batches] == [2, 1]
    assert train_batches[0].x.shape[0] == 4
    assert train_batches[1].x.shape[0] == 2

    val_y = []
    for batch in val_loader:
        val_y.extend(batch.y.tolist())
    test_y = []
    for batch in test_loader:
        test_y.extend(batch.y.tolist())
    assert val_y == [0, 1, 2]
    assert test_y == [0, 1, 2]
