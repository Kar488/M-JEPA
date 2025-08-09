import pandas as pd
import torch

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
