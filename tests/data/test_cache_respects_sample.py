from pathlib import Path

import pandas as pd

from data.mdataset import GraphDataset, _cache_schema_suffix, _write_graph_cache


def test_from_directory_rebuilds_underfilled_cache(tmp_path):
    df = pd.DataFrame({"smiles": ["CC", "CCC", "CCCC", "CCO", "CCN"]})
    parquet_path = tmp_path / "unlabeled.parquet"
    df.to_parquet(parquet_path)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Build a full dataset to capture schema metadata for the cache, then
    # overwrite the cache with a deliberately truncated copy.
    full_ds = GraphDataset.from_parquet(str(parquet_path), cache_dir=str(cache_dir))
    cache_path = cache_dir / f"{parquet_path.stem}_{_cache_schema_suffix(False)}.pkl"

    truncated = GraphDataset(full_ds.graphs[:2], None, full_ds.smiles[:2])
    _write_graph_cache(str(cache_path), truncated)

    rebuilt = GraphDataset.from_directory(
        str(tmp_path), cache_dir=str(cache_dir), max_graphs=5
    )

    assert len(rebuilt.graphs) == 5
    assert len(rebuilt.smiles or []) == 5
    assert Path(rebuilt.source_files[0]).name == parquet_path.name


def test_from_directory_uses_cache_when_capped(tmp_path):
    df = pd.DataFrame({"smiles": ["CC", "CCC", "CCO", "CCN"]})
    parquet_path = tmp_path / "unlabeled.parquet"
    df.to_parquet(parquet_path)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    warm_cache_dir = cache_dir / f"{parquet_path.stem}_{_cache_schema_suffix(False)}"

    # Warm the cache with the full file, then corrupt the source so any
    # attempt to rebuild would fail.
    GraphDataset.from_parquet(str(parquet_path), cache_dir=str(warm_cache_dir))
    parquet_path.write_bytes(b"corrupt")

    # A max_graphs cap triggers n_rows in from_directory; we should still hit
    # the warmed cache and avoid re-reading the corrupted file.
    ds = GraphDataset.from_directory(
        str(tmp_path), cache_dir=str(cache_dir), max_graphs=10
    )

    assert len(ds.graphs) == len(df)
