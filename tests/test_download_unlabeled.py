import pandas as pd
from scripts.download_unlabeled import save_parquet, save_shards


def test_save_shards(tmp_path):
    smiles = ["C", "O"]
    save_shards(smiles, tmp_path, shard_size=1)
    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files == ["0000.parquet", "0001.parquet"]
    df = pd.read_parquet(tmp_path / "0000.parquet")
    assert set(df.columns) == {"smiles", "x", "edge_index", "edge_attr"}


def test_save_parquet(tmp_path):
    smiles = ["C", "O"]
    out_file = tmp_path / "unlabelled.parquet"
    save_parquet(smiles, out_file)
    assert out_file.exists()
    df = pd.read_parquet(out_file)
    assert len(df) == 2
