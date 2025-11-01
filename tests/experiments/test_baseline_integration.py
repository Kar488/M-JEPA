from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
bi = pytest.importorskip("experiments.baseline_integration")

try:
    import pyarrow  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    try:
        import fastparquet  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover
        pytest.skip(
            "pyarrow or fastparquet is required for parquet tests",
            allow_module_level=True,
        )


def test_concat_dir_to_csv(tmp_path, wb):
    df_csv = pd.DataFrame({"smiles": ["a"], "label": [1], "extra": [0]})
    df_csv.to_csv(tmp_path / "part1.csv", index=False)
    df_parquet = pd.DataFrame({"smiles": ["b"], "label": [2], "extra": [1]})
    df_parquet.to_parquet(tmp_path / "part2.parquet", index=False)
    out = tmp_path / "out.csv"
    path = bi.concat_dir_to_csv(str(tmp_path), "smiles", "label", str(out))
    merged = pd.read_csv(path)
    assert list(merged["smiles"]) == ["a", "b"]
    wb.log({"concat_rows": len(merged)})


def test_baseline_pretrain_and_embed_cli(tmp_path, monkeypatch, wb):
    calls = {"train": 0, "embed": 0}

    class DummyCLI:
        def __init__(self, cfg_path):
            self.cfg = cfg_path

        def outputs_dir(self, method):
            return str(tmp_path)

        def train(self, method, unlabeled, out_dir):
            calls["train"] += 1
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "best.ckpt").write_text("ckpt")

        def embed(self, method, ckpt_path, smiles_file, emb_out):
            calls["embed"] += 1
            Path(emb_out).parent.mkdir(parents=True, exist_ok=True)
            np.save(emb_out, np.zeros((2, 2)))

    monkeypatch.setattr(bi, "BaselineCLI", DummyCLI)
    ckpt, emb = bi.baseline_pretrain_and_embed(
        "molclr", "unlabeled.csv", str(tmp_path / "eval.csv"), force_embed=True
    )
    assert Path(ckpt).exists() and Path(emb).exists()
    assert calls == {"train": 1, "embed": 1}
    # Second call skips training but embeds again due to force_embed
    ckpt2, emb2 = bi.baseline_pretrain_and_embed(
        "molclr", "unlabeled.csv", str(tmp_path / "eval.csv"), force_embed=True
    )
    assert calls == {"train": 1, "embed": 2}
    wb.log({"baseline_embed_calls": calls["embed"]})


def test_baseline_pretrain_and_embed_native(tmp_path, monkeypatch, wb):
    class DummyNative:
        def __init__(self, repo, train_cfg, embed_cfg):
            self.repo = repo
            self.train_cfg = train_cfg
            self.embed_cfg = embed_cfg

        def train(self, unlabeled, method):
            p = tmp_path / "native.ckpt"
            p.write_text("ckpt")
            return str(p)

        def embed(self, checkpoint, smiles_file):
            out = tmp_path / "native.npy"
            np.save(out, np.zeros(1))
            return str(out)

    monkeypatch.setattr(bi, "_HAS_NATIVE", True)
    monkeypatch.setattr(bi, "NativeBaseline", DummyNative)
    ckpt, emb = bi.baseline_pretrain_and_embed(
        "geomgcl",
        "unlabeled.smi",
        "eval.smi",
        use_native=True,
        native_repo="repo",
        native_train={"module": "m", "function": "f"},
        native_embed={"module": "m2", "function": "f2"},
    )
    assert Path(ckpt).exists() and Path(emb).exists()
    wb.log({"native_ckpt_exists": Path(ckpt).exists()})