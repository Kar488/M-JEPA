import os
import pickle
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.ci import cache_warm_prebuilt_datasets as cache_warm
from scripts.commands import dataset_cache

DEFAULT_SAMPLE_UNLABELED = cache_warm._DEFAULT_SAMPLE_UNLABELED
DEFAULT_MAX_GRAPHS_PER_RUN = cache_warm._DEFAULT_MAX_GRAPHS_PER_RUN
SWEEP_TEMPLATE_DEFAULTS = cache_warm._SWEEP_TEMPLATE_DEFAULTS


@pytest.fixture(autouse=True)
def _enable_data_fallbacks(monkeypatch):
    """Ensure synthetic shards are permitted when datasets are absent."""

    monkeypatch.setenv("MJEPA_ALLOW_DATA_FALLBACKS", "1")


def test_default_sample_unlabeled_tracks_sweep_templates():
    repo_root = Path(__file__).resolve().parents[2]
    expected = None
    for rel_path in SWEEP_TEMPLATE_DEFAULTS:
        template_path = repo_root / rel_path
        if not template_path.exists():
            continue
        with open(template_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        params = cfg.get("parameters", {}) or {}
        sample_cfg = params.get("sample_unlabeled") or {}
        if "value" in sample_cfg:
            expected = int(sample_cfg["value"])
            break
        values = sample_cfg.get("values")
        if isinstance(values, (list, tuple)) and values:
            expected = int(values[0])
            break
    assert DEFAULT_SAMPLE_UNLABELED == (expected or 0)


def test_default_max_graphs_per_run_tracks_sweep_templates():
    repo_root = Path(__file__).resolve().parents[2]
    expected = None
    for rel_path in SWEEP_TEMPLATE_DEFAULTS:
        template_path = repo_root / rel_path
        if not template_path.exists():
            continue
        with open(template_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        params = cfg.get("parameters", {}) or {}
        max_graphs_cfg = params.get("max_graphs_per_run") or {}
        if "value" in max_graphs_cfg:
            expected = int(max_graphs_cfg["value"])
            break
        values = max_graphs_cfg.get("values")
        if isinstance(values, (list, tuple)) and values:
            expected = int(values[0])
            break
    assert DEFAULT_MAX_GRAPHS_PER_RUN == (expected or 250_000)


class DummyDataset:
    def __init__(self, name: str):
        self.name = name


def _setup_loader(monkeypatch, records):
    def fake_loader(dirpath, **kwargs):
        payload = {"dirpath": dirpath, **kwargs}
        records.append(payload)
        return DummyDataset(dirpath)

    monkeypatch.setattr(cache_warm, "load_directory_dataset", fake_loader, raising=False)


def test_cache_warm_builds_missing(monkeypatch, tmp_path):
    calls = []
    _setup_loader(monkeypatch, calls)

    unlabeled = tmp_path / "unlabeled"
    labeled = tmp_path / "labeled"
    unlabeled.mkdir()
    labeled.mkdir()
    cache_dir = tmp_path / "cache"

    argv = [
        "--unlabeled-dir",
        str(unlabeled),
        "--labeled-dir",
        str(labeled),
        "--cache-dir",
        str(cache_dir),
        "--label-col",
        "target",
    ]
    assert cache_warm.main(argv) == 0
    assert len(calls) == 2

    cache_root = os.path.join(str(cache_dir), "prebuilt_datasets")
    assert dataset_cache.cache_exists(
        "unlabeled",
        {
            "path": str(unlabeled.resolve()),
            "add_3d": False,
            "sample": DEFAULT_SAMPLE_UNLABELED,
        },
        cache_root,
    )
    assert dataset_cache.cache_exists(
        "labeled",
        {
            "path": str(labeled.resolve()),
            "add_3d": False,
            "sample": 0,
            "label_col": "target",
        },
        cache_root,
    )


def test_dataframe_iter_chunks(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"smiles": ["a", "b", "c", "d", "e"]})
    df.to_csv(csv_path, index=False)

    chunked_lengths = [len(chunk) for chunk in cache_warm._dataframe_iter(str(csv_path), ".csv", None, 2)]
    assert chunked_lengths == [2, 2, 1]

    full_lengths = [len(chunk) for chunk in cache_warm._dataframe_iter(str(csv_path), ".csv", None, 0)]
    assert full_lengths == [5]


def test_cache_warm_accepts_explicit_prebuilt_dir(monkeypatch, tmp_path, capsys):
    calls = []
    _setup_loader(monkeypatch, calls)

    unlabeled = tmp_path / "unlabeled"
    labeled = tmp_path / "labeled"
    unlabeled.mkdir()
    labeled.mkdir()
    cache_root = tmp_path / "cache" / "graphs_10m" / "prebuilt_datasets"

    argv = [
        "--unlabeled-dir",
        str(unlabeled),
        "--labeled-dir",
        str(labeled),
        "--cache-dir",
        str(cache_root),
    ]

    assert cache_warm.main(argv) == 0

    out = capsys.readouterr().out
    assert "redirecting legacy cache root" not in out
    assert str(cache_root) in out

    assert dataset_cache.cache_exists(
        "unlabeled",
        {
            "path": str(unlabeled.resolve()),
            "add_3d": False,
            "sample": DEFAULT_SAMPLE_UNLABELED,
        },
        str(cache_root),
    )
    assert dataset_cache.cache_exists(
        "labeled",
        {"path": str(labeled.resolve()), "add_3d": False, "sample": 0},
        str(cache_root),
    )


def test_cache_warm_skips_when_present(monkeypatch, tmp_path):
    calls = []
    _setup_loader(monkeypatch, calls)

    unlabeled = tmp_path / "unlabeled"
    labeled = tmp_path / "labeled"
    unlabeled.mkdir()
    labeled.mkdir()
    cache_dir = tmp_path / "cache"

    argv = [
        "--unlabeled-dir",
        str(unlabeled),
        "--labeled-dir",
        str(labeled),
        "--cache-dir",
        str(cache_dir),
    ]

    cache_warm.main(argv)
    calls.clear()
    cache_warm.main(argv)
    assert calls == []


def test_cache_warm_skips_exhausted_manifest(monkeypatch, tmp_path):
    calls = []
    _setup_loader(monkeypatch, calls)

    unlabeled = tmp_path / "unlabeled"
    labeled = tmp_path / "labeled"
    unlabeled.mkdir()
    labeled.mkdir()
    cache_dir = tmp_path / "cache"
    cache_root = cache_dir / "prebuilt_datasets"
    cache_root.mkdir(parents=True)

    unlabeled_payload = cache_warm._normalized_payload(
        str(unlabeled), False, DEFAULT_SAMPLE_UNLABELED, None
    )
    labeled_payload = cache_warm._normalized_payload(str(labeled), False, 5, None)

    for kind, payload, total in (
        ("unlabeled", unlabeled_payload, 10),
        ("labeled", labeled_payload, 3),
    ):
        cache_path = dataset_cache.dataset_cache_path(kind, payload, str(cache_root))
        assert cache_path
        manifest = {
            "__dataset_cache_format__": dataset_cache.SHARDED_CACHE_FORMAT,
            "version": dataset_cache.SHARDED_CACHE_VERSION,
            "shards": [],
            "total_graphs": total,
            "has_labels": kind == "labeled",
            "exhausted": True,
        }
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(manifest, fh)

    argv = [
        "--unlabeled-dir",
        str(unlabeled),
        "--labeled-dir",
        str(labeled),
        "--cache-dir",
        str(cache_dir),
        "--sample-labeled",
        "5",
    ]

    cache_warm.main(argv)
    assert calls == []


def test_cache_warm_force_rebuild(monkeypatch, tmp_path):
    calls = []
    _setup_loader(monkeypatch, calls)

    unlabeled = tmp_path / "unlabeled"
    labeled = tmp_path / "labeled"
    unlabeled.mkdir()
    labeled.mkdir()
    cache_dir = tmp_path / "cache"

    argv = [
        "--unlabeled-dir",
        str(unlabeled),
        "--labeled-dir",
        str(labeled),
        "--cache-dir",
        str(cache_dir),
    ]

    cache_warm.main(argv)
    calls.clear()
    cache_warm.main(argv + ["--force"])
    assert len(calls) == 2


def test_warm_dataset_marks_exhausted_manifest_on_stall(monkeypatch, tmp_path):
    labeled = tmp_path / "labeled"
    labeled.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    payload = cache_warm._normalized_payload(str(labeled), False, 5, None)
    cache_path = dataset_cache.dataset_cache_path("labeled", payload, str(cache_root))
    assert cache_path

    initial_manifest = {
        "__dataset_cache_format__": dataset_cache.SHARDED_CACHE_FORMAT,
        "version": dataset_cache.SHARDED_CACHE_VERSION,
        "shards": [],
        "total_graphs": 3,
        "has_labels": False,
    }
    with open(cache_path, "wb") as fh:
        pickle.dump(initial_manifest, fh)

    def fake_ensure(kind, _payload, builder, _cache_root, *, force=False, log=None):
        assert kind == "labeled"
        assert builder() == dataset_cache.DatasetBuilderResult(data=None, already_persisted=True)
        with open(cache_path, "wb") as manifest_fh:
            pickle.dump(initial_manifest, manifest_fh)

    monkeypatch.setattr(dataset_cache, "ensure_dataset_cache", fake_ensure)

    cache_warm._warm_dataset_in_chunks(
        kind="labeled",
        payload=payload,
        builder=lambda: dataset_cache.DatasetBuilderResult(data=None, already_persisted=True),
        cache_root=str(cache_root),
        sample=5,
        per_run_limit=10,
        force=False,
        log=lambda _msg: None,
    )

    with open(cache_path, "rb") as fh:
        manifest = pickle.load(fh)

    assert manifest.get("exhausted") is True


def test_stream_directory_per_run_cap_resumable(monkeypatch, tmp_path):
    data_dir = tmp_path / "unlabeled"
    data_dir.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_path = cache_root / "unlabeled.pkl"

    monkeypatch.setattr(
        cache_warm, "_list_dataset_files", lambda _: [(os.path.join(str(data_dir), "data.csv"), ".csv")]
    )

    monkeypatch.setattr(
        cache_warm, "_dataframe_iter", lambda *_, **__: (pd.DataFrame({"smiles": [f"s{i}" for i in range(6)]}),)
    )

    def fake_iter_graph_states(smiles, **_kwargs):
        for idx, smi in enumerate(smiles):
            yield idx, {"smiles": smi}

    monkeypatch.setattr(cache_warm._mdataset, "_iter_graph_states", fake_iter_graph_states)
    monkeypatch.setattr(cache_warm._mdataset, "_resolve_worker_count", lambda _num: 0)

    class DummyGraphDataset:
        __module__ = "tests.fake"
        __qualname__ = "GraphDataset"

    cache_warm._mdataset.GraphDataset = DummyGraphDataset

    call_kwargs = dict(
        kind="unlabeled",
        dirpath=str(data_dir),
        cache_path=str(cache_path),
        label_col=None,
        add_3d=False,
        sample=6,
        per_run_limit=2,
        chunk_size=0,
        num_workers=0,
        force=False,
        log=lambda _msg: None,
    )

    cache_warm._stream_directory_to_cache(**call_kwargs)
    manifest_path = Path(cache_path)
    assert not manifest_path.exists()
    parts_dir = Path(f"{cache_path}.parts")
    assert len(list(parts_dir.glob("part-*.pkl"))) == 1

    cache_warm._stream_directory_to_cache(**call_kwargs)
    assert not manifest_path.exists()
    assert len(list(parts_dir.glob("part-*.pkl"))) == 2

    cache_warm._stream_directory_to_cache(**call_kwargs)
    assert manifest_path.exists()
    with open(manifest_path, "rb") as fh:
        manifest = pickle.load(fh)
    assert manifest["total_graphs"] == 6
    assert len(manifest["shards"]) == 3
