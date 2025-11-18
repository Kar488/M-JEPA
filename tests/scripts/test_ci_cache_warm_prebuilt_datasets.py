import os

import pytest

from scripts.ci import cache_warm_prebuilt_datasets as cache_warm
from scripts.commands import dataset_cache


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
        {"path": str(unlabeled.resolve()), "add_3d": False, "sample": 0},
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
