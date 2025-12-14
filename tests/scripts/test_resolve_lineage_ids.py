import os
import time
from pathlib import Path

from scripts.ci import resolve_lineage_ids as rli


def _touch(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder")
    os.utime(path, (mtime, mtime))


def test_resolve_lineage_prefers_latest_phase1_artifact(tmp_path):
    exp1 = tmp_path / "100" / "grid" / "phase1_export" / "stage-outputs"
    exp2 = tmp_path / "200" / "grid" / "phase1_export" / "stage-outputs"

    older = time.time() - 1000
    newer = time.time()
    _touch(exp1 / "phase1_runs.csv", older)
    _touch(exp2 / "phase1_runs.csv", newer)

    payload = rli._discover_lineage(tmp_path, default_id=None)
    assert payload["grid_exp_id"] == "200"
    assert payload["grid_dir"].endswith("200/grid")
    assert payload["pretrain_exp_id"] == "200"
    assert payload["pretrain_dir"].endswith("200")


def test_resolve_lineage_falls_back_to_default_when_no_artifacts(tmp_path):
    payload = rli._discover_lineage(tmp_path, default_id="321")
    assert payload["grid_exp_id"] == "321"
    assert payload["grid_dir"].endswith("321/grid")
    assert payload["pretrain_exp_id"] == "321"
    assert payload["pretrain_dir"].endswith("321")


def test_resolve_lineage_honours_explicit_default_even_with_artifacts(tmp_path):
    exp_dir = tmp_path / "200" / "grid" / "phase1_export" / "stage-outputs"
    _touch(exp_dir / "phase1_runs.csv", time.time())

    payload = rli._discover_lineage(tmp_path, default_id="999")
    assert payload["grid_exp_id"] == "999"
    assert payload["grid_dir"].endswith("999/grid")
    assert payload["pretrain_exp_id"] == "999"
    assert payload["pretrain_dir"].endswith("999")


def test_resolve_lineage_prefers_existing_default_in_fallback_root(tmp_path):
    cache_root = tmp_path / ".." / "cache"
    exp_dir = cache_root / "123" / "grid" / "phase1_export" / "stage-outputs"
    _touch(exp_dir / "phase1_runs.csv", time.time())

    payload = rli._discover_lineage(tmp_path, default_id="123", fallback_root=cache_root)
    assert payload["grid_exp_id"] == "123"
    assert payload["grid_dir"].startswith(str(cache_root.resolve()))
    assert payload["grid_dir"].endswith("123/grid")
    assert payload["pretrain_exp_id"] == "123"
    assert payload["pretrain_dir"].startswith(str(cache_root.resolve()))
