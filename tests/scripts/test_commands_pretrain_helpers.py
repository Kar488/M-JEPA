import json
import types
from pathlib import Path

import pytest

import scripts.commands.pretrain as pre


def test_safe_float_and_infer_direction():
    assert pre._safe_float("1.5") == 1.5
    assert pre._safe_float("oops") is None
    assert pre._infer_metric_direction("val_rmse") is False
    assert pre._infer_metric_direction("val_auc") is True
    assert pre._infer_metric_direction(None, True) is True


def test_normalize_and_record_stage_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_OUTPUTS_DIR", str(tmp_path))
    path = pre._normalize_stage_outputs_dir()
    assert path == tmp_path

    payload = {"metric": 0.5}
    pre._record_stage_outputs(payload)
    saved = json.loads((tmp_path / "pretrain.json").read_text())
    assert saved == payload


def test_extract_metric_from_args(monkeypatch):
    args = types.SimpleNamespace(
        best_metric={"name": "val_rmse", "value": 0.2, "higher_is_better": False},
        validation_metric=None,
    )
    metric = pre._extract_metric_from_args(args)
    assert metric == {"name": "val_rmse", "value": 0.2, "higher_is_better": False}

    args2 = types.SimpleNamespace(
        best_metric={},
        validation_metric=0.3,
        validation_metric_name="val_auc",
        validation_metric_higher_is_better="true",
    )
    metric2 = pre._extract_metric_from_args(args2)
    assert metric2["higher_is_better"] is True

    monkeypatch.setenv("PRETRAIN_VAL_METRIC_VALUE", "0.4")
    monkeypatch.setenv("PRETRAIN_VAL_METRIC_NAME", "val_auc")
    metric3 = pre._extract_metric_from_args(types.SimpleNamespace(best_metric={}))
    assert metric3["value"] == 0.4


def test_extract_metric_from_manifest_and_load(tmp_path):
    manifest = {"metrics": {"best": {"name": "rmse", "value": 0.9}}}
    metric = pre._extract_metric_from_manifest(manifest)
    assert metric["value"] == 0.9

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    loaded = pre._load_existing_manifest(path)
    assert loaded == manifest


def test_metric_is_better():
    new = {"name": "val_auc", "value": 0.8, "higher_is_better": True}
    old = {"name": "val_auc", "value": 0.7, "higher_is_better": True}
    assert pre._metric_is_better(new, old)
    assert not pre._metric_is_better({"value": None}, old)
    assert pre._metric_is_better({"name": "loss", "value": 0.2}, {"name": "loss", "value": 0.3})


def test_collect_run_metadata():
    run = types.SimpleNamespace(id="1", name="run", project="p", entity="e", group="g", job_type="job", url="url")
    wb = types.SimpleNamespace(run=run)
    meta = pre._collect_run_metadata(wb)
    assert meta["id"] == "1"

