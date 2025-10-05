from __future__ import annotations

import json

import pytest

from scripts import bench


def test_resolve_metric_threshold_known_task():
    rule = bench.resolve_metric_threshold("tox21", "NR-AR")
    assert rule.metric == "roc_auc"
    assert pytest.approx(0.65) == rule.threshold


def test_resolve_metric_threshold_fallback_default():
    rule = bench.resolve_metric_threshold("tox21", "unknown-task")
    assert rule.metric == "roc_auc"
    assert pytest.approx(0.65) == rule.threshold


def test_bench_cli_outputs_json(capsys):
    exit_code = bench.main(["--dataset", "esol"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metric"] == "rmse"
    assert pytest.approx(0.60) == out["threshold"]
