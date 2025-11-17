from __future__ import annotations

import json

import pytest

from scripts import bench


def test_resolve_metric_threshold_known_task():
    rule = bench.resolve_metric_threshold("tox21", "NR-AR")
    assert rule.metric == "roc_auc"
    assert pytest.approx(0.86) == rule.threshold


def test_resolve_metric_threshold_fallback_default():
    rule = bench.resolve_metric_threshold("tox21", "unknown-task")
    assert rule.metric == "roc_auc"
    assert pytest.approx(0.86) == rule.threshold


def test_bench_cli_outputs_json(capsys):
    exit_code = bench.main(["--dataset", "esol"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metric"] == "rmse"
    assert pytest.approx(0.60) == out["threshold"]


def test_coerce_rule_validates_payload():
    with pytest.raises(ValueError):
        bench._coerce_rule({})
    rule = bench._coerce_rule({"metric": "roc_auc", "threshold": "0.8"})
    assert rule.metric == "roc_auc"
    assert pytest.approx(0.8) == rule.threshold


def test_iter_rule_entries_supports_default_and_tasks():
    block = {
        "metric": "roc_auc",
        "threshold": 0.7,
        "nr-ar": {"metric": "roc_auc", "threshold": 0.8},
        "bad": {"metric": "roc_auc"},
    }
    entries = dict(bench._iter_rule_entries(block))
    assert "__default__" in entries
    assert entries["nr-ar"].threshold == pytest.approx(0.8)


def test_parse_simple_yaml_supports_scalars():
    payload = bench._parse_simple_yaml(
        """
foo: true
bar: 1.5
nested:
  child: 2
  name: "NR-AR"
        """.strip()
    )
    assert payload["foo"] is True
    assert pytest.approx(1.5) == payload["bar"]
    assert payload["nested"]["child"] == pytest.approx(2.0)
    assert payload["nested"]["name"] == "NR-AR"


def test_load_rules_uses_simple_parser_when_yaml_missing(tmp_path, monkeypatch):
    manifest = tmp_path / "bench.yml"
    manifest.write_text(
        """
tox21:
  metric: roc_auc
  threshold: 0.6
  nr-ar:
    metric: roc_auc
    threshold: 0.9
        """.strip()
    )
    monkeypatch.setattr(bench, "yaml", None)
    rules = bench.load_rules(manifest)
    assert rules["tox21"]["__default__"].threshold == pytest.approx(0.6)
    assert rules["tox21"]["nr-ar"].threshold == pytest.approx(0.9)


def test_resolve_metric_threshold_requires_known_dataset():
    with pytest.raises(KeyError):
        bench.resolve_metric_threshold("unknown-dataset")
