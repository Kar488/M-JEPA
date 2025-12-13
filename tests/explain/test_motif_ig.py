import json
from pathlib import Path

import numpy as np
import pytest

from data.mdataset import GraphData, GraphDataset
from explain.motif_ig import aggregate_motif_ig, compute_motif_deltas
from training.supervised import _MotifIGArtifactLogger


def test_aggregate_motif_ig_includes_edges():
    node_scores = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    edge_scores = np.array([0.3, -0.1], dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    motif_map = {"ring": [0, 1], "tail": [2]}

    aggregated = aggregate_motif_ig(node_scores, edge_scores, motif_map, edge_index)

    assert pytest.approx(aggregated["ring"]) == -0.7  # 1.0 - 2.0 + 0.3
    assert pytest.approx(aggregated["tail"]) == 0.5


torch = pytest.importorskip("torch")
import torch.nn as nn


def test_compute_motif_deltas_masks_features():
    graph = GraphData(
        x=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )

    def model_fn(ns):
        summed = ns.x.sum()
        first_two = ns.x[:2].sum()
        return torch.stack([summed, first_two])

    motif_map = {"motif_a": [0, 1], "motif_b": [2]}

    deltas = compute_motif_deltas(model_fn, graph, motif_map, device=torch.device("cpu"))

    assert deltas["motif_a"] == [3.0, 3.0]
    assert deltas["motif_b"] == [3.0, 0.0]


def test_motif_logger_writes_artifacts(tmp_path):
    graph = GraphData(
        x=np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    dataset = GraphDataset([graph], labels=[1], smiles=["CO"])

    class DummyEncoder(nn.Module):
        def forward(self, x, edge_index=None, edge_attr=None):
            return x

    encoder = DummyEncoder()
    head = nn.Linear(2, 2)

    logger = _MotifIGArtifactLogger(
        dataset=dataset,
        encoder=encoder,
        head_module=head,
        task_type="classification",
        device=torch.device("cpu"),
        explain_mode="ig_motif",
        explain_config={"output_dir": str(tmp_path), "steps": 2, "task_name": "task"},
        stage_config={"task_name": "task"},
    )

    assert logger.enabled

    logger.process_batch({"batch_indices": [0]})
    metrics: dict = {}
    logger.finalize(metrics)

    assert "ig_motif_artifact_root" in metrics
    assert Path(metrics["ig_motif_artifact_root"]).is_dir()
    records = metrics.get("ig_motif_artifacts")
    assert records
    record = records[0]
    assert Path(record["artifact_dir"]).is_dir()
    assert Path(record["summary_csv"]).is_file()
    assert Path(record["deltas_json"]).is_file()
    with open(record["deltas_json"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload
