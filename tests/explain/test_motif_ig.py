import json
from pathlib import Path

import numpy as np
import pytest

from data.mdataset import GraphData, GraphDataset
from explain import motif_ig
from explain.motif_ig import aggregate_motif_ig, compute_motif_deltas, draw_motif_heatmap, save_motif_artifacts
from training.supervised import _MotifIGArtifactLogger


def test_aggregate_motif_ig_includes_edges():
    node_scores = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    edge_scores = np.array([0.3, -0.1], dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    motif_map = {"ring": [0, 1], "tail": [2]}

    aggregated = aggregate_motif_ig(node_scores, edge_scores, motif_map, edge_index)

    assert pytest.approx(aggregated["ring"]) == -0.7  # 1.0 - 2.0 + 0.3
    assert pytest.approx(aggregated["tail"]) == 0.5


def test_find_motifs_from_smiles_includes_ring():
    assert motif_ig._has_rdkit, "RDKit must be available for motif extraction."
    motifs = motif_ig.find_motifs("c1ccccc1")
    ring_keys = [key for key in motifs if key.startswith("ring_")]
    assert ring_keys
    assert all(len(motifs[key]) > 0 for key in ring_keys)


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


def test_save_motif_artifacts_empty_map_has_nonzero_size(tmp_path):
    assert motif_ig._has_rdkit, "RDKit must be available for motif extraction."
    artifacts = save_motif_artifacts(
        smiles="CO",
        motif_map={},
        motif_scores={"molecule": 1.0},
        motif_deltas={"molecule": [0.1]},
        task_names=["task_0"],
        output_dir=str(tmp_path),
        normalise_mode="signed",
    )

    summary_path = Path(artifacts["summary_csv"])
    assert summary_path.is_file()
    with summary_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip().split(",") for line in handle.readlines() if line.strip()]
    assert lines[0] == ["motif", "ig_score", "norm_score", "size"]
    rows = {row[0]: row for row in lines[1:]}
    assert "molecule" in rows
    assert int(rows["molecule"][3]) > 0


def test_draw_motif_heatmap_handles_empty_motifs(tmp_path):
    output_path = tmp_path / "heatmap.png"
    motif_map = {"if_motif": []}

    result = draw_motif_heatmap("CO", {"if_motif": 0.1}, motif_map, str(output_path))

    assert Path(result).is_file()
    assert output_path.stat().st_size > 0


def test_motif_logger_prefers_smiles_for_motifs(tmp_path):
    assert motif_ig._has_rdkit, "RDKit must be available for motif extraction."
    graph = GraphData(
        x=np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    dataset = GraphDataset([graph], labels=[1], smiles=["c1ccccc1"])

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

    logger.process_batch({"batch_indices": [0]})
    metrics: dict = {}
    logger.finalize(metrics)

    record = metrics["ig_motif_artifacts"][0]
    summary_csv = Path(record["summary_csv"])
    with summary_csv.open("r", encoding="utf-8") as handle:
        motifs = [line.split(",")[0] for line in handle.readlines()[1:] if line.strip()]
    assert any(name.startswith("ring_") for name in motifs) or len(motifs) > 1
