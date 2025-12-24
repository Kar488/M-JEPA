import json
from pathlib import Path

from scripts.ci import generate_graph_visuals


def test_synthetic_graph_visuals_when_dataset_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("DATASET_DIR", raising=False)

    output_dir: Path = tmp_path / "graphs"
    exit_code = generate_graph_visuals.main(
        ["--output-dir", str(output_dir), "--num-samples", "2"]
    )

    assert exit_code == 0

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dataset_path"] == "synthetic"
    assert summary["num_rendered"] == 2
    assert summary["loader"] == "synthetic"

    sample_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir())
    assert len(sample_dirs) >= 2
    for sample_dir in sample_dirs[:2]:
        assert (sample_dir / "molecule.png").is_file()
        assert (sample_dir / "molecule.html").is_file()
        metadata = json.loads((sample_dir / "metadata.json").read_text())
        assert "avg_degree" in metadata
        assert "node_atom_mapping" in metadata
    assert (output_dir / "index.html").is_file()


def test_complexity_metrics_without_smiles():
    graph = generate_graph_visuals.GraphData(3)
    graph.edge_index = [[0, 1, 2], [1, 2, 0]]

    metrics = generate_graph_visuals._complexity_metrics(graph, smiles=None)

    assert metrics["num_nodes"] == 3
    assert metrics["num_edges"] == 3
    assert metrics["avg_degree"] == 2.0
    assert metrics["max_degree"] == 2
    assert metrics["cyclomatic_number"] == 1
