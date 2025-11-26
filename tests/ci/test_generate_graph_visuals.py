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

    for idx in (0, 1):
        sample_dir = output_dir / f"sample_{idx:03d}"
        assert (sample_dir / "molecule.png").is_file()
        assert (sample_dir / "molecule.html").is_file()
