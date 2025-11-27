import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci import generate_graph_visuals as graph_visuals


def _run_script(dataset: Path, output_dir: Path, *, env=None):
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "scripts/ci/generate_graph_visuals.py",
        "--dataset-path",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--num-samples",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)


def _assert_visuals(output_dir: Path) -> dict:
    assert output_dir.exists(), "output directory should be created"
    pngs = list(output_dir.rglob("*.png"))
    htmls = list(output_dir.rglob("*.html"))
    assert pngs, "expected at least one PNG visual"
    assert htmls, "expected at least one HTML visual"
    summary_path = output_dir / "summary.json"
    assert summary_path.exists(), "summary metadata missing"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("num_rendered", 0) >= 1
    return summary


def test_placeholder_png_is_readable(tmp_path):
    png_path = tmp_path / "molecule.png"
    graph_visuals._render_placeholder_png(png_path, "RDKit rendering failed for sample 1")

    data = png_path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 500, "placeholder should be a non-trivial PNG"

    caption = png_path.with_suffix(".txt").read_text(encoding="utf-8")
    assert caption.strip() == "RDKit rendering failed for sample 1"
    assert caption.endswith("\n")


def test_placeholder_reports_rdkit_import_error(tmp_path, monkeypatch):
    graph = graph_visuals._StubGraphData(num_nodes=1)
    graph.edge_index = [[0, 0], [0, 0]]

    monkeypatch.setattr(graph_visuals, "RDKit_AVAILABLE", False)
    monkeypatch.setattr(graph_visuals, "RDKit_INSTALLED", True)
    monkeypatch.setattr(graph_visuals, "RDKit_IMPORT_ERROR", "ImportError: nope")
    monkeypatch.setattr(graph_visuals, "_MATPLOTLIB_AVAILABLE", False)

    record_dir = tmp_path / "sample"
    renderer = graph_visuals._render_sample(record_dir, graph, "C", None, 0)

    caption = (record_dir / "molecule.txt").read_text(encoding="utf-8").strip()
    assert caption == "RDKit import failed; placeholder molecule"
    assert renderer == "placeholder"


def test_generate_graph_visuals_produces_files(tmp_path):
    dataset = tmp_path / "toy.csv"
    dataset.write_text("smiles\nC1=CC=CC=C1\n", encoding="utf-8")
    output_dir = tmp_path / "visuals"
    _run_script(dataset, output_dir)
    summary = _assert_visuals(output_dir)
    assert summary.get("loader") in {"graphdataset", "fallback", "synthetic"}


def test_generate_graph_visuals_falls_back_when_requested(tmp_path):
    dataset = tmp_path / "toy.csv"
    dataset.write_text("smiles\nC1=CC=CC=C1\n", encoding="utf-8")
    output_dir = tmp_path / "visuals_fallback"
    env = os.environ.copy()
    env["GRAPH_VISUALS_FORCE_FALLBACK"] = "1"
    _run_script(dataset, output_dir, env=env)
    summary = _assert_visuals(output_dir)
    assert summary.get("loader") == "fallback"
    assert summary.get("fallback_forced") is True


def test_load_dataset_recovers_when_graphdataset_returns_zero(tmp_path, monkeypatch):
    dataset = tmp_path / "toy.csv"
    dataset.write_text("smiles\nC\n", encoding="utf-8")

    class _EmptyDataset:
        def __init__(self):
            self.graphs = []
            self.labels = None
            self.smiles = []

    class _EmptyGraphDataset:
        @classmethod
        def from_csv(cls, *_args, **_kwargs):
            return _EmptyDataset()

        @classmethod
        def from_directory(cls, directory: str, **_kwargs):
            csv_path = Path(directory) / "toy.csv"
            return cls.from_csv(str(csv_path))

    monkeypatch.setattr(graph_visuals, "GraphDataset", _EmptyGraphDataset)
    monkeypatch.setattr(graph_visuals, "GRAPH_DATASET_AVAILABLE", True)
    monkeypatch.setattr(graph_visuals, "_FORCE_FALLBACK_LOADER", False)

    dataset_obj, loader = graph_visuals._load_dataset(str(dataset), limit=2)
    assert loader == "fallback"
    assert graph_visuals._graph_count(dataset_obj) > 0


def test_generate_graph_visuals_synthesizes_dataset_when_source_missing(tmp_path):
    missing_dataset = tmp_path / "missing.csv"
    output_dir = tmp_path / "visuals_missing"
    _run_script(missing_dataset, output_dir)
    summary = _assert_visuals(output_dir)
    assert summary.get("loader") == "synthetic"
    assert summary.get("num_graphs", 0) >= 1


def test_guess_extension_requires_supported_entries(tmp_path):
    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "readme.txt").write_text("hi", encoding="utf-8")

    with pytest.raises(ValueError):
        graph_visuals._guess_extension(str(directory))

    csv_path = directory / "toy.csv"
    csv_path.write_text("smiles\nC\n", encoding="utf-8")
    assert graph_visuals._guess_extension(str(directory)) == "csv"
    assert graph_visuals._guess_extension(str(csv_path)) == "csv"


def test_select_indices_even_spacing():
    assert graph_visuals._select_indices(total=0, limit=5) == []
    assert graph_visuals._select_indices(total=3, limit=5) == [0, 1, 2]
    assert graph_visuals._select_indices(total=5, limit=3) == [0, 2, 4]
    assert graph_visuals._select_indices(total=10, limit=1) == [0]


def test_graph_positions_and_edges_fallbacks():
    graph = graph_visuals._StubGraphData(num_nodes=4)
    graph.edge_index = [[0, 1, 1, 2], [1, 0, 2, 2]]
    positions = graph_visuals._graph_positions(graph)
    assert len(positions) == 4
    assert positions[0][2] == pytest.approx(-0.5)

    edges = graph_visuals._unique_edges(graph)
    assert set(edges) == {(0, 1), (1, 2)}


def test_normalise_matrix_handles_iterables():
    assert graph_visuals._normalise_matrix([[1, 2], (3, 4)]) == [[1.0, 2.0], [3.0, 4.0]]
    assert graph_visuals._normalise_matrix(None) is None
