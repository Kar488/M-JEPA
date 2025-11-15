import json
import os
import subprocess
import sys
from pathlib import Path


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


def test_generate_graph_visuals_produces_files(tmp_path):
    dataset = tmp_path / "toy.csv"
    dataset.write_text("smiles\nC1=CC=CC=C1\n", encoding="utf-8")
    output_dir = tmp_path / "visuals"
    _run_script(dataset, output_dir)
    summary = _assert_visuals(output_dir)
    assert summary.get("loader") in {"graphdataset", "fallback"}


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
