from pathlib import Path


def test_tox21_explain_defaults_are_enabled():
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "scripts" / "ci" / "train_jepa_ci.yml"
    content = cfg_path.read_text(encoding="utf-8")

    assert "explain_mode: ${TOX21_EXPLAIN_MODE:-ig,ig_motif}" in content
    assert "explain_steps: ${TOX21_EXPLAIN_STEPS:-50}" in content
