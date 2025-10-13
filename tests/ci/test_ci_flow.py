import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd, env):
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def test_pretrain_tox21_dry_run(tmp_path):
    experiments_root = tmp_path / "experiments"
    shim_path = tmp_path / "stage_shim.sh"
    shim_path.write_text("""#!/usr/bin/env bash
set -euo pipefail
stage="$1"
case "$stage" in
  pretrain)
    mkdir -p "$PRETRAIN_DIR" "$PRETRAIN_DIR/stage-outputs" "$PRETRAIN_ARTIFACTS_DIR"
    printf 'stub' > "$PRETRAIN_DIR/encoder.pt"
    cat <<JSON > "$PRETRAIN_ARTIFACTS_DIR/encoder_manifest.json"
{"paths":{"encoder":"$PRETRAIN_DIR/encoder.pt"}}
JSON
    cat <<JSON > "$PRETRAIN_DIR/stage-outputs/pretrain.json"
{"manifest_path":"$PRETRAIN_ARTIFACTS_DIR/encoder_manifest.json","encoder_checkpoint":"$PRETRAIN_DIR/encoder.pt"}
JSON
    ;;
  tox21)
    mkdir -p "$TOX21_DIR/stage-outputs"
    cat <<'JSON' > "$TOX21_DIR/stage-outputs/tox21_pretrain_frozen.json"
{"met_benchmark": true}
JSON
    ;;
  *)
    ;;
esac
""")
    shim_path.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": "1759795103",
            "MJEPACI_STAGE_SHIM": str(shim_path),
            "GITHUB_ENV": str(tmp_path / "github_env"),
            "WANDB_API_KEY": "",
        }
    )

    _run(["bash", "scripts/ci/run-pretrain.sh"], env)
    _run(["bash", "scripts/ci/run-tox21.sh"], env)

    manifest_path = experiments_root / "1759795103" / "artifacts" / "encoder_manifest.json"
    state_path = experiments_root / "1759795103" / "pretrain_state.json"
    legacy_state = experiments_root / "pretrain_state.json"

    assert manifest_path.is_file(), manifest_path
    assert state_path.is_file(), state_path
    assert legacy_state.is_file(), legacy_state
    assert legacy_state.read_text(encoding="utf-8") == state_path.read_text(encoding="utf-8")

    with state_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["id"] == "1759795103"
    assert Path(payload["encoder_manifest"]).name == "encoder_manifest.json"

    all_paths = {str(p) for p in experiments_root.rglob("*")}
    assert all("18296078427" not in p for p in all_paths)


def test_pretrain_materializes_phase2_artifacts(tmp_path):
    experiments_root = tmp_path / "experiments"
    phase2_root = tmp_path / "phase2" / "winner"
    phase2_root.mkdir(parents=True)

    remote_checkpoint = phase2_root / "encoder.pt"
    remote_checkpoint.write_text("phase2-weights", encoding="utf-8")
    remote_manifest = phase2_root / "encoder_manifest.json"
    remote_manifest.write_text(
        json.dumps({"paths": {"encoder": str(remote_checkpoint)}}) + "\n",
        encoding="utf-8",
    )

    shim_path = tmp_path / "phase2_stage_shim.sh"
    shim_path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
stage=\"$1\"
case \"$stage\" in
  pretrain)
    mkdir -p \"$PRETRAIN_DIR\" \"$PRETRAIN_DIR/stage-outputs\" \"$PRETRAIN_ARTIFACTS_DIR\"
    cp \"{remote_checkpoint}\" \"$PRETRAIN_DIR/encoder.pt\"
    cat <<JSON > \"$PRETRAIN_DIR/stage-outputs/pretrain.json\"
{{\"manifest_path\":\"{remote_manifest}\",\"encoder_checkpoint\":\"{remote_checkpoint}\"}}
JSON
    ;;
  *)
    ;;
esac
"""
    )
    shim_path.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": "987654321",
            "MJEPACI_STAGE_SHIM": str(shim_path),
            "GITHUB_ENV": str(tmp_path / "github_env"),
            "WANDB_API_KEY": "",
            "PRETRAIN_EXPERIMENT_ROOT": str(phase2_root),
        }
    )

    _run(["bash", "scripts/ci/run-pretrain.sh"], env)

    experiment_dir = experiments_root / "987654321"
    manifest_path = experiment_dir / "artifacts" / "encoder_manifest.json"
    encoder_path = experiment_dir / "pretrain" / "encoder.pt"

    assert manifest_path.is_file(), manifest_path
    assert manifest_path.read_text(encoding="utf-8") == remote_manifest.read_text(encoding="utf-8")
    assert encoder_path.is_symlink(), encoder_path
    assert encoder_path.resolve() == remote_checkpoint.resolve()


def test_phase1_allocates_new_grid_and_exp_id(tmp_path):
    experiments_root = tmp_path / "experiments"
    shim_path = tmp_path / "phase1_shim.sh"
    shim_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
stage="$1"
case "$stage" in
  grid_search)
    mkdir -p "$OUT_DIR/stage-outputs"
    printf '%s' "${EXP_ID:-}" > "$OUT_DIR/stage-outputs/exp_id.txt"
    printf '%s' "${GRID_EXP_ID:-}" > "$OUT_DIR/stage-outputs/grid_exp_id.txt"
    ;;
esac
"""
    )
    shim_path.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENTS_ROOT": str(experiments_root),
            "RUN_ID": "424242",
            "MJEPACI_STAGE_SHIM": str(shim_path),
            "GRID_MODE": "custom",
            "WANDB_API_KEY": "",
            "PRETRAIN_EXP_ID": "13579",
        }
    )

    _run(["bash", "scripts/ci/run-grid-or-phase1.sh"], env)

    stage_dir = experiments_root / "424242" / "grid_search" / "stage-outputs"
    exp_value = (stage_dir / "exp_id.txt").read_text(encoding="utf-8")
    grid_value = (stage_dir / "grid_exp_id.txt").read_text(encoding="utf-8")

    assert exp_value == "424242"
    assert grid_value == "424242"


def test_artifact_collection_script_contains_mkdir_and_warnings():
    script = (REPO_ROOT / "scripts" / "ci" / "collect-pretrain-artifacts.sh").read_text(encoding="utf-8")
    assert "mkdir -p '${PRETRAIN_EXPERIMENT_ROOT}/artifacts'" in script
    assert "::warning::rsync failed" in script
    assert "missing required pretrain artifacts" in script
    assert "::warning::PRETRAIN_EXPERIMENT_ROOT not resolved" in script

    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-vast.yml").read_text(encoding="utf-8")
    assert "scripts/ci/collect-pretrain-artifacts.sh" in workflow


def test_tox21_uses_resolved_python():
    script = (REPO_ROOT / "scripts" / "ci" / "run-tox21.sh").read_text(encoding="utf-8")
    assert "resolve_ci_python python_cmd" in script
    assert "python - \"$MANIFEST_PATH\"" not in script
    assert "python - <<'PY'" not in script
    assert '"${python_cmd[@]}" - "$MANIFEST_PATH"' in script
