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

    manifest_path = experiments_root / "1759795103" / "artifacts" / "encoder_manifest.json"
    missing_path = manifest_path.parent / "missing_encoder.pt"
    manifest_path.write_text(
        json.dumps({"paths": {"encoder": str(missing_path)}}) + "\n",
        encoding="utf-8",
    )
    assert not missing_path.exists()

    _run(["bash", "scripts/ci/run-tox21.sh"], env)

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


def test_finetune_met_benchmark_flag_handles_missing_gate(tmp_path):
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    shim_path = tmp_path / "finetune_stage_shim.sh"
    shim_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
stage="$1"
if [[ "$stage" == "finetune" ]]; then
  mkdir -p "$FINETUNE_DIR/stage-outputs"
  env | sort >"$TMP_ENV_CAPTURE"
fi
""",
        encoding="utf-8",
    )
    shim_path.chmod(0o755)

    pretrain_root = experiments_root / "pretrain-demo"
    (pretrain_root / "pretrain").mkdir(parents=True)
    (pretrain_root / "artifacts").mkdir(parents=True)
    (pretrain_root / "pretrain" / "encoder.pt").write_text("stub", encoding="utf-8")
    (pretrain_root / "artifacts" / "encoder_manifest.json").write_text(
        json.dumps({"paths": {"encoder": "encoder.pt"}}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": "finetune-demo",
            "PRETRAIN_EXP_ID": "pretrain-demo",
            "MJEPACI_STAGE_SHIM": str(shim_path),
            "WANDB_API_KEY": "",
        }
    )

    capture_one = tmp_path / "env_missing.txt"
    env["TMP_ENV_CAPTURE"] = str(capture_one)
    subprocess.run(
        ["bash", "scripts/ci/run-finetune.sh"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    capture_text = capture_one.read_text(encoding="utf-8")
    assert "MET_BENCHMARK_BASELINE=unknown" in capture_text

    met_env = experiments_root / "finetune-demo" / "met_benchmark.env"
    met_env.write_text("MET_BENCHMARK_BASELINE=false\n", encoding="utf-8")

    capture_two = tmp_path / "env_failed.txt"
    env["TMP_ENV_CAPTURE"] = str(capture_two)
    subprocess.run(
        ["bash", "scripts/ci/run-finetune.sh"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    capture_text = capture_two.read_text(encoding="utf-8")
    assert "MET_BENCHMARK_BASELINE=false" in capture_text


def test_run_tox21_exports_full_finetune_when_finetuned(tmp_path):
    experiments_root = tmp_path / "experiments"
    exp_dir = experiments_root / "111111"
    pretrain_dir = exp_dir / "pretrain"
    artifacts_dir = exp_dir / "artifacts"
    finetune_dir = exp_dir / "finetune"
    tox21_dir = exp_dir / "tox21"
    for path in (pretrain_dir, artifacts_dir, finetune_dir, tox21_dir):
        path.mkdir(parents=True, exist_ok=True)

    encoder_path = pretrain_dir / "encoder.pt"
    encoder_path.write_text("stub", encoding="utf-8")
    manifest_path = artifacts_dir / "encoder_manifest.json"
    manifest_path.write_text(
        json.dumps({"paths": {"encoder": str(encoder_path)}}) + "\n",
        encoding="utf-8",
    )
    (finetune_dir / "encoder_ft.pt").write_text("ft", encoding="utf-8")

    capture_path = tmp_path / "tox21_env_capture.txt"
    shim_path = tmp_path / "tox21_shim.sh"
    shim_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
stage="$1"
if [[ "$stage" == "tox21" ]]; then
  if [[ -n "${ENV_CAPTURE:-}" ]]; then
    {
      printf 'TOX21_FULL_FINETUNE=%s\n' "${TOX21_FULL_FINETUNE:-<unset>}"
      printf 'TOX21_EVALUATION_MODE=%s\n' "${TOX21_EVALUATION_MODE:-<unset>}"
    } >"$ENV_CAPTURE"
  fi
fi
""",
        encoding="utf-8",
    )
    shim_path.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": "111111",
            "PRETRAIN_EXP_ID": "111111",
            "PRETRAIN_DIR": str(pretrain_dir),
            "PRETRAIN_ARTIFACTS_DIR": str(artifacts_dir),
            "PRETRAIN_MANIFEST": str(manifest_path),
            "PRETRAIN_TOX21_ENV": str(tmp_path / "tox21_gate.env"),
            "FINETUNE_DIR": str(finetune_dir),
            "TOX21_DIR": str(tox21_dir),
            "MJEPACI_STAGE_SHIM": str(shim_path),
            "GITHUB_ENV": str(tmp_path / "github_env"),
            "WANDB_API_KEY": "",
            "FROZEN": "0",
            "ENV_CAPTURE": str(capture_path),
        }
    )

    _run(["bash", "scripts/ci/run-tox21.sh"], env)

    payload = capture_path.read_text(encoding="utf-8").strip().splitlines()
    values = dict(line.split("=", 1) for line in payload if "=" in line)
    assert values["TOX21_EVALUATION_MODE"] == "end_to_end"
    assert values["TOX21_FULL_FINETUNE"].lower() == "true"


def test_build_stage_args_respects_full_finetune_env(tmp_path):
    best_cfg = {
        "config": {
            "gnn_type": {"value": "mpnn"},
            "hidden_dim": {"value": 64},
            "num_layers": {"value": 2},
        }
    }
    best_path = tmp_path / "best_grid_config.json"
    best_path.write_text(json.dumps(best_cfg), encoding="utf-8")

    encoder_path = tmp_path / "encoder.pt"
    encoder_path.write_text("stub", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"paths": {"encoder": str(encoder_path)}}), encoding="utf-8")
    tox21_dir = tmp_path / "tox21"
    tox21_dir.mkdir()

    base_env = os.environ.copy()
    base_env.update(
        {
            "APP_DIR": str(REPO_ROOT),
            "GRID_SOURCE_DIR": str(tmp_path),
            "GRID_DIR": str(tmp_path),
            "TOX21_DIR": str(tox21_dir),
            "TOX21_ENCODER_CHECKPOINT": str(encoder_path),
            "TOX21_ENCODER_MANIFEST": str(manifest_path),
            "PRETRAIN_MANIFEST": str(manifest_path),
            "PRETRAIN_DIR": str(tmp_path / "pretrain"),
            "PRETRAIN_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "FINETUNE_DIR": str(tmp_path / "finetune"),
            "PRETRAIN_EPOCHS": "5",
            "FINETUNE_EPOCHS": "5",
        }
    )

    def _invoke(flag, output_name):
        env = base_env.copy()
        output_path = tmp_path / output_name
        env["ARGS_CAPTURE"] = str(output_path)
        if flag is None:
            env.pop("TOX21_FULL_FINETUNE", None)
        else:
            env["TOX21_FULL_FINETUNE"] = flag
        script = f"""
set -euo pipefail
source \"{REPO_ROOT}/scripts/ci/common.sh\"
source \"{REPO_ROOT}/scripts/ci/stage.sh\"
build_stage_args tox21
printf '%s\\n' \"${{STAGE_ARGS[@]}}\" > \"$ARGS_CAPTURE\"
"""
        subprocess.run(["bash", "-lc", script], check=True, cwd=REPO_ROOT, env=env)
        return (output_path.read_text(encoding="utf-8").strip().splitlines())

    args_true = _invoke("true", "args_true.txt")
    assert "--full-finetune" in args_true
    assert "--no-full-finetune" not in args_true

    args_false = _invoke("false", "args_false.txt")
    assert "--no-full-finetune" in args_false

    args_unset = _invoke(None, "args_unset.txt")
    assert "--full-finetune" not in args_unset
    assert "--no-full-finetune" not in args_unset
