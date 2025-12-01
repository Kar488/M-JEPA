import json
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_stub(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_collect_phase2_artifacts_copies_all_steps(tmp_path):
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(parents=True, exist_ok=True)

    ssh_stub = stub_dir / "ssh"
    _write_stub(
        ssh_stub,
        """#!/usr/bin/env bash
set -euo pipefail
if [ -t 0 ]; then
  payload=""
else
  payload="$(cat)"
fi
args=()
while (($#)); do
  case "$1" in
    -i|-p|-o)
      shift 2
      continue
      ;;
  esac
  args+=("$1")
  shift

done

if ((${#args[@]} == 0)); then
  exit 0
fi

cmd=("${args[@]:1}")
if ((${#cmd[@]} == 0)); then
  exit 0
fi

    if [[ "${cmd[0]}" == *"bash" ]]; then
      "${cmd[@]}" <<<"$payload"
    else
      bash -c "${cmd[*]}" <<<"$payload"
    fi
""",
    )

    rsync_stub = stub_dir / "rsync"
    _write_stub(
        rsync_stub,
        """#!/usr/bin/env bash
set -euo pipefail
args=("$@")
if ((${#args[@]} < 2)); then
  exit 1
fi

dest="${args[-1]}"
src="${args[-2]}"
src="${src#*:}"

mkdir -p "$dest"

if [[ -d "$src" ]]; then
  cp -a "$src"/. "$dest/"
  exit 0
fi

if [[ -f "$src" ]]; then
  cp -a "$src" "$dest/"
  exit 0
fi

exit 1
""",
    )

    remote_root = tmp_path / "remote"
    experiments_root = remote_root / "experiments"
    grid_dir = experiments_root / "123" / "grid"
    for step in ("phase2_sweep", "phase2_recheck", "phase2_export"):
        logs = grid_dir / step / "logs"
        outputs = grid_dir / step / "stage-outputs"
        logs.mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        (logs / f"{step}.log").write_text(step, encoding="utf-8")
        (outputs / f"{step}.json").write_text(step, encoding="utf-8")

    metadata = {
        "config": {"hidden_dim": 64},
        "summary": {"metric": 0.1},
    }
    (grid_dir / "phase2_sweep_id.txt").write_text("sweep-abc", encoding="utf-8")
    (grid_dir / "best_grid_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    (grid_dir / "recheck_summary.json").write_text(json.dumps(metadata), encoding="utf-8")
    (grid_dir / "grid_state.json").write_text("state", encoding="utf-8")
    (grid_dir / "phase2_winner.txt").write_text("winner", encoding="utf-8")

    dest_root = tmp_path / "collected"

    env = os.environ.copy()
    env.update(
        {
            "SSH_KEY": "dummy",
            "VAST_USER": "user",
            "VAST_HOST": "host",
            "VAST_PORT": "22",
            "EXP_ID": "123",
            "PRETRAIN_EXP_ID": "123",
            "EXPERIMENTS_ROOT": str(experiments_root),
            "GRID_EXP_ID": "123",
            "GRID_DIR": str(grid_dir),
            "GRID_SOURCE_DIR": str(grid_dir),
            "PATH": f"{stub_dir}:{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path / "runner"),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/ci/collect-phase2-artifacts.sh", str(dest_root)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr

    lineage_root = dest_root / "lineage"
    current_root = dest_root / "current"
    for collected_root in (lineage_root, current_root):
        assert (collected_root / "grid" / "best_grid_config.json").is_file()
        assert (collected_root / "grid" / "recheck_summary.json").is_file()
        for step in ("phase2_sweep", "phase2_recheck", "phase2_export"):
            log_dir = collected_root / step / "logs"
            outputs_dir = collected_root / step / "stage-outputs"
            assert log_dir.is_dir()
            assert outputs_dir.is_dir()


def test_phase2_export_stage_emits_metadata(tmp_path):
    grid_dir = tmp_path / "grid"
    (grid_dir / "phase2_recheck").mkdir(parents=True, exist_ok=True)
    (grid_dir / "phase2_export").mkdir(parents=True, exist_ok=True)

    metadata = {"config": {"hidden_dim": 128}, "summary": {"metric": 0.2}}
    best_path = grid_dir / "best_grid_config.json"
    summary_path = grid_dir / "recheck_summary.json"
    best_path.write_text(json.dumps(metadata), encoding="utf-8")
    summary_path.write_text(json.dumps(metadata), encoding="utf-8")

    sentinel = grid_dir / "phase2_recheck" / "recheck_done.ok"
    sentinel.write_text("done", encoding="utf-8")

    helper = grid_dir / "phase2_winner.txt"
    helper.write_text("winner", encoding="utf-8")

    (grid_dir / "phase2_sweep_id.txt").write_text("sweep-abc", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "APP_DIR": str(REPO_ROOT),
            "GRID_DIR": str(grid_dir),
            "GRID_SOURCE_DIR": str(grid_dir),
            "EXP_ID": "123",
            "PRETRAIN_EXP_ID": "123",
        }
    )

    cmd = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(REPO_ROOT / 'scripts/ci/common.sh'))}; "
        f"source {shlex.quote(str(REPO_ROOT / 'scripts/ci/stage.sh'))}; "
        f"run_phase2_export_stage {shlex.quote(str(grid_dir / 'phase2_export'))} phase2_export"
    )

    proc = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr

    outputs_dir = grid_dir / "phase2_export" / "stage-outputs"
    assert (outputs_dir / "best_grid_config.json").is_file()
    assert (outputs_dir / "recheck_summary.json").is_file()
    helpers_dir = outputs_dir / "helpers"
    assert (helpers_dir / "phase2_winner.txt").is_file()

    metadata_path = outputs_dir / "phase2_export.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["best_config_path"] == str(best_path)
    assert payload["summary_path"] == str(summary_path)
    assert any(path.endswith("phase2_winner.txt") for path in payload["helper_files"])
