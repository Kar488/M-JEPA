import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import pytest
import re

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd, env):
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _write_train_stub(script_path: Path) -> None:
    script_path.write_text(
        """
import os
import sys
from pathlib import Path


def _extract_arg(tokens, name):
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return "missing"


def main() -> int:
    args = sys.argv[1:]
    devices = _extract_arg(args, "--devices")
    device = _extract_arg(args, "--device")
    bf16 = _extract_arg(args, "--bf16")

    count_path = Path(os.environ["TRAIN_INVOCATION_FILE"])
    try:
        count = int(count_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        count = 0
    count_path.write_text(str(count + 1), encoding="utf-8")

    payload = f"devices={devices} device={device} bf16={bf16}"
    payload_path = Path(os.environ["TRAIN_LOG_FILE"])
    payload_path.write_text(payload, encoding="utf-8")
    print(f"train invoked {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_pretrain_tox21_dry_run(tmp_path):
    experiments_root = tmp_path / "experiments"
    dataset_csv = tmp_path / "toy_dataset.csv"
    dataset_csv.write_text("smiles\nC1=CC=CC=C1\n", encoding="utf-8")
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
            "DATASET_DIR": str(dataset_csv),
        }
    )

    _run(["bash", "scripts/ci/run-pretrain.sh"], env)

    graphs_dir = experiments_root / "1759795103" / "graphs"
    summary_path = graphs_dir / "summary.json"
    assert graphs_dir.is_dir(), graphs_dir
    assert summary_path.is_file(), summary_path
    assert list(graphs_dir.rglob("*.html")), "expected HTML graph visuals"
    assert list(graphs_dir.rglob("*.png")), "expected PNG graph visuals"

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


def test_common_sh_falls_back_when_mamba_root_unwritable(tmp_path):
    common_sh = REPO_ROOT / "scripts" / "ci" / "common.sh"

    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)

    runner_tmp = tmp_path / "runner"
    runner_tmp.mkdir(exist_ok=True)

    blocked_path = tmp_path / "blocked"
    blocked_path.write_text("stub", encoding="utf-8")

    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "DATA_ROOT": str(data_root),
            "RUNNER_TEMP": str(runner_tmp),
            "HOME": str(fake_home),
            "MAMBA_ROOT_PREFIX": str(blocked_path),
            # These fallback behaviors are the point of the test, so make
            # sure global CI settings that disable fallbacks do not bleed
            # into the test environment.
            "MJEPA_ALLOW_DATA_FALLBACKS": "1",
        }
    )

    cmd = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(common_sh))}; "
        "printf '%s' \"$MAMBA_ROOT_PREFIX\""
    )

    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    expected = fake_home / "micromamba"
    assert proc.stdout.strip() == str(expected)


def test_common_sh_rewrites_unwritable_sweep_cache(tmp_path):
    common_sh = REPO_ROOT / "scripts" / "ci" / "common.sh"

    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)

    runner_tmp = tmp_path / "runner"
    runner_tmp.mkdir(exist_ok=True)

    blocked_sweep = tmp_path / "blocked_sweep"
    blocked_sweep.write_text("stub", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "DATA_ROOT": str(data_root),
            "RUNNER_TEMP": str(runner_tmp),
            "CACHE_DIR": str(data_root / "cache" / "graphs_10m"),
            "SWEEP_CACHE_DIR": str(blocked_sweep),
            "MJEPA_ALLOW_DATA_FALLBACKS": "1",
        }
    )

    cmd = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(common_sh))}; "
        "printf '%s\n%s' \"$CACHE_DIR\" \"$SWEEP_CACHE_DIR\""
    )

    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    cache_dir_str, sweep_dir_str = proc.stdout.strip().splitlines()
    assert cache_dir_str == sweep_dir_str

    cache_dir = Path(cache_dir_str)
    assert cache_dir_str != str(blocked_sweep)
    assert cache_dir.name == "graphs_10m"
    assert cache_dir.is_dir(), cache_dir


def test_common_sh_rewrites_legacy_cache_root(tmp_path):
    common_sh = REPO_ROOT / "scripts" / "ci" / "common.sh"

    data_root = tmp_path / "data"
    legacy_cache = data_root / "cache" / "graphs_250k"
    preferred_cache = data_root / "cache" / "graphs_10m"

    preferred_cache.mkdir(parents=True, exist_ok=True)
    legacy_cache.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "DATA_ROOT": str(data_root),
            "CACHE_DIR": str(legacy_cache),
            "SWEEP_CACHE_DIR": str(legacy_cache),
            "MJEPA_ALLOW_DATA_FALLBACKS": "1",
        }
    )

    cmd = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(common_sh))}; "
        "printf '%s\n%s' \"$CACHE_DIR\" \"$SWEEP_CACHE_DIR\""
    )

    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    cache_dir_str, sweep_dir_str = proc.stdout.strip().splitlines()
    assert cache_dir_str == sweep_dir_str

    cache_dir = Path(cache_dir_str)
    assert cache_dir.name == "graphs_10m"
    assert cache_dir.is_dir(), cache_dir


def test_common_sh_attempts_privileged_fix_for_experiments_root(tmp_path):
    common_sh = REPO_ROOT / "scripts" / "ci" / "common.sh"
    fake_sudo = REPO_ROOT / "tests" / "ci" / "fake_sudo.sh"

    runner_tmp = tmp_path / "runner"
    runner_tmp.mkdir()

    blocked_parent = tmp_path / "blocked_parent"
    blocked_parent.mkdir()
    blocked_parent.chmod(0o555)

    target_root = blocked_parent / "experiments"

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENTS_ROOT": str(target_root),
            "RUNNER_TEMP": str(runner_tmp),
            "MJEPA_SUDO_BIN": str(fake_sudo),
            "MJEPA_FAKE_SUDO_FIX_PATH": str(blocked_parent),
        }
    )

    cmd = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(common_sh))}; "
        "printf '%s\n%s' \"$EXPERIMENTS_ROOT\" \"$DATA_ROOT\""
    )

    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    resolved_experiments, resolved_data = proc.stdout.strip().splitlines()
    assert resolved_experiments == str(target_root)
    assert resolved_data == str(blocked_parent)
    assert target_root.is_dir()
    assert os.access(target_root, os.W_OK)


def test_common_sh_handles_privileged_fix_when_tty_required(tmp_path):
    if shutil.which("script") is None:
        pytest.skip("script command unavailable")

    common_sh = REPO_ROOT / "scripts" / "ci" / "common.sh"
    fake_sudo = REPO_ROOT / "tests" / "ci" / "fake_sudo.sh"

    runner_tmp = tmp_path / "runner"
    runner_tmp.mkdir()

    blocked_parent = tmp_path / "blocked_parent"
    blocked_parent.mkdir()
    blocked_parent.chmod(0o555)

    target_root = blocked_parent / "experiments"

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENTS_ROOT": str(target_root),
            "RUNNER_TEMP": str(runner_tmp),
            "MJEPA_SUDO_BIN": str(fake_sudo),
            "MJEPA_FAKE_SUDO_FIX_PATH": str(blocked_parent),
            "MJEPA_FAKE_SUDO_REQUIRE_TTY": "1",
        }
    )

    cmd = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(common_sh))}; "
        "printf '%s\n%s' \"$EXPERIMENTS_ROOT\" \"$DATA_ROOT\""
    )

    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    resolved_experiments, resolved_data = proc.stdout.strip().splitlines()
    assert resolved_experiments == str(target_root)
    assert resolved_data == str(blocked_parent)
    assert target_root.is_dir()
    assert os.access(target_root, os.W_OK)


def test_common_sh_aborts_when_experiments_fallback_disabled(tmp_path):
    common_sh = REPO_ROOT / "scripts" / "ci" / "common.sh"
    runner_tmp = tmp_path / "runner"
    runner_tmp.mkdir()

    blocked_parent = tmp_path / "blocked_parent"
    blocked_parent.mkdir()
    target_root = blocked_parent / "experiments"
    target_root.write_text("", encoding="utf-8")
    blocked_parent.chmod(0o555)
    fake_sudo = shutil.which("false") or "/bin/false"

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENTS_ROOT": str(target_root),
            "RUNNER_TEMP": str(runner_tmp),
            "MJEPA_ALLOW_DATA_FALLBACKS": "0",
            "MJEPA_SUDO_BIN": fake_sudo,
            "MJEPA_SUDO_ALLOW_TTY_WRAPPER": "0",
        }
    )

    cmd = f"set -euo pipefail; source {shlex.quote(str(common_sh))}"
    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "fallbacks disabled" in proc.stderr.lower()


def test_common_sh_aborts_when_cache_dir_fallback_disabled(tmp_path):
    common_sh = REPO_ROOT / "scripts" / "ci" / "common.sh"
    runner_tmp = tmp_path / "runner"
    runner_tmp.mkdir()

    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)

    blocked_parent = tmp_path / "blocked_cache"
    blocked_parent.mkdir(exist_ok=True)
    blocked_cache = blocked_parent / "cache"
    blocked_cache.write_text("", encoding="utf-8")
    blocked_parent.chmod(0o555)
    fake_sudo = shutil.which("false") or "/bin/false"

    env = os.environ.copy()
    env.update(
        {
            "DATA_ROOT": str(data_root),
            "CACHE_DIR": str(blocked_cache),
            "RUNNER_TEMP": str(runner_tmp),
            "MJEPA_ALLOW_DATA_FALLBACKS": "0",
            "MJEPA_SUDO_BIN": fake_sudo,
            "MJEPA_SUDO_ALLOW_TTY_WRAPPER": "0",
        }
    )

    cmd = f"set -euo pipefail; source {shlex.quote(str(common_sh))}"
    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "fallbacks disabled" in proc.stderr.lower()


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


def test_tox21_ddp_fallback(tmp_path):
    fake_site = tmp_path / "fake_site"
    (fake_site / "torch" / "distributed").mkdir(parents=True)

    (fake_site / "torch" / "__init__.py").write_text(
        """
class _Cuda:
    def is_available(self):
        return True

    def device_count(self):
        return 2


cuda = _Cuda()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (fake_site / "torch" / "distributed" / "__init__.py").write_text("", encoding="utf-8")
    (fake_site / "torch" / "distributed" / "run.py").write_text(
        """
import os
import sys
from pathlib import Path


def main() -> int:
    attempts_path = os.environ.get("DDP_ATTEMPTS_FILE")
    if attempts_path:
        path = Path(attempts_path)
        try:
            count = int(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            count = 0
        path.write_text(str(count + 1), encoding="utf-8")
    exit_code = int(os.environ.get("DDP_SIMULATED_EXIT", "5"))
    print("simulated torch.distributed.run failure", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
""".strip()
        + "\n",
        encoding="utf-8",
    )

    app_dir = tmp_path / "app"
    (app_dir / "scripts").mkdir(parents=True)
    train_invocations = tmp_path / "train_invocations.txt"
    train_args_log = tmp_path / "train_args.txt"
    _write_train_stub(app_dir / "scripts" / "train_jepa.py")

    log_dir = tmp_path / "logs"
    ddp_attempts = tmp_path / "ddp_attempts.txt"

    env = os.environ.copy()
    base_pythonpath = env.get("PYTHONPATH", "")
    combined_pythonpath = f"{fake_site}:{base_pythonpath}" if base_pythonpath else str(fake_site)
    env.update(
        {
            "APP_DIR": str(app_dir),
            "LOG_DIR": str(log_dir),
            "MJEPACI_FORCE_SYSTEM_PYTHON": "1",
            "MJEPACI_SYSTEM_PYTHON_BIN": sys.executable,
            "PYTHONPATH": combined_pythonpath,
            "DDP_ATTEMPTS_FILE": str(ddp_attempts),
            "DDP_SIMULATED_EXIT": "7",
            "TRAIN_LOG_FILE": str(train_args_log),
            "TRAIN_INVOCATION_FILE": str(train_invocations),
        }
    )

    script = "\n".join(
        [
            "set -euo pipefail",
            "source scripts/ci/common.sh",
            "source scripts/ci/stage.sh",
            "mkdir -p \"${LOG_DIR}\"",
            "STAGE_ARGS=(--devices 2 --epochs 1)",
            "set +e",
            "run_with_timeout tox21 STAGE_ARGS",
            "rc=$?",
            "set -e",
            "exit $rc",
        ]
    )

    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert ddp_attempts.read_text(encoding="utf-8") == "1"
    assert train_invocations.read_text(encoding="utf-8") == "1"
    payload = train_args_log.read_text(encoding="utf-8")
    assert "devices=1" in payload
    assert "device=missing" in payload
    assert "bf16=missing" in payload
    assert (log_dir / "tox21.log").is_file()
    log_contents = (log_dir / "tox21.log").read_text(encoding="utf-8")
    assert "train invoked devices=1" in log_contents
    assert "[stage:tox21] warn: distributed launch failed" in proc.stderr


def test_tox21_cpu_fallback_when_cuda_missing(tmp_path):
    fake_site = tmp_path / "fake_site_cpu"
    (fake_site / "torch" / "distributed").mkdir(parents=True)

    (fake_site / "torch" / "__init__.py").write_text(
        """
class _Cuda:
    def is_available(self):
        return False

    def device_count(self):
        return 0


cuda = _Cuda()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (fake_site / "torch" / "distributed" / "__init__.py").write_text("", encoding="utf-8")
    (fake_site / "torch" / "distributed" / "run.py").write_text(
        """
def main():
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".strip()
        + "\n",
        encoding="utf-8",
    )

    app_dir = tmp_path / "app_cpu"
    (app_dir / "scripts").mkdir(parents=True)
    train_invocations = tmp_path / "train_invocations_cpu.txt"
    train_args_log = tmp_path / "train_args_cpu.txt"
    _write_train_stub(app_dir / "scripts" / "train_jepa.py")

    log_dir = tmp_path / "logs_cpu"

    env = os.environ.copy()
    base_pythonpath = env.get("PYTHONPATH", "")
    combined_pythonpath = f"{fake_site}:{base_pythonpath}" if base_pythonpath else str(fake_site)
    env.update(
        {
            "APP_DIR": str(app_dir),
            "LOG_DIR": str(log_dir),
            "MJEPACI_FORCE_SYSTEM_PYTHON": "1",
            "MJEPACI_SYSTEM_PYTHON_BIN": sys.executable,
            "PYTHONPATH": combined_pythonpath,
            "TRAIN_LOG_FILE": str(train_args_log),
            "TRAIN_INVOCATION_FILE": str(train_invocations),
        }
    )

    script = "\n".join(
        [
            "set -euo pipefail",
            "source scripts/ci/common.sh",
            "source scripts/ci/stage.sh",
            "mkdir -p \"${LOG_DIR}\"",
            "STAGE_ARGS=(--device cuda --devices 2 --bf16 1 --epochs 1)",
            "set +e",
            "run_with_timeout tox21 STAGE_ARGS",
            "rc=$?",
            "set -e",
            "exit $rc",
        ]
    )

    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = train_args_log.read_text(encoding="utf-8")
    assert "devices=1" in payload
    assert "device=cpu" in payload
    assert "bf16=0" in payload
    assert train_invocations.read_text(encoding="utf-8") == "1"
    assert (log_dir / "tox21.log").is_file()
    log_contents = (log_dir / "tox21.log").read_text(encoding="utf-8")
    assert "train invoked devices=1 device=cpu bf16=0" in log_contents
    assert "[stage:tox21] warn: CUDA unavailable" in proc.stderr

def test_dedupe_stage_args_handles_joined_aliases():
    common_sh = REPO_ROOT / "scripts" / "ci" / "common.sh"
    stage_sh = REPO_ROOT / "scripts" / "ci" / "stage.sh"
    cmd = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(common_sh))}; "
        f"source {shlex.quote(str(stage_sh))}; "
        "arr=(--pin-memory 1 --pin_memory=0 --pin-memory 0 --pin-memory=1 "
        "--num-workers 4 --num_workers=8 keep); "
        "dedupe_stage_args arr; "
        "printf '%s\\n' \"${arr[@]}\""
    )
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    )
    tokens = [line for line in proc.stdout.splitlines() if line]
    assert tokens == ["--pin-memory=1", "--num_workers=8", "keep"]


def test_tox21_cli_command_has_unique_flags(tmp_path):
    experiments_root = tmp_path / "experiments"
    exp_id = "314159"
    pretrain_root = experiments_root / exp_id
    pretrain_dir = pretrain_root / "pretrain"
    pretrain_artifacts = pretrain_root / "artifacts"
    tox21_dir = pretrain_root / "tox21"
    grid_dir = pretrain_root / "grid"

    for directory in (pretrain_dir, pretrain_artifacts, tox21_dir, grid_dir / "phase2_export"):
        directory.mkdir(parents=True, exist_ok=True)

    encoder_path = pretrain_dir / "encoder.pt"
    encoder_path.write_text("stub", encoding="utf-8")

    manifest_path = pretrain_artifacts / "encoder_manifest.json"
    manifest_payload = {"paths": {"encoder": str(encoder_path)}}
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    state_path = pretrain_root / "pretrain_state.json"
    state_payload = {"id": exp_id, "encoder_manifest": str(manifest_path)}
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    best_cfg = {
        "prefetch_factor": 2,
        "persistent_workers": 0,
        "pin_memory": 0,
        "bf16": 0,
        "devices": 3,
        "num_workers": 6,
        "pretrain_batch_size": 128,
    }
    best_paths = [grid_dir / "best_grid_config.json", grid_dir / "phase2_export" / "best_grid_config.json"]
    for cfg_path in best_paths:
        cfg_path.write_text(json.dumps(best_cfg), encoding="utf-8")

    micromamba_stub = tmp_path / "micromamba"
    micromamba_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "if [[ \"${1:-}\" == \"run\" ]]; then\n"
        "  shift\n"
        "  if [[ \"${1:-}\" == \"-n\" ]]; then\n"
        "    shift 2\n"
        "  fi\n"
        "  exec \"$@\"\n"
        "elif [[ \"${1:-}\" == \"shell\" && \"${2:-}\" == \"hook\" ]]; then\n"
        "  exit 0\n"
        "else\n"
        "  echo \"micromamba stub unsupported: $*\" >&2\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    micromamba_stub.chmod(0o755)

    stage_stub = tmp_path / "stage_stub.sh"
    stage_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"${TOX21_DIR}/stage-outputs\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stage_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "APP_DIR": str(REPO_ROOT),
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": exp_id,
            "PRETRAIN_EXP_ID": exp_id,
            "PRETRAIN_EXPERIMENT_ROOT": str(pretrain_root),
            "PRETRAIN_DIR": str(pretrain_dir),
            "PRETRAIN_ARTIFACTS_DIR": str(pretrain_artifacts),
            "PRETRAIN_MANIFEST": str(manifest_path),
            "PRETRAIN_STATE_FILE": str(state_path),
            "PRETRAIN_STATE_FILE_CANONICAL": str(state_path),
            "PRETRAIN_TOX21_ENV": str(pretrain_root / "tox21_gate.env"),
            "TOX21_DIR": str(tox21_dir),
            "GITHUB_ENV": str(pretrain_root / "tox21_gate.env"),
            "WANDB_API_KEY": "",
            "STAGE_BIN": str(stage_stub),
            "MMBIN": str(micromamba_stub),
            "MAMBA_ROOT_PREFIX": str(tmp_path / "micromamba_root"),
            "GRID_DIR": str(grid_dir),
            "GRID_SOURCE_DIR": str(grid_dir),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/ci/run-tox21.sh"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr

    command_line = None
    for line in proc.stderr.splitlines():
        if "[diag] python_cmd=" in line and "scripts.commands.tox21" in line:
            command_line = line.split("python_cmd=", 1)[1]
    assert command_line, proc.stderr

    tokens = shlex.split(command_line)
    assert "-m" in tokens and "scripts.commands.tox21" in tokens
    tox_index = tokens.index("scripts.commands.tox21")
    tox_args = tokens[tox_index + 1 :]

    flag_counts: dict[str, int] = {}
    resolved_values: dict[str, str] = {}
    i = 0
    while i < len(tox_args):
        token = tox_args[i]
        if token.startswith("--"):
            flag = token
            value = None
            consumed = 1
            if "=" in flag:
                flag, value = flag.split("=", 1)
            elif i + 1 < len(tox_args) and not tox_args[i + 1].startswith("--"):
                value = tox_args[i + 1]
                consumed = 2
            canonical = flag.replace("_", "-")
            flag_counts[canonical] = flag_counts.get(canonical, 0) + 1
            if value is not None:
                resolved_values[canonical] = value
            i += consumed
            continue
        i += 1

    for canonical in ("--num-workers", "--prefetch-factor", "--persistent-workers", "--pin-memory", "--bf16", "--devices", "--batch-size"):
        assert flag_counts.get(canonical) == 1, f"duplicate flag {canonical}: {flag_counts}"

    forbidden = "--finetune-batch-size"
    assert forbidden not in {flag.split("=", 1)[0] for flag in tox_args if flag.startswith("--")}

    assert resolved_values.get("--num-workers") == "6"
    assert resolved_values.get("--prefetch-factor") == "2"
    assert resolved_values.get("--persistent-workers") == "0"
    assert resolved_values.get("--pin-memory") == "0"
    assert resolved_values.get("--bf16") == "0"
    assert resolved_values.get("--devices") == "3"
    assert resolved_values.get("--batch-size") == "128"


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
            "MJEPA_ALLOW_DATA_FALLBACKS": "1",
        }
    )

    # Strip host-provided cache overrides so run-finetune derives paths from the
    # temporary experiments root created by the test.  Some CI environments set
    # these variables globally (e.g., to /data/mjepa/cache/*), which causes the
    # stage wrapper to look for encoder artifacts outside of the test fixture
    # and fail immediately.
    for key in (
        "CACHE_DIR",
        "SWEEP_CACHE_DIR",
        "FINETUNE_CACHE_DIR",
        "BENCH_CACHE_DIR",
        "TOX21_CACHE_DIR",
        "REPORTS_CACHE_DIR",
        "GRID_CACHE_DIR",
        "PRETRAIN_CACHE_DIR",
    ):
        env.pop(key, None)

    met_env = experiments_root / env["EXP_ID"] / "met_benchmark.env"
    
    for key in (
        "PRETRAIN_ENCODER_PATH",
        "PRETRAIN_MANIFEST",
        "PRETRAIN_TOX21_ENV",
        "MET_BENCHMARK_BASELINE",
    ):
        env.pop(key, None)

    def invoke_finetune(env_map, capture_path, *, label, extra_debug_keys=None):
        env_for_run = env_map.copy()
        env_for_run["TMP_ENV_CAPTURE"] = str(capture_path)

        diag_stream = sys.stderr

        def diag(message: str) -> None:
            print(message, flush=True)

        debug_filter = {
            "ARTIFACTS_DIR",
            "EXP_ID",
            "EXP_ROOT",
            "EXPERIMENT_DIR",
            "EXPERIMENTS_ROOT",
            "FINETUNE_DIR",
            "MJEPACI_DEBUG",
            "MJEPACI_STAGE_SHIM",
            "TMP_ENV_CAPTURE",
        }
        debug_keys = {
            key
            for key in env_for_run
            if key.startswith("PRETRAIN")
            or key.startswith("FINETUNE")
            or key in debug_filter
        }
        if extra_debug_keys:
            debug_keys.update(key for key in extra_debug_keys if key in env_for_run)

        diag(
            f"[finetune-test] invoking run-finetune for {label} (capture={capture_path})"
        )
        for key in sorted(debug_keys):
            diag(f"[finetune-test]   {key}={env_for_run[key]}")

        required_env_keys = {
            "PRETRAIN_DIR",
            "PRETRAIN_ARTIFACTS_DIR",
            "PRETRAIN_MANIFEST",
            "PRETRAIN_ENCODER_PATH",
            "PRETRAIN_EXPERIMENT_ROOT",
            "PRETRAIN_TOX21_ENV",
            "FINETUNE_DIR",
        }
        missing_env = sorted(key for key in required_env_keys if key not in env_for_run)
        if missing_env:
            diag(
                "[finetune-test]   missing_env_keys="
                + ", ".join(missing_env)
            )

        try:
             result = subprocess.run(
                ["bash", "scripts/ci/run-finetune.sh"],
                check=True,
                cwd=REPO_ROOT,
                env=env_for_run,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            diag(f"[finetune-test] run-finetune failed for {label} (rc={exc.returncode})")
            # print stderr from the failed subprocess, filtered for numeric markers
            for line in exc.stdout.splitlines():
                if re.fullmatch(r'\d+(?:\.\d+)*', line.strip()):
                    diag(f"[finetune-test] out: {line}")
            for line in exc.stderr.splitlines():
                if re.fullmatch(r'\d+(?:\.\d+)*', line.strip()):
                    diag(f"[finetune-test] err: {line}")
            #traceback.print_exc(file=diag_stream)
            raise
        except Exception:
            diag(f"[finetune-test] unexpected exception while invoking {label}")
            #traceback.print_exc(file=diag_stream)
            diag_stream.flush()
            if capture_path.exists():
                capture_text = capture_path.read_text(encoding="utf-8")
                diag(
                    f"[finetune-test] capture for {label} (unexpected failure):\n{capture_text}"
                )
            raise
        
        
        # On success, show only the debug markers from stdout/stderr
        for line in result.stdout.splitlines():
            if re.fullmatch(r'\d+(?:\.\d+)*', line.strip()):
                diag(f"[finetune-test] out: {line}")
        for line in result.stderr.splitlines():
            if re.fullmatch(r'\d+(?:\.\d+)*', line.strip()):
                diag(f"[finetune-test] err: {line}")

        if capture_path.exists():
            return capture_path.read_text(encoding="utf-8")
        return ""

    capture_one = tmp_path / "env_missing.txt"
    capture_text = invoke_finetune(env, capture_one, label="local gate missing")
    assert "MET_BENCHMARK_BASELINE=unknown" in capture_text

    # Uppercase/whitespace variants of the reroute flag should still be
    # parsed correctly when they evaluate to "false".
    met_env.write_text(
        "  export MET_BENCHMARK_BASELINE = FALSE  \r\n",
        encoding="utf-8",
    )

    capture_two = tmp_path / "env_failed.txt"
    capture_text = invoke_finetune(env, capture_two, label="local gate present")
    assert "MET_BENCHMARK_BASELINE=false" in capture_text

    # Remove the finetune-local gate and ensure the pretrain fallback is
    # honoured when the reroute signal only exists under the lineage root.
    met_env.unlink(missing_ok=True)
    pretrain_gate = experiments_root / "pretrain-demo" / "met_benchmark.env"
    pretrain_gate.write_text(
        "  # gate summary\r\n"
        "export MET_BENCHMARK_BASELINE=false\r\n"
        "MET_GATE_DEBUG=observed value  \r\n",
        encoding="utf-8",
    )

    capture_three = tmp_path / "env_fallback.txt"
    capture_text = invoke_finetune(env, capture_three, label="pretrain gate fallback")
    assert "MET_BENCHMARK_BASELINE=false" in capture_text
    assert "MET_GATE_DEBUG=observed value" in capture_text

    # Environments that inject PRETRAIN_DIR/PRETRAIN_ARTIFACTS_DIR without
    # declaring PRETRAIN_EXP_ID should still locate the lineage gate.
    env_direct = env.copy()
    env_direct.update(
        {
            "PRETRAIN_EXP_ID": "mismatched-pretrain-id",
            "PRETRAIN_DIR": str(pretrain_root / "pretrain"),
            "PRETRAIN_ARTIFACTS_DIR": str(pretrain_root / "artifacts"),
            "ARTIFACTS_DIR": str(pretrain_root / "artifacts"),
            "PRETRAIN_MANIFEST": str(pretrain_root / "artifacts" / "encoder_manifest.json"),
            "PRETRAIN_ENCODER_PATH": str(pretrain_root / "pretrain" / "encoder.pt"),
        }
    )

    capture_direct = tmp_path / "env_direct.txt"
    capture_text = invoke_finetune(
        env_direct,
        capture_direct,
        label="pretrain env_direct reroute",
        extra_debug_keys={"ARTIFACTS_DIR"},
    )
    assert "MET_BENCHMARK_BASELINE=false" in capture_text
    assert "MET_GATE_DEBUG=observed value" in capture_text

    # Force the resolver to rely solely on the manifest entry when the
    # advertised PRETRAIN_ENCODER_PATH is missing so relative manifest entries
    # are interpreted relative to the pretrain lineage.
    env_manifest_only = env.copy()
    env_manifest_only["PRETRAIN_ENCODER_PATH"] = str(
        experiments_root / "finetune-demo" / "artifacts" / "missing_encoder.pt"
    )
    capture_manifest = tmp_path / "env_manifest_relative.txt"
    capture_text = invoke_finetune(
        env_manifest_only,
        capture_manifest,
        label="manifest relative encoder",
    )
    assert "MET_BENCHMARK_BASELINE=false" in capture_text

    # Empty or whitespace-only gate files should act like a missing
    # reroute signal and leave the baseline flag marked as unknown.
    pretrain_gate.write_text("\n  \t  # comment only\r\n\t\n", encoding="utf-8")

    capture_blank = tmp_path / "env_blank.txt"
    capture_text = invoke_finetune(env, capture_blank, label="blank pretrain gate")
    assert "MET_BENCHMARK_BASELINE=unknown" in capture_text

    # Uppercase/whitespace variants of the baseline gate should still
    # short-circuit the stage before the shim executes.
    met_env.write_text(
        "  export MET_BENCHMARK_BASELINE = TRUE  \r\n",
        encoding="utf-8",
    )

    capture_skip = tmp_path / "env_skip.txt"
    if capture_skip.exists():
        capture_skip.unlink()
    invoke_finetune(env, capture_skip, label="uppercase gate entry")
    assert not capture_skip.exists()


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


def test_tox21_cached_finetune_artifacts_are_discovered(tmp_path):
    experiments_root = tmp_path / "experiments"
    exp_id = "19678842966"
    pretrain_root = experiments_root / exp_id
    pretrain_dir = pretrain_root / "pretrain"
    pretrain_artifacts = pretrain_root / "artifacts"
    finetune_dir = pretrain_root / "finetune"
    tox21_dir = pretrain_root / "tox21"

    for path in (pretrain_dir, pretrain_artifacts, finetune_dir, tox21_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = pretrain_artifacts / "encoder_manifest.json"
    manifest_path.write_text(json.dumps({"paths": {"encoder": "missing"}}), encoding="utf-8")
    state_path = pretrain_root / "pretrain_state.json"
    state_path.write_text("{}", encoding="utf-8")

    data_root = tmp_path / "data"
    cache_root = data_root / "cache" / "finetune"
    encoder_path = cache_root / "encoder_ft.pt"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.write_text("cached", encoding="utf-8")

    stage_outputs = cache_root / "stage-outputs"
    stage_outputs.mkdir(parents=True, exist_ok=True)
    finetune_json = stage_outputs / "finetune.json"
    finetune_json.write_text(
        json.dumps({"encoder_finetuned": {"checkpoint": str(encoder_path)}}),
        encoding="utf-8",
    )

    micromamba_stub = tmp_path / "micromamba"
    micromamba_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "if [[ \"${1:-}\" == \"run\" ]]; then\n"
        "  shift\n"
        "  if [[ \"${1:-}\" == \"-n\" ]]; then\n"
        "    shift 2\n"
        "  fi\n"
        "  exec \"$@\"\n"
        "elif [[ \"${1:-}\" == \"shell\" && \"${2:-}\" == \"hook\" ]]; then\n"
        "  exit 0\n"
        "else\n"
        "  echo \"micromamba stub unsupported: $*\" >&2\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    micromamba_stub.chmod(0o755)

    stage_stub = tmp_path / "stage_stub.sh"
    stage_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"${TOX21_DIR}/stage-outputs\"\n"
        "printf '{}' > \"${TOX21_DIR}/stage-outputs/tox21_${TOX21_EVALUATION_MODE}.json\"\n"
        "printf '%s' \"${TOX21_ENCODER_CHECKPOINT}\" > \"${TOX21_DIR}/selected_checkpoint.txt\"\n",
        encoding="utf-8",
    )
    stage_stub.chmod(0o755)

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "APP_DIR": str(REPO_ROOT),
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": exp_id,
            "PRETRAIN_EXP_ID": exp_id,
            "PRETRAIN_EXPERIMENT_ROOT": str(pretrain_root),
            "PRETRAIN_DIR": str(pretrain_dir),
            "PRETRAIN_ARTIFACTS_DIR": str(pretrain_artifacts),
            "PRETRAIN_MANIFEST": str(manifest_path),
            "PRETRAIN_STATE_FILE": str(state_path),
            "PRETRAIN_STATE_FILE_CANONICAL": str(state_path),
            "PRETRAIN_TOX21_ENV": str(pretrain_root / "tox21_gate.env"),
            "FINETUNE_DIR": str(finetune_dir),
            "TOX21_DIR": str(tox21_dir),
            "GITHUB_ENV": str(pretrain_root / "tox21_gate.env"),
            "WANDB_API_KEY": "",
            "STAGE_BIN": str(stage_stub),
            "MMBIN": str(micromamba_stub),
            "MAMBA_ROOT_PREFIX": str(tmp_path / "micromamba_root"),
            "GRID_DIR": str(grid_dir),
            "GRID_SOURCE_DIR": str(grid_dir),
            "DATA_ROOT": str(data_root),
            "TOX21_EVALUATION_MODE": "fine_tuned",
            "TOX21_ENCODER_SOURCE": "fine_tuned",
            "MJEPACI_STAGE_SHIM": str(stage_stub),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/ci/run-tox21.sh"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    selected = (tox21_dir / "selected_checkpoint.txt").read_text(encoding="utf-8")
    assert selected == str(encoder_path)


def test_tox21_prefers_assay_task_checkpoints_over_seed_zero(tmp_path):
    experiments_root = tmp_path / "experiments"
    exp_id = "19678842966"
    pretrain_root = experiments_root / exp_id
    pretrain_dir = pretrain_root / "pretrain"
    pretrain_artifacts = pretrain_root / "artifacts"
    finetune_dir = pretrain_root / "finetune"
    tox21_dir = pretrain_root / "tox21"

    for path in (pretrain_dir, pretrain_artifacts, finetune_dir, tox21_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = pretrain_artifacts / "encoder_manifest.json"
    manifest_path.write_text(json.dumps({"paths": {"encoder": "missing"}}), encoding="utf-8")
    state_path = pretrain_root / "pretrain_state.json"
    state_path.write_text("{}", encoding="utf-8")

    data_root = tmp_path / "data"
    cache_root = data_root / "cache" / "finetune"
    assay_ckpt = cache_root / "NR-AR" / "seed_3" / "ft_best.pt"
    assay_ckpt.parent.mkdir(parents=True, exist_ok=True)
    assay_ckpt.write_text("assay-best", encoding="utf-8")

    stage_outputs = cache_root / "stage-outputs"
    stage_outputs.mkdir(parents=True, exist_ok=True)
    finetune_json = stage_outputs / "finetune.json"
    finetune_json.write_text(
        json.dumps(
            {
                "tasks": {
                    "NR-AR": {
                        "encoder_finetuned": {"checkpoint": str(assay_ckpt)},
                        "diagnostics": {"encoder_checkpoint": str(assay_ckpt)},
                    },
                    "SR-ARE": {"selected_path": str(assay_ckpt)},
                },
                "primary_task": "NR-AR",
                "task_order": ["NR-AR", "SR-ARE"],
            }
        ),
        encoding="utf-8",
    )

    micromamba_stub = tmp_path / "micromamba"
    micromamba_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "if [[ \"${1:-}\" == \"run\" ]]; then\n"
        "  shift\n"
        "  if [[ \"${1:-}\" == \"-n\" ]]; then\n"
        "    shift 2\n"
        "  fi\n"
        "  exec \"$@\"\n"
        "elif [[ \"${1:-}\" == \"shell\" && \"${2:-}\" == \"hook\" ]]; then\n"
        "  exit 0\n"
        "else\n"
        "  echo \"micromamba stub unsupported: $*\" >&2\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    micromamba_stub.chmod(0o755)

    stage_stub = tmp_path / "stage_stub.sh"
    stage_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"${TOX21_DIR}/stage-outputs\"\n"
        "printf '{}' > \"${TOX21_DIR}/stage-outputs/tox21_${TOX21_EVALUATION_MODE}.json\"\n"
        "printf '%s' \"${TOX21_ENCODER_CHECKPOINT}\" > \"${TOX21_DIR}/selected_checkpoint.txt\"\n",
        encoding="utf-8",
    )
    stage_stub.chmod(0o755)

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "APP_DIR": str(REPO_ROOT),
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": exp_id,
            "PRETRAIN_EXP_ID": exp_id,
            "PRETRAIN_EXPERIMENT_ROOT": str(pretrain_root),
            "PRETRAIN_DIR": str(pretrain_dir),
            "PRETRAIN_ARTIFACTS_DIR": str(pretrain_artifacts),
            "PRETRAIN_MANIFEST": str(manifest_path),
            "PRETRAIN_STATE_FILE": str(state_path),
            "PRETRAIN_STATE_FILE_CANONICAL": str(state_path),
            "PRETRAIN_TOX21_ENV": str(pretrain_root / "tox21_gate.env"),
            "FINETUNE_DIR": str(finetune_dir),
            "TOX21_DIR": str(tox21_dir),
            "GITHUB_ENV": str(pretrain_root / "tox21_gate.env"),
            "WANDB_API_KEY": "",
            "STAGE_BIN": str(stage_stub),
            "MMBIN": str(micromamba_stub),
            "MAMBA_ROOT_PREFIX": str(tmp_path / "micromamba_root"),
            "GRID_DIR": str(grid_dir),
            "GRID_SOURCE_DIR": str(grid_dir),
            "DATA_ROOT": str(data_root),
            "TOX21_EVALUATION_MODE": "fine_tuned",
            "TOX21_ENCODER_SOURCE": "fine_tuned",
            "MJEPACI_STAGE_SHIM": str(stage_stub),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/ci/run-tox21.sh"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    selected = (tox21_dir / "selected_checkpoint.txt").read_text(encoding="utf-8")
    assert selected == str(assay_ckpt)


def test_tox21_consumes_cached_assay_stage_outputs(tmp_path):
    experiments_root = tmp_path / "experiments"
    exp_id = "19678843010"
    pretrain_root = experiments_root / exp_id
    pretrain_dir = pretrain_root / "pretrain"
    pretrain_artifacts = pretrain_root / "artifacts"
    finetune_dir = pretrain_root / "finetune"
    tox21_dir = pretrain_root / "tox21"

    for path in (pretrain_dir, pretrain_artifacts, finetune_dir, tox21_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = pretrain_artifacts / "encoder_manifest.json"
    manifest_path.write_text(json.dumps({"paths": {"encoder": "missing"}}), encoding="utf-8")
    state_path = pretrain_root / "pretrain_state.json"
    state_path.write_text("{}", encoding="utf-8")

    data_root = tmp_path / "data"
    cache_root = data_root / "cache" / "finetune"

    assay_ckpt = cache_root / "NR-AhR" / "seed_5" / "ft_best.pt"
    assay_ckpt.parent.mkdir(parents=True, exist_ok=True)
    assay_ckpt.write_text("assay-best", encoding="utf-8")

    stage_outputs = cache_root / "stage-outputs" / "NR-AhR"
    stage_outputs.mkdir(parents=True, exist_ok=True)
    finetune_json = stage_outputs / "finetune_NR-AhR.json"
    finetune_json.write_text(
        json.dumps(
            {
                "encoder_finetuned": {"checkpoint": str(assay_ckpt)},
                "tasks": {
                    "NR-AhR": {
                        "encoder_finetuned": {"checkpoint": str(assay_ckpt)},
                        "diagnostics": {"encoder_checkpoint": str(assay_ckpt)},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    micromamba_stub = tmp_path / "micromamba"
    micromamba_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "if [[ \"${1:-}\" == \"run\" ]]; then\n"
        "  shift\n"
        "  if [[ \"${1:-}\" == \"-n\" ]]; then\n"
        "    shift 2\n"
        "  fi\n"
        "  exec \"$@\"\n"
        "elif [[ \"${1:-}\" == \"shell\" && \"${2:-}\" == \"hook\" ]]; then\n"
        "  exit 0\n"
        "else\n"
        "  echo \"micromamba stub unsupported: $*\" >&2\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    micromamba_stub.chmod(0o755)

    stage_stub = tmp_path / "stage_stub.sh"
    stage_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"${TOX21_DIR}/stage-outputs\"\n"
        "printf '{}' > \"${TOX21_DIR}/stage-outputs/tox21_${TOX21_EVALUATION_MODE}.json\"\n"
        "printf '%s' \"${TOX21_ENCODER_CHECKPOINT}\" > \"${TOX21_DIR}/selected_checkpoint.txt\"\n",
        encoding="utf-8",
    )
    stage_stub.chmod(0o755)

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "APP_DIR": str(REPO_ROOT),
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": exp_id,
            "PRETRAIN_EXP_ID": exp_id,
            "PRETRAIN_EXPERIMENT_ROOT": str(pretrain_root),
            "PRETRAIN_DIR": str(pretrain_dir),
            "PRETRAIN_ARTIFACTS_DIR": str(pretrain_artifacts),
            "PRETRAIN_MANIFEST": str(manifest_path),
            "PRETRAIN_STATE_FILE": str(state_path),
            "PRETRAIN_STATE_FILE_CANONICAL": str(state_path),
            "PRETRAIN_TOX21_ENV": str(pretrain_root / "tox21_gate.env"),
            "FINETUNE_DIR": str(finetune_dir),
            "TOX21_DIR": str(tox21_dir),
            "GITHUB_ENV": str(pretrain_root / "tox21_gate.env"),
            "WANDB_API_KEY": "",
            "STAGE_BIN": str(stage_stub),
            "MMBIN": str(micromamba_stub),
            "MAMBA_ROOT_PREFIX": str(tmp_path / "micromamba_root"),
            "GRID_DIR": str(grid_dir),
            "GRID_SOURCE_DIR": str(grid_dir),
            "DATA_ROOT": str(data_root),
            "TOX21_EVALUATION_MODE": "fine_tuned",
            "TOX21_ENCODER_SOURCE": "fine_tuned",
            "MJEPACI_STAGE_SHIM": str(stage_stub),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/ci/run-tox21.sh"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    selected = (tox21_dir / "selected_checkpoint.txt").read_text(encoding="utf-8")
    assert selected == str(assay_ckpt)


def test_tox21_prioritizes_assay_task_over_export_checkpoint(tmp_path):
    experiments_root = tmp_path / "experiments"
    exp_id = "19678842999"
    pretrain_root = experiments_root / exp_id
    pretrain_dir = pretrain_root / "pretrain"
    pretrain_artifacts = pretrain_root / "artifacts"
    finetune_dir = pretrain_root / "finetune"
    tox21_dir = pretrain_root / "tox21"

    for path in (pretrain_dir, pretrain_artifacts, finetune_dir, tox21_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = pretrain_artifacts / "encoder_manifest.json"
    manifest_path.write_text(json.dumps({"paths": {"encoder": "missing"}}), encoding="utf-8")
    state_path = pretrain_root / "pretrain_state.json"
    state_path.write_text("{}", encoding="utf-8")

    stage_outputs = finetune_dir / "stage-outputs"
    stage_outputs.mkdir(parents=True, exist_ok=True)

    export_ckpt = finetune_dir / "encoder_ft.pt"
    export_ckpt.write_text("export", encoding="utf-8")

    assay_ckpt = finetune_dir / "NR-AR" / "seed_4" / "ft_best.pt"
    assay_ckpt.parent.mkdir(parents=True, exist_ok=True)
    assay_ckpt.write_text("assay-best", encoding="utf-8")

    finetune_json = stage_outputs / "finetune.json"
    finetune_json.write_text(
        json.dumps(
            {
                "encoder_finetuned": {"checkpoint": str(export_ckpt)},
                "tasks": {
                    "NR-AR": {
                        "encoder_finetuned": {"checkpoint": str(assay_ckpt)},
                        "diagnostics": {"encoder_checkpoint": str(assay_ckpt)},
                    },
                    "SR-ARE": {"selected_path": str(assay_ckpt)},
                },
                "primary_task": "NR-AR",
                "task_order": ["NR-AR", "SR-ARE"],
            }
        ),
        encoding="utf-8",
    )

    micromamba_stub = tmp_path / "micromamba"
    micromamba_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "if [[ \"${1:-}\" == \"run\" ]]; then\n"
        "  shift\n"
        "  if [[ \"${1:-}\" == \"-n\" ]]; then\n"
        "    shift 2\n"
        "  fi\n"
        "  exec \"$@\"\n"
        "elif [[ \"${1:-}\" == \"shell\" && \"${2:-}\" == \"hook\" ]]; then\n"
        "  exit 0\n"
        "else\n"
        "  echo \"micromamba stub unsupported: $*\" >&2\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    micromamba_stub.chmod(0o755)

    stage_stub = tmp_path / "stage_stub.sh"
    stage_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"${TOX21_DIR}/stage-outputs\"\n"
        "printf '{}' > \"${TOX21_DIR}/stage-outputs/tox21_${TOX21_EVALUATION_MODE}.json\"\n"
        "printf '%s' \"${TOX21_ENCODER_CHECKPOINT}\" > \"${TOX21_DIR}/selected_checkpoint.txt\"\n",
        encoding="utf-8",
    )
    stage_stub.chmod(0o755)

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "APP_DIR": str(REPO_ROOT),
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": exp_id,
            "PRETRAIN_EXP_ID": exp_id,
            "PRETRAIN_EXPERIMENT_ROOT": str(pretrain_root),
            "PRETRAIN_DIR": str(pretrain_dir),
            "PRETRAIN_ARTIFACTS_DIR": str(pretrain_artifacts),
            "PRETRAIN_MANIFEST": str(manifest_path),
            "PRETRAIN_STATE_FILE": str(state_path),
            "PRETRAIN_STATE_FILE_CANONICAL": str(state_path),
            "PRETRAIN_TOX21_ENV": str(pretrain_root / "tox21_gate.env"),
            "FINETUNE_DIR": str(finetune_dir),
            "TOX21_DIR": str(tox21_dir),
            "GITHUB_ENV": str(pretrain_root / "tox21_gate.env"),
            "WANDB_API_KEY": "",
            "STAGE_BIN": str(stage_stub),
            "MMBIN": str(micromamba_stub),
            "MAMBA_ROOT_PREFIX": str(tmp_path / "micromamba_root"),
            "GRID_DIR": str(grid_dir),
            "GRID_SOURCE_DIR": str(grid_dir),
            "DATA_ROOT": str(tmp_path / "data"),
            "TOX21_EVALUATION_MODE": "fine_tuned",
            "TOX21_ENCODER_SOURCE": "fine_tuned",
            "MJEPACI_STAGE_SHIM": str(stage_stub),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/ci/run-tox21.sh"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    selected = (tox21_dir / "selected_checkpoint.txt").read_text(encoding="utf-8")
    assert selected == str(assay_ckpt)


def test_tox21_consumes_best_available_seed_checkpoint(tmp_path):
    experiments_root = tmp_path / "experiments"
    exp_id = "222222"
    pretrain_root = experiments_root / exp_id
    pretrain_dir = pretrain_root / "pretrain"
    pretrain_artifacts = pretrain_root / "artifacts"
    finetune_dir = pretrain_root / "finetune"
    tox21_dir = pretrain_root / "tox21"

    for path in (pretrain_dir, pretrain_artifacts, finetune_dir, tox21_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = pretrain_artifacts / "encoder_manifest.json"
    manifest_path.write_text(json.dumps({"paths": {"encoder": "missing"}}), encoding="utf-8")
    state_path = pretrain_root / "pretrain_state.json"
    state_path.write_text("{}", encoding="utf-8")

    stage_outputs = finetune_dir / "stage-outputs"
    stage_outputs.mkdir(parents=True, exist_ok=True)
    finetune_json = stage_outputs / "finetune.json"
    finetune_json.write_text(json.dumps({"encoder_finetuned": {"checkpoint": "missing"}}), encoding="utf-8")

    seed_one_ckpt = finetune_dir / "seed_1" / "ft_best.pt"
    seed_two_ckpt = finetune_dir / "seed_2" / "ft_best.pt"
    seed_one_ckpt.parent.mkdir(parents=True, exist_ok=True)
    seed_two_ckpt.parent.mkdir(parents=True, exist_ok=True)
    seed_one_ckpt.write_text("seed-1", encoding="utf-8")
    seed_two_ckpt.write_text("seed-2", encoding="utf-8")

    micromamba_stub = tmp_path / "micromamba"
    micromamba_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "if [[ \"${1:-}\" == \"run\" ]]; then\n"
        "  shift\n"
        "  if [[ \"${1:-}\" == \"-n\" ]]; then\n"
        "    shift 2\n"
        "  fi\n"
        "  exec \"$@\"\n"
        "elif [[ \"${1:-}\" == \"shell\" && \"${2:-}\" == \"hook\" ]]; then\n"
        "  exit 0\n"
        "else\n"
        "  echo \"micromamba stub unsupported: $*\" >&2\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    micromamba_stub.chmod(0o755)

    stage_stub = tmp_path / "stage_stub.sh"
    stage_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"${TOX21_DIR}/stage-outputs\"\n"
        "printf '{}' > \"${TOX21_DIR}/stage-outputs/tox21_${TOX21_EVALUATION_MODE}.json\"\n"
        "printf '%s' \"${TOX21_ENCODER_CHECKPOINT}\" > \"${TOX21_DIR}/selected_checkpoint.txt\"\n",
        encoding="utf-8",
    )
    stage_stub.chmod(0o755)

    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "APP_DIR": str(REPO_ROOT),
            "EXPERIMENTS_ROOT": str(experiments_root),
            "EXP_ID": exp_id,
            "PRETRAIN_EXP_ID": exp_id,
            "PRETRAIN_EXPERIMENT_ROOT": str(pretrain_root),
            "PRETRAIN_DIR": str(pretrain_dir),
            "PRETRAIN_ARTIFACTS_DIR": str(pretrain_artifacts),
            "PRETRAIN_MANIFEST": str(manifest_path),
            "PRETRAIN_STATE_FILE": str(state_path),
            "PRETRAIN_STATE_FILE_CANONICAL": str(state_path),
            "PRETRAIN_TOX21_ENV": str(pretrain_root / "tox21_gate.env"),
            "FINETUNE_DIR": str(finetune_dir),
            "TOX21_DIR": str(tox21_dir),
            "GITHUB_ENV": str(pretrain_root / "tox21_gate.env"),
            "WANDB_API_KEY": "",
            "STAGE_BIN": str(stage_stub),
            "MMBIN": str(micromamba_stub),
            "MAMBA_ROOT_PREFIX": str(tmp_path / "micromamba_root"),
            "GRID_DIR": str(grid_dir),
            "GRID_SOURCE_DIR": str(grid_dir),
            "DATA_ROOT": str(tmp_path / "data"),
            "TOX21_EVALUATION_MODE": "fine_tuned",
            "TOX21_ENCODER_SOURCE": "fine_tuned",
            "MJEPACI_STAGE_SHIM": str(stage_stub),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/ci/run-tox21.sh"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    selected = (tox21_dir / "selected_checkpoint.txt").read_text(encoding="utf-8")
    assert selected == str(seed_one_ckpt)


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


def test_build_stage_args_injects_no_calibrate_from_env(tmp_path):
    best_cfg = {"config": {}}
    best_path = tmp_path / "best_grid_config.json"
    best_path.write_text(json.dumps(best_cfg), encoding="utf-8")

    encoder_path = tmp_path / "encoder.pt"
    encoder_path.write_text("stub", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"paths": {"encoder": str(encoder_path)}}), encoding="utf-8"
    )
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
            "WANDB_API_KEY": "",
        }
    )

    def _collect(extra_env: dict[str, str]) -> list[str]:
        env = base_env.copy()
        env.update(extra_env)
        capture_path = tmp_path / f"calibrate_capture_{len(extra_env)}.txt"
        env["ARGS_CAPTURE"] = str(capture_path)
        script = f"""
set -euo pipefail
source \"{REPO_ROOT}/scripts/ci/common.sh\"
source \"{REPO_ROOT}/scripts/ci/stage.sh\"
build_stage_args tox21
printf '%s\\n' \"${{STAGE_ARGS[@]}}\" > \"$ARGS_CAPTURE\"
"""
        subprocess.run(["bash", "-lc", script], check=True, cwd=REPO_ROOT, env=env)
        return capture_path.read_text(encoding="utf-8").splitlines()

    args_default = _collect({})
    assert "--no-calibrate" not in args_default

    args_flag = _collect({"TOX21_NO_CALIBRATE": "1"})
    assert "--no-calibrate" in args_flag

    args_calibrate_false = _collect({"TOX21_CALIBRATE": "false"})
    assert "--no-calibrate" in args_calibrate_false


def test_build_stage_args_deduplicates_bestcfg_overrides(tmp_path):
    best_cfg = {
        "config": {
            "batch_size": {"value": 128},
            "prefetch_factor": {"value": 2},
            "num_workers": {"value": 6},
            "persistent_workers": {"value": 1},
            "pin_memory": {"value": 1},
            "bf16": {"value": 1},
        }
    }
    best_path = tmp_path / "best_grid_config.json"
    best_path.write_text(json.dumps(best_cfg), encoding="utf-8")

    encoder_path = tmp_path / "encoder.pt"
    encoder_path.write_text("stub", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"paths": {"encoder": str(encoder_path)}}), encoding="utf-8"
    )
    tox21_dir = tmp_path / "tox21"
    tox21_dir.mkdir()

    env = os.environ.copy()
    env.update(
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
            "WANDB_API_KEY": "",
        }
    )

    capture_path = tmp_path / "dedup_args.txt"
    env["ARGS_CAPTURE"] = str(capture_path)

    script = f"""
set -euo pipefail
source \"{REPO_ROOT}/scripts/ci/common.sh\"
source \"{REPO_ROOT}/scripts/ci/stage.sh\"
build_stage_args tox21
printf '%s\\n' \"${{STAGE_ARGS[@]}}\" > \"$ARGS_CAPTURE\"
"""
    subprocess.run(["bash", "-lc", script], check=True, cwd=REPO_ROOT, env=env)

    tokens = capture_path.read_text(encoding="utf-8").splitlines()

    def _value(flag: str) -> str | None:
        assert flag in tokens, flag
        idx = tokens.index(flag)
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            return tokens[idx + 1]
        return None

    for flag in (
        "--batch-size",
        "--prefetch-factor",
        "--num-workers",
        "--persistent-workers",
        "--pin-memory",
        "--bf16",
    ):
        assert tokens.count(flag) <= 1

    assert _value("--batch-size") == "128"
    assert _value("--prefetch-factor") == "2"
    assert _value("--num-workers") == "6"
    if "--persistent-workers" in tokens:
        assert _value("--persistent-workers") in {"1", None}
    if "--pin-memory" in tokens:
        assert _value("--pin-memory") in {"1", None}
    if "--bf16" in tokens:
        assert _value("--bf16") in {"1", None}
