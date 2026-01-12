import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_bash(cmd: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_benchmark_forces_single_device_without_ddp() -> None:
    cmd = (
        "source scripts/ci/stage.sh; "
        "args=(--devices 4); "
        "ci_benchmark_enforce_single_device args bench; "
        "printf '%s\\n' \"${args[@]}\""
    )
    result = _run_bash(cmd)
    assert result.returncode == 0
    assert "--devices" in result.stdout
    assert "\n1\n" in result.stdout
    assert "forcing --devices 1" in result.stderr


def test_benchmark_allows_external_ddp() -> None:
    env = os.environ.copy()
    env["WORLD_SIZE"] = "2"
    cmd = (
        "source scripts/ci/stage.sh; "
        "args=(--devices 4); "
        "ci_benchmark_enforce_single_device args bench; "
        "printf '%s\\n' \"${args[@]}\""
    )
    result = _run_bash(cmd, env=env)
    assert result.returncode == 0
    assert "forcing --devices 1" not in result.stderr
    assert "4" in result.stdout
