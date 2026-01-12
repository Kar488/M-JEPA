import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _lock_env(tmp_path: Path, exp_id: str) -> dict:
    data_root = tmp_path / "data"
    exp_root = data_root / "experiments"
    exp_dir = exp_root / exp_id
    env = os.environ.copy()
    env.update(
        {
            "APP_DIR": "/srv/mjepa",
            "DATA_ROOT": str(data_root),
            "EXPERIMENTS_ROOT": str(exp_root),
            "EXPERIMENT_DIR": str(exp_dir),
            "EXP_ID": exp_id,
            "MJEPACI_DISABLE_LOCKS": "0",
        }
    )
    return env


def _run_bash(cmd: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_lock_blocks_concurrent_run(tmp_path: Path) -> None:
    exp_id = "lock-live"
    env = _lock_env(tmp_path, exp_id)
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        lock_path_result = _run_bash("source scripts/ci/common.sh; ci_stage_lock_path pretrain", env)
        assert lock_path_result.returncode == 0
        lock_path = Path(lock_path_result.stdout.strip())
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            f"PID={sleeper.pid}\nSTART_TS=now\nSTAGE=pretrain\nEXP_ID={exp_id}\nCMDLINE=sleep\n",
            encoding="utf-8",
        )

        result = _run_bash("source scripts/ci/common.sh; ci_stage_lock_acquire pretrain", env)
        assert result.returncode == 3
        assert "lock already held" in result.stderr
        assert lock_path.exists()
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)


def test_lock_clears_stale_pid(tmp_path: Path) -> None:
    exp_id = "lock-stale"
    env = _lock_env(tmp_path, exp_id)
    lock_path_result = _run_bash("source scripts/ci/common.sh; ci_stage_lock_path pretrain", env)
    assert lock_path_result.returncode == 0
    lock_path = Path(lock_path_result.stdout.strip())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        "PID=999999\nSTART_TS=stale\nSTAGE=pretrain\nEXP_ID=lock-stale\nCMDLINE=stale\n",
        encoding="utf-8",
    )

    result = _run_bash(
        "source scripts/ci/common.sh; ci_stage_lock_acquire pretrain; cat \"$MJEPACI_LOCK_PATH\"",
        env,
    )
    assert result.returncode == 0
    assert "PID=999999" not in result.stdout


def test_rerun_blocked_when_lock_held(tmp_path: Path) -> None:
    exp_id = "lock-rerun"
    env = _lock_env(tmp_path, exp_id)
    env.update(
        {
            "TOX21_DIR": str(tmp_path / "tox21"),
            "FORCE_RERUN": "tox21",
            "MJEPACI_DISABLE_CLEANUP": "1",
        }
    )
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        lock_path_result = _run_bash("source scripts/ci/common.sh; ci_stage_lock_path tox21", env)
        assert lock_path_result.returncode == 0
        lock_path = Path(lock_path_result.stdout.strip())
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            f"PID={sleeper.pid}\nSTART_TS=now\nSTAGE=tox21\nEXP_ID={exp_id}\nCMDLINE=sleep\n",
            encoding="utf-8",
        )

        result = _run_bash("source scripts/ci/common.sh; source scripts/ci/stage.sh; run_stage tox21", env)
        assert result.returncode == 3
        assert "rerun blocked by existing lock" in result.stderr
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)
