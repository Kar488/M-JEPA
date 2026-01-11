import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _spawn_train_process(exp_id: str, env: dict, extra_args: list[str] | None = None) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-c",
        "import time; time.sleep(300)",
        "/srv/mjepa/scripts/train_jepa.py",
        "pretrain",
        "--cache-dir",
        f"/data/mjepa/experiments/{exp_id}/cache",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_cleanup(stage: str, env: dict, dry_run: bool = False) -> subprocess.CompletedProcess:
    cleanup_env = env.copy()
    cleanup_env["MJEPACI_STAGE"] = stage
    if dry_run:
        cleanup_env["MJEPACI_CLEANUP_DRYRUN"] = "1"
    cmd = f"source scripts/ci/common.sh; cleanup_preflight {shlex.quote(stage)}"
    return subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        env=cleanup_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _base_env(tmp_path: Path, exp_id: str) -> dict:
    env = os.environ.copy()
    data_root = tmp_path / "data"
    exp_root = data_root / "experiments"
    exp_dir = exp_root / exp_id
    env.update(
        {
            "APP_DIR": "/srv/mjepa",
            "DATA_ROOT": str(data_root),
            "EXPERIMENTS_ROOT": str(exp_root),
            "EXPERIMENT_DIR": str(exp_dir),
            "EXP_ID": exp_id,
            "MJEPACI_DISABLE_CLEANUP": "0",
        }
    )
    return env


def _wait_for_exit(proc: subprocess.Popen, timeout: float = 5) -> bool:
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def _wait_for_train_proc_exit(pid: int, timeout: float = 5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except FileNotFoundError:
            return True
        if b"train_jepa.py" not in cmdline:
            return True
        time.sleep(0.2)
    return False


def test_cleanup_dry_run_lists_candidates_without_killing(tmp_path: Path) -> None:
    exp_id = "cleanup-dryrun"
    env = _base_env(tmp_path, exp_id)
    proc = _spawn_train_process(exp_id, env)
    try:
        result = _run_cleanup("pretrain", env, dry_run=True)
        assert result.returncode == 0
        assert "dry-run" in result.stderr
        assert proc.poll() is None
    finally:
        proc.send_signal(signal.SIGTERM)
        _wait_for_exit(proc)


def test_cleanup_kills_matching_process(tmp_path: Path) -> None:
    exp_id = "cleanup-kill"
    env = _base_env(tmp_path, exp_id)
    proc = _spawn_train_process(exp_id, env)
    try:
        result = _run_cleanup("pretrain", env, dry_run=False)
        assert result.returncode == 0
        assert _wait_for_exit(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)


def test_cleanup_skips_unrelated_process(tmp_path: Path) -> None:
    exp_id = "cleanup-unrelated"
    env = _base_env(tmp_path, exp_id)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        result = _run_cleanup("pretrain", env, dry_run=False)
        assert result.returncode == 0
        assert proc.poll() is None
    finally:
        proc.send_signal(signal.SIGTERM)
        _wait_for_exit(proc)


def test_cleanup_kills_torchrun_process_group(tmp_path: Path) -> None:
    exp_id = "cleanup-torchrun"
    env = _base_env(tmp_path, exp_id)
    parent_code = "\n".join(
        [
            "import os",
            "import subprocess",
            "import sys",
            "import time",
            "child = subprocess.Popen([",
            "    sys.executable,",
            "    '-c',",
            "    'import time; time.sleep(300)',",
            "    '/srv/mjepa/scripts/train_jepa.py',",
            "    'pretrain',",
            "    '--cache-dir',",
            "    os.path.join(os.environ['EXPERIMENT_DIR'], 'cache'),",
            "], env=os.environ.copy())",
            "print(child.pid, flush=True)",
            "time.sleep(300)",
        ]
    )
    parent_cmd = [sys.executable, "-c", parent_code, "torchrun"]
    parent = subprocess.Popen(
        parent_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    child_pid = None
    try:
        assert parent.stdout is not None
        child_line = parent.stdout.readline().strip()
        assert child_line
        child_pid = int(child_line)
        time.sleep(0.5)
        cmdline = Path(f"/proc/{child_pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8")
        assert "/srv/mjepa/scripts/train_jepa.py" in cmdline
        assert env["EXPERIMENT_DIR"] in cmdline
        assert os.getpgid(parent.pid) == os.getpgid(child_pid)
        result = _run_cleanup("pretrain", env, dry_run=False)
        assert result.returncode == 0
        assert "match" in result.stderr
        assert _wait_for_exit(parent)
        if child_pid:
            assert _wait_for_train_proc_exit(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
