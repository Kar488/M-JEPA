import json
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
COMMON_SH = os.getenv("COMMON_SH", str(ROOT / "scripts" / "ci" / "common.sh"))
BASH = os.getenv("BASH") or shutil.which("bash")

pytestmark = [
    pytest.mark.skipif(
        not pathlib.Path(COMMON_SH).exists(),
        reason=f"common.sh not found at {COMMON_SH}",
    ),
    pytest.mark.skipif(BASH is None, reason="bash not found; set env BASH to Git Bash"),
]


def run_bestcfg(stage: str, cfg: dict, env: dict | None = None) -> tuple[str, str]:
    """Execute best_config_args <stage> and capture stdout/stderr."""

    with tempfile.TemporaryDirectory() as td:
        grid_dir = pathlib.Path(td)
        (grid_dir / "best_grid_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        data_dir = grid_dir / "data"
        cache_dir = grid_dir / ".cache"
        data_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        run_env = dict(os.environ, **(env or {}))
        run_env.setdefault("DATA_DIR", str(data_dir))
        run_env.setdefault("XDG_CACHE_HOME", str(cache_dir))

        shim = r'''
rewrite_path() {
  case "$1" in
    /data|/data/*) printf '%s' "$DATA_DIR${1#/data}";;
    *)             printf '%s' "$1";;
  esac
}
_wrap_args_and_exec() {
  local cmd="$1"; shift
  local a args=()
  for a in "$@"; do
    args+=("$(rewrite_path "$a")")
  done
  command "$cmd" "${args[@]}"
}
mkdir()  { _wrap_args_and_exec mkdir "$@"; }
install(){ _wrap_args_and_exec install "$@"; }
cp()     { _wrap_args_and_exec cp "$@"; }
mv()     { _wrap_args_and_exec mv "$@"; }
touch()  { _wrap_args_and_exec touch "$@"; }
tee()    { _wrap_args_and_exec tee "$@"; }
export -f rewrite_path _wrap_args_and_exec mkdir install cp mv touch tee
'''

        msys_common = pathlib.Path(COMMON_SH).as_posix()
        msys_grid = pathlib.Path(str(grid_dir)).as_posix()
        cmd = (
            "set -euo pipefail; set -x; "
            f"export DATA_DIR={shlex.quote(pathlib.Path(data_dir).as_posix())}; "
            f"export XDG_CACHE_HOME={shlex.quote(pathlib.Path(cache_dir).as_posix())}; "
            f"{shim} "
            f"source {shlex.quote(msys_common)}; "
            f"export GRID_DIR={shlex.quote(msys_grid)}; "
            f"best_config_args {shlex.quote(stage)}"
        )
        proc = subprocess.run(
            [BASH, "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=run_env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "best_config_args failed:\n"
                f"----- STDERR -----\n{proc.stderr}\n"
                f"----- STDOUT -----\n{proc.stdout}\n"
            )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        return stdout, stderr


def extract_summary(stderr: str) -> dict:
    for line in stderr.splitlines():
        if line.startswith("[bestcfg][summary] "):
            payload = line.split(" ", 1)[1]
            return json.loads(payload)
    raise AssertionError(f"summary line not found in stderr:\n{stderr}")


def test_wandb_shape_injects_model_flags_for_bench():
    cfg = {
        "parameters": {
            "gnn_type": {"value": "gine"},
            "hidden_dim": {"value": 128},
            "num_layers": {"value": 2},
            "learning_rate": {"value": 2.774674e-4},
            "training_method": {"value": "contrastive"},
        }
    }
    stdout, stderr = run_bestcfg("bench", cfg)
    assert "--gnn-type" in stdout
    assert "--hidden-dim" in stdout
    assert "--num-layers" in stdout
    assert "--lr" in stdout and "--learning-rate" not in stdout
    assert "--contrastive" in stdout

    tokens = stdout.split()
    assert tokens.count("--lr") == 1
    assert tokens.count("--contrastive") == 1

    summary = extract_summary(stderr)
    assert summary["stage"] == "bench"
    assert "lr" in summary["best_config_keys"]


@pytest.mark.parametrize(
    "stage,cfg,present,absent,yaml_hits",
    [
        (
            "pretrain",
            {
                "mask_ratio": 0.2154,
                "pretrain_batch_size": 64,
                "lr": 3e-4,
                "prefetch_factor": 2,
                "cache_dir": "/tmp/cache",
                "wandb_project": "ci-pretrain",
            },
            ["--mask-ratio", "--batch-size", "--lr"],
            ["--prefetch-factor", "--cache-dir", "--wandb-project"],
            {"prefetch_factor", "cache_dir", "wandb_project"},
        ),
        (
            "finetune",
            {
                "finetune_batch_size": 32,
                "finetune_epochs": 3,
                "lr": 1e-4,
                "jepa_encoder": "/tmp/encoder.pt",
                "ckpt_dir": "/tmp/ckpt",
            },
            ["--batch-size", "--epochs", "--lr"],
            ["--jepa-encoder", "--ckpt-dir"],
            {"jepa_encoder", "ckpt_dir"},
        ),
        (
            "bench",
            {
                "gnn_type": "gine",
                "hidden_dim": 128,
                "num_layers": 2,
                "lr": 1e-4,
                "dataset": "tox21",
                "task": "roc_auc",
                "jepa_encoder": "/tmp/encoder.pt",
                "ft_ckpt": "/tmp/ft.pt",
            },
            ["--gnn-type", "--hidden-dim", "--num-layers", "--lr"],
            ["--dataset", "--task", "--jepa-encoder", "--ft-ckpt"],
            {"dataset", "task", "jepa_encoder", "ft_ckpt"},
        ),
        (
            "tox21",
            {
                "gnn_type": "gine",
                "hidden_dim": 64,
                "num_layers": 2,
                "lr": 2e-4,
                "pretrain_epochs": 5,
                "finetune_epochs": 1,
                "pretrain_time_budget_mins": 60,
                "finetune_time_budget_mins": 30,
                "report_dir": "/tmp/report",
            },
            ["--gnn-type", "--hidden-dim", "--num-layers", "--lr", "--pretrain-epochs", "--finetune-epochs"],
            ["--pretrain-time-budget-mins", "--finetune-time-budget-mins", "--report-dir"],
            {"pretrain_time_budget_mins", "finetune_time_budget_mins", "report_dir"},
        ),
        (
            "grid",
            {
                "methods": ["jepa", "contrastive"],
                "pretrain_batch_sizes": [32, 64],
                "cache_dir": "/tmp/cache",
                "wandb_tags": ["ci"],
            },
            ["--methods", "--pretrain-batch-sizes"],
            ["--cache-dir", "--wandb-tags"],
            {"cache_dir", "wandb_tags"},
        ),
    ],
)

def test_stage_policy_filters_yaml_owned_keys(stage, cfg, present, absent, yaml_hits):
    stdout, stderr = run_bestcfg(stage, cfg)
    for flag in present:
        assert flag in stdout, f"expected {flag} in stdout for {stage}" 
    for flag in absent:
        assert flag not in stdout, f"expected {flag} to be suppressed for {stage}"

    summary = extract_summary(stderr)
    assert summary["stage"] == ("bench" if stage == "benchmark" else stage)
    assert set(yaml_hits).issubset(set(summary["yaml_owned"]))


def test_bestcfg_keep_overrides_yaml_policy():
    cfg = {
        "mask_ratio": 0.2,
        "pretrain_batch_size": 32,
        "prefetch_factor": 2,
    }
    stdout, stderr = run_bestcfg("pretrain", cfg, env={"BESTCFG_KEEP": "prefetch_factor"})
    assert "--prefetch-factor" in stdout
    summary = extract_summary(stderr)
    assert "prefetch_factor" in summary["forced"]
    assert "prefetch_factor" not in summary["yaml_owned"]


def test_bestcfg_skip_reflected_in_summary():
    cfg = {
        "gnn_type": "gine",
        "hidden_dim": 128,
        "num_layers": 2,
        "lr": 1e-4,
    }
    stdout, stderr = run_bestcfg("bench", cfg, env={"BESTCFG_SKIP": "lr"})
    assert "--lr" not in stdout
    summary = extract_summary(stderr)
    assert "lr" in summary["skipped"]


def test_bestcfg_no_epochs_respected_for_tox21():
    cfg = {
        "gnn_type": "gine",
        "hidden_dim": 128,
        "num_layers": 2,
        "pretrain_epochs": 5,
        "finetune_epochs": 2,
    }
    stdout, stderr = run_bestcfg("tox21", cfg, env={"BESTCFG_NO_EPOCHS": "1"})
    assert "--pretrain-epochs" not in stdout
    assert "--finetune-epochs" not in stdout
    summary = extract_summary(stderr)
    assert set(["pretrain_epochs", "finetune_epochs"]).issubset(set(summary["skipped"]))

