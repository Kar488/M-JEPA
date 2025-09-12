import os, json, tempfile, subprocess, shlex, pathlib,sys
import pytest


# Path to your real common.sh (override with env COMMON_SH if repo layout differs)
ROOT = pathlib.Path(__file__).resolve().parents[2]
COMMON_SH = os.getenv("COMMON_SH", str(ROOT / "scripts" / "ci" / "common.sh"))

pytestmark = pytest.mark.skipif(not pathlib.Path(COMMON_SH).exists(),
                                reason=f"common.sh not found at {COMMON_SH}")

import os
import pytest


def run_bestcfg(stage: str, cfg: dict, env: dict | None = None) -> str:
    """Call best_config_args <stage> from common.sh with a temp best_grid_config.json."""
    with tempfile.TemporaryDirectory() as td:
        grid_dir = pathlib.Path(td)
        (grid_dir / "best_grid_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        data_dir = grid_dir / "data"
        cache_dir = grid_dir / ".cache"
        data_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Inherit env and inject writable dirs for the bash subprocess.
        run_env = dict(os.environ, **(env or {}))
        run_env.setdefault("DATA_DIR", str(data_dir))
        run_env.setdefault("XDG_CACHE_HOME", str(cache_dir))
         
        # Shim: rewrite any '/data...' path used by common.sh (or scripts it sources)
        # to the per-test DATA_DIR. We wrap common file ops that might target /data.
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
        # Export DATA_DIR/XDG_CACHE_HOME, install shim, then source common.sh.
        # set -x to surface commands in stderr for easier debugging.
        cmd = (
            "set -euo pipefail; set -x; "
            f"export DATA_DIR={shlex.quote(str(data_dir))}; "
            f"export XDG_CACHE_HOME={shlex.quote(str(cache_dir))}; "
            f"{shim} "
            f"source {shlex.quote(COMMON_SH)}; "
            f"export GRID_DIR={shlex.quote(str(grid_dir))}; "
            f"best_config_args {shlex.quote(stage)}"
        )
        proc = subprocess.run(
            ["bash", "-lc", cmd],
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
        # Be robust if the script prints nothing on some platforms
        # Always return a string (never None)
        out = proc.stdout
        if out is None:
            out = ""
        return out

def test_wandb_shape_injects_model_flags_for_bench():
    # Simulate W&B-like JSON: parameters.<key>.value
    cfg = {
        "parameters": {
            "gnn_type": {"value": "gine"},
            "hidden_dim": {"value": 128},
            "num_layers": {"value": 2},
            "learning_rate": {"value": 2.774674e-4},
            "training_method": {"value": "contrastive"},
            "pretrain_epochs": {"value": 5},
            "finetune_epochs": {"value": 1},
        }
    }
    out = run_bestcfg("bench", cfg)
    # Flags must be present for benchmark
    assert "--gnn-type" in out and "gine" in out
    assert "--hidden-dim" in out and "128" in out
    assert "--num-layers" in out and "2" in out
    # learning_rate should map to --lr (not --learning-rate) so benchmark keeps it
    assert "--lr" in out and "--learning-rate" not in out
    # method should toggle --contrastive
    assert "--contrastive" in out

def test_flat_shape_also_works_and_epochs_can_be_skipped_for_tox21():
    # Simulate flat JSON keys
    cfg = {
        "gnn_type": "gine",
        "hidden_dim": 128,
        "num_layers": 2,
        "lr": 3e-4,
        "training_method": "contrastive",
        "pretrain_epochs": 5,
        "finetune_epochs": 1,
    }
    # By default, tox21 should include epochs
    out_with_epochs = run_bestcfg("tox21", cfg)
    tokens = out_with_epochs.split()
    def val(flag):
        return tokens[tokens.index(flag)+1] if flag in tokens else None
    # present
    assert "--pretrain-epochs" in tokens
    assert "--finetune-epochs" in tokens
    # correct values
    assert val("--pretrain-epochs") == "5"
    assert val("--finetune-epochs") == "1"

    # Skip behavior
    out_skipped = run_bestcfg("tox21", cfg, env={"BESTCFG_SKIP": "pretrain_epochs, finetune_epochs"})
    toks2 = out_skipped.split()
    assert "--pretrain-epochs" not in toks2
    assert "--finetune-epochs" not in toks2
    
    # When BESTCFG_NO_EPOCHS=1, epochs must be stripped but model flags remain
    out_no_epochs = run_bestcfg("tox21", cfg, env={"BESTCFG_NO_EPOCHS": "1"})
    assert "--pretrain-epochs" not in out_no_epochs
    assert "--finetune-epochs" not in out_no_epochs
    assert "--gnn-type" in out_no_epochs and "gine" in out_no_epochs
    assert "--hidden-dim" in out_no_epochs and "--num-layers" in out_no_epochs

def test_finetune_stage_emits_epochs_and_honors_no_epochs():
    cfg = {"finetune_epochs": 3, "gnn_type": "gine", "hidden_dim": 64, "num_layers": 2}
    out = run_bestcfg("finetune", cfg)
    toks = out.split()
    assert "--epochs" in toks
    assert toks[toks.index("--epochs")+1] == "3"
    out2 = run_bestcfg("finetune", cfg, env={"BESTCFG_NO_EPOCHS": "1"})
    assert "--epochs" not in out2.split()

def test_bench_alias_and_bestcfg_skip_lr():
    cfg = {
        "parameters": {
            "gnn_type": {"value": "gine"},
            "hidden_dim": {"value": 128},
            "num_layers": {"value": 2},
            "learning_rate": {"value": 1e-4},
        }
    }
    # "benchmark" alias should behave like "bench"
    out_bench = run_bestcfg("bench", cfg)
    out_benchmark = run_bestcfg("benchmark", cfg)
    assert out_bench.strip().split() == out_benchmark.strip().split()

    # BESTCFG_SKIP should drop selected keys (use JSON keys, not CLI flags)
    out_skip_lr = run_bestcfg("bench", cfg, env={"BESTCFG_SKIP": "lr, learning_rate"})
    assert "--lr" not in out_skip_lr  # dropped
    # but other model flags remain
    assert "--gnn-type" in out_skip_lr and "--hidden-dim" in out_skip_lr and "--num-layers" in out_skip_lr

def test_finetune_stage_emits_epochs_and_honors_no_epochs(tmp_path, monkeypatch):
    cfg = {"finetune_epochs": 3, "gnn_type": "gine", "hidden_dim": 64, "num_layers": 2}
    out = run_bestcfg("finetune", cfg)
    toks = out.split()
    assert "--epochs" in toks and toks[toks.index("--epochs")+1] == "3"
    out2 = run_bestcfg("finetune", cfg, env={"BESTCFG_NO_EPOCHS": "1"})
    assert "--epochs" not in out2.split()