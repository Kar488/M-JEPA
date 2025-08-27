#!/usr/bin/env bash
# Common helpers for Vast GPU CI stages
set -euo pipefail

# --- centralised environment variables ---
: "${APP_DIR:=/srv/mjepa}"
: "${VENV_DIR:=/srv/mjepa/.venv}"
: "${MAMBA_ROOT_PREFIX:=/data/mjepa/micromamba}"
: "${CACHE_DIR:=/data/mjepa/cache/graphs}"
: "${RUN_ID:=$(date +%s)}"
: "${EXP_ROOT:=/data/mjepa/experiments/${RUN_ID}}"

# Ensure project modules are discoverable when running from subdirectories.
export PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Determine an available Python interpreter. Prefer 'python', fallback to 'python3'.
python_bin() {
  if command -v python >/dev/null 2>&1; then
    echo python
  elif command -v python3 >/dev/null 2>&1; then
    echo python3
  else
    return 1
  fi
}

# Allow cache directories to be overridden by env vars supplied by the workflow
GRID_DIR="${GRID_CACHE_DIR:-$EXP_ROOT/grid}"
PRETRAIN_DIR="${PRETRAIN_CACHE_DIR:-$EXP_ROOT/pretrain}"
FINETUNE_DIR="${FINETUNE_CACHE_DIR:-$EXP_ROOT/finetune}"
BENCH_DIR="${BENCH_CACHE_DIR:-$EXP_ROOT/bench}"
TOX21_DIR="${TOX21_CACHE_DIR:-$EXP_ROOT/tox21}"
LOG_DIR="$EXP_ROOT/logs"
mkdir -p "$GRID_DIR" "$PRETRAIN_DIR" "$FINETUNE_DIR" "$BENCH_DIR" "$TOX21_DIR" "$LOG_DIR"

# --- micromamba bootstrap ---
ensure_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    MMBIN="$(command -v micromamba)"
  elif [ -x "$HOME/micromamba/bin/micromamba" ]; then
    MMBIN="$HOME/micromamba/bin/micromamba"
  else
    echo "micromamba not found" >&2
    return 1
  fi
  export MAMBA_ROOT_PREFIX
  export MMBIN
  eval "$("$MMBIN" shell hook -s bash)" || true
}

# --- cache stamp utilities ---
_stamp() { echo "$1/.stamp"; }
needs_stage() {
  local dir="$1"; shift
  local stamp=$(_stamp "$dir")
  if [ ! -f "$stamp" ]; then return 0; fi
  for f in "$@"; do
    [ ! -e "$f" ] && return 0
    [ "$f" -nt "$stamp" ] && return 0
  done
  return 1
}
mark_stage_done() { touch "$(_stamp "$1")"; }

# --- progress bar helper ---
progress_bar() {
  local pct="$1"
  local w=40
  local done=$((pct*w/100))
  printf '\r['
  printf '%0.s#' $(seq 1 $done)
  printf '%0.s-' $(seq $((done+1)) $w)
  printf '] %s%%' "$pct"
}

simulate_progress() {
  for p in 0 20 40 60 80 100; do
    progress_bar $p
    sleep 0.1
  done
  printf '\n'
}

# --- yaml argument helper ---
yaml_args() {
  # Usage: yaml_args <section>
  # Prints one argument per line: --key <value> OR --flag (for booleans)
  # Rules:
  #  - KEEP underscores in flag names (train_jepa.py uses underscores)
  #  - Strings with spaces => wrap in double quotes
  #  - Env refs like ${VAR} or $VAR => keep for shell to expand (double-quote)
  #  - Lists => repeat the flag
  #  - true => boolean flag present; false => omitted

   local py; py=$(python_bin) || { echo "python not found" >&2; return 127; }
   "$py" - "$@" <<'PY'
import sys, os, yaml, re
section = sys.argv[1] if len(sys.argv) > 1 else "grid_search"
with open(os.environ.get("TRAIN_JEPA_CI"), "r") as f:
    cfg = yaml.safe_load(f) or {}
node = cfg.get(section, {})
env_ref = re.compile(r'^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$')

def emit(k, v):
    key = "--" + k.replace("_","-")
    if isinstance(v, bool):
        if v:
            print(key)
        return
    if isinstance(v, (int, float)):
        print(key); print(str(v)); return
    if isinstance(v, list):
        for item in v:
            emit(k, item)
        return
    if v is None:
        return
    s = str(v)
    if env_ref.match(s):
        print(key); print(f"\"{s}\""); return
    if " " in s or "\t" in s:
        print(key); print(f"\"{s}\""); return
    print(key); print(s)

for k, v in (node or {}).items():
    emit(k, v)
PY
  }

build_argv_from_yaml() {
  # Returns an array named ARGV built from yaml_args output (one token per line)
  local section="${1:-grid_search}"
  local -a tmp
  # Read one token per line, preserve spaces via previous quoting
  mapfile -t tmp < <(yaml_args "$section")
  ARGV=("${tmp[@]}")
}

# --- inject best grid search configuration ---
best_config_args() {
  # Usage: best_config_args <stage>
  # Reads best_grid_config.json and prints CLI args for the given stage
  local stage="$1"
  local cfg="$GRID_DIR/best_grid_config.json"
  [ -f "$cfg" ] || return 0
  local py; py=$(python_bin) || { echo "python not found" >&2; return 127; }
  "$py" - "$cfg" "$stage" <<'PY'
import json, sys
path, stage = sys.argv[1], sys.argv[2]
cfg = json.load(open(path))

common = {
    "gnn_type": "--gnn-type",
    "hidden_dim": "--hidden-dim",
    "num_layers": "--num-layers",
    "ema_decay": "--ema-decay",
    "contiguous": "--contiguous",
    "add_3d": "--add-3d",
    "lr": "--lr",
}
pre = {
    "mask_ratio": "--mask-ratio",
    "pretrain_batch_size": "--batch-size",
    "pretrain_epochs": "--epochs",
    "temperature": "--temperature",
    "contrastive": "--contrastive",
}
ft = {
    "finetune_batch_size": "--batch-size",
    "finetune_epochs": "--epochs",
}
maps = {
    "pretrain": {**common, **pre},
    "finetune": {**common, **ft},
    "bench": {**common, **ft},
    "tox21": {**common, **ft},
}
mapping = maps.get(stage, {})
for key, flag in mapping.items():
    if key not in cfg:
        continue
    val = cfg[key]
    if isinstance(val, bool):
        if val:
            print(flag)
    else:
        print(flag)
        print(val)
PY
}
