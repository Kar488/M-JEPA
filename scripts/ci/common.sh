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
  local section="$1"
  "$MMBIN" run -n mjepa python - "$section" <<'PY'
import os, sys, yaml, shlex
section = sys.argv[1]
path = os.path.join(os.environ["APP_DIR"], "scripts/ci/train_jepa_ci.yml")
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
cfg = data[section]

def expand(val):
    if isinstance(val, str):
        return os.path.expandvars(val)
    if isinstance(val, list):
        return [expand(v) for v in val]
    return val

cfg = {k: expand(v) for k, v in cfg.items()}
args = []
for k, v in cfg.items():
    key = "--" + k
    if isinstance(v, bool):
        if v:
            args.append(key)
    elif isinstance(v, list):
        for item in v:
            args.extend([key, str(item)])
    else:
        args.extend([key, str(v)])
print(" ".join(shlex.quote(a) for a in args))
PY
}
