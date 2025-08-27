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
  # Usage: yaml_args <section>
  # Prints one argument per line: --key <value> OR --flag (for booleans)
  # Rules:
  #  - KEEP underscores in flag names (train_jepa.py uses underscores)
  #  - Strings with spaces => wrap in double quotes
  #  - Env refs like ${VAR} or $VAR => keep for shell to expand (double-quote)
  #  - Lists => repeat the flag
  #  - true => boolean flag present; false => omitted
  python - "$@" <<'PY'
  import sys, os, yaml, re
  section = sys.argv[1] if len(sys.argv) > 1 else "grid_search"
  with open(os.environ.get("TRAIN_JEPA_CI", "scripts/ci/train_jepa_ci.yml"), "r") as f:
      cfg = yaml.safe_load(f) or {}
  node = cfg.get(section, {})
  env_ref = re.compile(r'^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$')

  def emit(k, v):
      key = "--" + k  # KEEP underscores
      if isinstance(v, bool):
          if v: print(key)
          return
      if isinstance(v, (int, float)):
          print(key); print(str(v)); return
      if isinstance(v, list):
          for item in v:
              emit(k, item)
          return
      if v is None:
          return
      # string
      s = str(v)
      if env_ref.match(s):
          # leave $VAR / ${VAR} for shell expansion; double-quote to keep spaces
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
