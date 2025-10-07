#!/usr/bin/env bash
# Common helpers for Vast GPU CI stages
set -euo pipefail

: "${MJEPACI_STAGE:=}"
: "${PRETRAIN_STATE_FILE:=}"
: "${EXP_ID:=}"
: "${RUN_ID:=$(date +%s)}"

# --- centralised environment variables ---
: "${APP_DIR:=/srv/mjepa}"
: "${VENV_DIR:=/srv/mjepa/.venv}"
: "${LOG_DIR:=${APP_DIR}/logs}"
: "${PRETRAIN_EXP_ID:=}"
: "${PRETRAIN_EXPERIMENT_ROOT:=}"
: "${PRETRAIN_ARTIFACTS_DIR:=}"

__resolve_data_root() {
  local runner_tmp="${RUNNER_TEMP:-/tmp}"
  local env_root="${MJEPA_DATA_ROOT:-}"

  if [[ -n "$env_root" ]]; then
    if mkdir -p "$env_root" 2>/dev/null; then
      printf '%s\n' "$env_root"
      return 0
    fi
    echo "[ci] warn: unable to use MJEPA_DATA_ROOT=$env_root" >&2
  fi

  local vast_root="/data/mjepa"
  if mkdir -p "$vast_root" 2>/dev/null; then
    printf '%s\n' "$vast_root"
    return 0
  fi

  local emergency="${runner_tmp%/}/mjepa"
  echo "[ci] warn: using fallback DATA_ROOT=${emergency}" >&2
  if mkdir -p "$emergency" 2>/dev/null; then
    printf '%s\n' "$emergency"
    return 0
  fi

  echo "[ci] error: unable to create DATA_ROOT at ${env_root:-'<unset>'}, /data/mjepa, or ${emergency}" >&2
  return 1
}

if [[ -z "${APP_DIR:-}" ]]; then
  _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../scripts/ci
  _root="$(cd "${_here}/../.." && pwd)"                      # repo root guess
  if [[ -f "$_root/scripts/train_jepa.py" ]]; then
    APP_DIR="$_root"
  fi
fi

if [[ -n "${EXPERIMENTS_ROOT:-}" ]]; then
  if ! mkdir -p "$EXPERIMENTS_ROOT" 2>/dev/null; then
    local fallback_root="${RUNNER_TEMP:-/tmp}/mjepa/experiments"
    echo "[ci] warn: falling back EXPERIMENTS_ROOT=$fallback_root" >&2
    EXPERIMENTS_ROOT="$fallback_root"
    mkdir -p "$EXPERIMENTS_ROOT" 2>/dev/null || {
      echo "[ci] error: unable to create EXPERIMENTS_ROOT=$EXPERIMENTS_ROOT" >&2
      exit 1
    }
  fi
fi

if [[ -n "${DATA_ROOT:-}" ]]; then
  if ! mkdir -p "$DATA_ROOT" 2>/dev/null; then
    echo "[ci] warn: DATA_ROOT=$DATA_ROOT not writable; falling back to resolver" >&2
    DATA_ROOT=""
  fi
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -n "${EXPERIMENTS_ROOT:-}" ]]; then
    DATA_ROOT="$(dirname "${EXPERIMENTS_ROOT}")"
  else
    DATA_ROOT="$(__resolve_data_root)"
  fi
fi

if [[ -z "${EXPERIMENTS_ROOT:-}" ]]; then
  EXPERIMENTS_ROOT="${DATA_ROOT%/}/experiments"
  mkdir -p "$EXPERIMENTS_ROOT" 2>/dev/null || {
    local fallback_root="${RUNNER_TEMP:-/tmp}/mjepa/experiments"
    echo "[ci] warn: falling back EXPERIMENTS_ROOT=$fallback_root" >&2
    EXPERIMENTS_ROOT="$fallback_root"
    mkdir -p "$EXPERIMENTS_ROOT" 2>/dev/null || {
      echo "[ci] error: unable to create EXPERIMENTS_ROOT=$EXPERIMENTS_ROOT" >&2
      exit 1
    }
  }
fi

: "${MAMBA_ROOT_PREFIX:=${DATA_ROOT}/micromamba}"
: "${MAMBA_ROOT_PREFIX:=${DATA_ROOT}/micromamba}"
: "${CACHE_DIR:=${DATA_ROOT}/cache/graphs_50k}"
# Allow sweeps to reuse the standard graph cache unless the workflow overrides it.
: "${SWEEP_CACHE_DIR:=$CACHE_DIR}"
: "${WANDB_DIR:=${DATA_ROOT}/wandb}"
: "${PRETRAIN_STATE_FILE_LEGACY:=${EXPERIMENTS_ROOT}/pretrain_state.json}"

export DATA_ROOT
export APP_DIR
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CACHE_DIR
export SWEEP_CACHE_DIR

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

resolve_ci_python() {
  local target="${1:-PYTHON_CMD}"
  local -n ref="$target"
  if py=$(python_bin 2>/dev/null); then
    ref=(env PYTHONUNBUFFERED=1 "$py" -u)
  else
    ensure_micromamba
    ref=("$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 python -u)
  fi
}

resolve_encoder_checkpoint() {
  local candidate="${PRETRAIN_ENCODER_PATH:-}"
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  local manifest="${PRETRAIN_MANIFEST:-}"
  if [[ -n "$manifest" && -f "$manifest" ]]; then
    local py resolved=""
    if py=$(python_bin 2>/dev/null); then
      resolved="$("$py" - "$manifest" <<'PY'
import json
import os
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh) or {}
except Exception:
    sys.exit(0)

paths = data.get("paths") or {}
for key in ("encoder", "encoder_symlink"):
    value = paths.get(key)
    if isinstance(value, str) and value.strip():
        print(os.path.abspath(value))
        break
PY
      )"
    fi

    if [[ -n "$resolved" ]]; then
      printf '%s\n' "$resolved"
      return 0
    fi
  fi

  if [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
  fi
}

__parse_pretrain_state_file() {
  local state_path="$1"
  [[ -f "$state_path" ]] || return 1

  local py
  py=$(python_bin 2>/dev/null) || return 1

  local output
  output="$("$py" - "$state_path" <<'PY'
import json
import os
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh) or {}
except FileNotFoundError:
    sys.exit(0)

def emit(key, value):
    if value is None:
        value = ""
    if isinstance(value, (dict, list)):
        return
    print(f"{key}={value}")

emit("id", data.get("id"))
emit("pretrain_exp_id", data.get("pretrain_exp_id"))
emit("experiment_root", data.get("experiment_root"))
emit("artifacts_dir", data.get("artifacts_dir"))
emit("encoder_manifest", data.get("encoder_manifest"))
emit("encoder_checkpoint", data.get("encoder_checkpoint"))
emit("tox21_env", data.get("tox21_env"))
PY
  )" || return 1

  local line key value
  while IFS='=' read -r key value; do
    [[ -z "$key" ]] && continue
    case "$key" in
      id)
        if [[ -n "$value" ]]; then
          [[ -n "${EXP_ID:-}" ]] || EXP_ID="$value"
          [[ -n "${PRETRAIN_EXP_ID:-}" ]] || PRETRAIN_EXP_ID="$value"
        fi
        ;;
      pretrain_exp_id)
        [[ -n "${PRETRAIN_EXP_ID:-}" ]] || PRETRAIN_EXP_ID="$value"
        ;;
      experiment_root)
        [[ -n "${PRETRAIN_EXPERIMENT_ROOT:-}" ]] || PRETRAIN_EXPERIMENT_ROOT="$value"
        ;;
      artifacts_dir)
        [[ -n "${PRETRAIN_ARTIFACTS_DIR:-}" ]] || PRETRAIN_ARTIFACTS_DIR="$value"
        ;;
      encoder_manifest)
        [[ -n "${PRETRAIN_MANIFEST:-}" ]] || PRETRAIN_MANIFEST="$value"
        ;;
      encoder_checkpoint)
        [[ -n "${PRETRAIN_ENCODER_PATH:-}" ]] || PRETRAIN_ENCODER_PATH="$value"
        ;;
      tox21_env)
        [[ -n "${PRETRAIN_TOX21_ENV:-}" ]] || PRETRAIN_TOX21_ENV="$value"
        ;;
    esac
  done <<<"$output"

  return 0
}

__load_pretrain_state() {
  local -a candidates=()

  if [[ -n "${EXP_ID:-}" ]]; then
    candidates+=("${EXPERIMENTS_ROOT}/${EXP_ID}/pretrain_state.json")
  fi

  if [[ -n "${PRETRAIN_STATE_FILE:-}" ]]; then
    candidates+=("$PRETRAIN_STATE_FILE")
  fi

  candidates+=("$PRETRAIN_STATE_FILE_LEGACY")

  local seen=""
  local path
  for path in "${candidates[@]}"; do
    [[ -n "$path" ]] || continue
    [[ -f "$path" ]] || continue
    if __parse_pretrain_state_file "$path"; then
      PRETRAIN_STATE_FILE="$path"
      if [[ -n "${EXP_ID:-}" ]]; then
        local canonical="${EXPERIMENTS_ROOT}/${EXP_ID}/pretrain_state.json"
        if [[ "$canonical" != "$path" && -f "$canonical" ]]; then
          PRETRAIN_STATE_FILE="$canonical"
          __parse_pretrain_state_file "$canonical" || true
        fi
      fi
      return 0
    fi
  done

  return 1
}

# --- accelerator discovery helpers ---
# Return the list of CUDA device ids visible to this process.  We honour an
# existing CUDA_VISIBLE_DEVICES mask so production runs can pin agents to
# subsets of GPUs.  When the variable is absent, fall back to querying PyTorch
# for the number of visible devices.  Any failure to import torch (e.g. CPU-only
# CI) gracefully reports zero GPUs.
visible_gpu_ids() {
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  # Normalise separators and strip whitespace
  visible="${visible//[[:space:]]/}"
  if [[ -n "$visible" && "$visible" != "-1" ]]; then
    IFS=',' read -r -a __grid_cuda_ids <<< "$visible"
    for id in "${__grid_cuda_ids[@]}"; do
      [[ -n "$id" ]] && printf '%s\n' "$id"
    done
    unset __grid_cuda_ids
    return 0
  fi

  local py out
  if py=$(python_bin 2>/dev/null); then
    out="$("$py" - 2>/dev/null <<'PY'
try:
    import torch
    count = torch.cuda.device_count()
except Exception:
    count = 0
print(' '.join(str(i) for i in range(count)))
PY
    )"
  fi

  for id in $out; do
    [[ -n "$id" ]] && printf '%s\n' "$id"
  done
}

gpu_count() {
  local -a ids
  mapfile -t ids < <(visible_gpu_ids)
  echo "${#ids[@]}"
}

# Split a list of GPU ids into roughly even, non-overlapping chunks so each
# parallel agent can be pinned to its own CUDA_VISIBLE_DEVICES mask.  The result
# is written to the named output array as comma-separated strings.
split_gpu_ids() {
  local out_name="$1"; shift
  local agent_count="${1:-0}"; shift || true
  local -a ids=("$@")
  local total="${#ids[@]}"
  local -a result=()

  if (( agent_count <= 0 )); then
    local -n ref="$out_name"
    ref=()
    return 0
  fi

  local base=$(( total / agent_count ))
  local remainder=$(( total % agent_count ))
  local idx=0
  local i
  for (( i=0; i<agent_count; i++ )); do
    local take=$base
    if (( remainder > 0 )); then
      take=$((take + 1))
      ((remainder--))
    fi

    if (( take <= 0 )); then
      result+=("")
      continue
    fi

    local -a chunk=()
    local limit=$((idx + take))
    while (( idx < limit && idx < total )); do
      chunk+=("${ids[idx]}")
      ((idx++))
    done
    result+=("$(IFS=,; echo "${chunk[*]}")")
  done

  local -n ref="$out_name"
  ref=("${result[@]}")
}

# Allow cache directories to be overridden by env vars supplied by the workflow. If Grid_Dir is not set in yaml it uses cache dir

if [[ "${MJEPACI_STAGE}" != "pretrain" ]]; then
  __load_pretrain_state || true
fi

if [[ -z "${EXP_ID:-}" && -n "${PRETRAIN_EXP_ID:-}" ]]; then
  EXP_ID="$PRETRAIN_EXP_ID"
fi

if [[ "${MJEPACI_STAGE}" == "pretrain" && -z "${EXP_ID:-}" ]]; then
  EXP_ID="$RUN_ID"
fi

if [[ -z "${PRETRAIN_EXP_ID:-}" && -n "${EXP_ID:-}" ]]; then
  PRETRAIN_EXP_ID="$EXP_ID"
fi

_needs_pretrain_state=0
case "${MJEPACI_STAGE}" in
  finetune|bench|benchmark|tox21|report)
    _needs_pretrain_state=1
    ;;
esac

if (( _needs_pretrain_state )) && [[ -z "${EXP_ID:-}" ]]; then
  local_hint="${EXPERIMENTS_ROOT}/<EXP_ID>/pretrain_state.json"
  echo "[ci] error: pretrain experiment id missing. Expected EXP_ID env var or state at ${local_hint}" >&2
  exit 1
fi

ensure_dir_var() {
  local var_name="$1"
  local fallback="${2:-}"
  local emergency_rel="${3:-}"
  local current="${!var_name:-}"
  local runner_tmp_base="${RUNNER_TEMP:-/tmp}/mjepa"
  local -a attempts=()

  add_candidate() {
    local path="$1"
    local label="$2"
    [[ -z "$path" ]] && return
    local existing
    for existing in "${attempts[@]}"; do
      if [[ "$existing" == "$path" ]]; then
        return
      fi
    done
    attempts+=("$path")
  }

  add_candidate "$current" "current"
  add_candidate "$fallback" "fallback"

  local suffix="${var_name,,}"
  local emergency="${runner_tmp_base%/}/fallback/${suffix}"
  if [[ -n "${EXP_ID:-}" ]]; then
    emergency="${emergency}/${EXP_ID}"
  fi
  add_candidate "$emergency" "emergency"

  local -a tried=()
  local idx
  for idx in "${!attempts[@]}"; do
    local path="${attempts[$idx]}"
    [[ -z "$path" ]] && continue
    if mkdir -p "$path" 2>/dev/null; then
      printf -v "$var_name" '%s' "$path"
      if (( idx > 0 )); then
        echo "[ci] warn: falling back to $var_name=$path" >&2
      fi
      return 0
    fi
    tried+=("$path")
  done

  echo "[ci] error: unable to create $var_name (attempted: ${tried[*]:-none})" >&2
  return 1
}

if [[ -z "${EXP_ID:-}" ]]; then
  EXP_ID="$RUN_ID"
fi

EXPERIMENT_DIR="${EXPERIMENTS_ROOT%/}/${EXP_ID}"
EXP_ROOT="$EXPERIMENT_DIR"

: "${ARTIFACTS_DIR:=${EXPERIMENT_DIR}/artifacts}"

if [[ -z "${PRETRAIN_EXP_ID:-}" ]]; then
  PRETRAIN_EXP_ID="$EXP_ID"
fi

: "${PRETRAIN_EXPERIMENT_ROOT:=${EXPERIMENT_DIR}}"
: "${PRETRAIN_ARTIFACTS_DIR:=${ARTIFACTS_DIR}}"

# Allow cache directories to be overridden by env vars supplied by the workflow. If Grid_Dir is not set in yaml it uses cache dir
: "${GRID_DIR:=${GRID_CACHE_DIR:-$EXP_ROOT/grid}}"

ensure_dir_var ARTIFACTS_DIR "${EXPERIMENT_DIR}/artifacts" "experiments/${EXP_ID}/artifacts"
ensure_dir_var PRETRAIN_ARTIFACTS_DIR "${ARTIFACTS_DIR}" "experiments/${PRETRAIN_EXP_ID}/artifacts"

if [[ -n "${EXP_ID:-}" ]]; then
  PRETRAIN_STATE_FILE_CANONICAL="${EXPERIMENTS_ROOT}/${EXP_ID}/pretrain_state.json"
else
  PRETRAIN_STATE_FILE_CANONICAL=""
fi

if [[ -z "${PRETRAIN_STATE_FILE:-}" ]]; then
  if [[ -n "${PRETRAIN_STATE_FILE_CANONICAL:-}" ]]; then
    PRETRAIN_STATE_FILE="$PRETRAIN_STATE_FILE_CANONICAL"
  else
    PRETRAIN_STATE_FILE="$PRETRAIN_STATE_FILE_LEGACY"
  fi
fi

if [[ -z "${PRETRAIN_DIR:-}" ]]; then
  if [[ -n "${PRETRAIN_CACHE_DIR:-}" ]]; then
    PRETRAIN_DIR="$PRETRAIN_CACHE_DIR"
  else
    PRETRAIN_DIR="$EXPERIMENT_DIR"
  fi
fi

if [[ -z "${PRETRAIN_MANIFEST:-}" ]]; then
  PRETRAIN_MANIFEST="${PRETRAIN_ARTIFACTS_DIR}/encoder_manifest.json"
fi

if [[ -z "${PRETRAIN_ENCODER_PATH:-}" ]]; then
  PRETRAIN_ENCODER_PATH="${PRETRAIN_DIR}/encoder.pt"
fi

if [[ -z "${PRETRAIN_TOX21_ENV:-}" ]]; then
  PRETRAIN_TOX21_ENV="${PRETRAIN_EXPERIMENT_ROOT}/tox21_gate.env"
fi

: "${FINETUNE_DIR:=${FINETUNE_CACHE_DIR:-$EXP_ROOT/finetune}}"
: "${BENCH_DIR:=${BENCH_CACHE_DIR:-$EXP_ROOT/bench}}"
: "${TOX21_DIR:=${TOX21_CACHE_DIR:-$EXP_ROOT/tox21}}"
: "${REPORTS_DIR:=${REPORTS_CACHE_DIR:-$EXP_ROOT/report}}"

mkdir -p "$CACHE_DIR" "$GRID_DIR" "$PRETRAIN_DIR" "$FINETUNE_DIR" "$BENCH_DIR" \
  "$TOX21_DIR" "$REPORTS_DIR" "$LOG_DIR" "$WANDB_DIR" "$ARTIFACTS_DIR" \
  "$PRETRAIN_ARTIFACTS_DIR"
export GRID_DIR PRETRAIN_DIR FINETUNE_DIR BENCH_DIR TOX21_DIR REPORTS_DIR LOG_DIR
export EXPERIMENT_DIR ARTIFACTS_DIR EXP_ROOT EXPERIMENTS_ROOT
export PRETRAIN_MANIFEST PRETRAIN_EXP_ID PRETRAIN_EXPERIMENT_ROOT PRETRAIN_ARTIFACTS_DIR \
  PRETRAIN_STATE_FILE PRETRAIN_STATE_FILE_CANONICAL PRETRAIN_STATE_FILE_LEGACY \
  PRETRAIN_ENCODER_PATH PRETRAIN_TOX21_ENV EXP_ID

ci_print_env_diag() {
  local stage_bin_value="${1:-${STAGE_BIN:-<unset>}}"
  echo "[ci] EXP_ID=${EXP_ID}" >&2
  echo "[ci] EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT}" >&2
  echo "[ci] DATA_ROOT=${DATA_ROOT}" >&2
  echo "[ci] EXPERIMENT_DIR=${EXPERIMENT_DIR}" >&2
  echo "[ci] PRETRAIN_DIR=${PRETRAIN_DIR}" >&2
  echo "[ci] ARTIFACTS_DIR=${ARTIFACTS_DIR}" >&2
  echo "[ci] PRETRAIN_ARTIFACTS_DIR=${PRETRAIN_ARTIFACTS_DIR}" >&2
  echo "[ci] STAGE_BIN=${stage_bin_value}" >&2

  if [[ -n "${PRETRAIN_EXPERIMENT_ROOT:-}" ]]; then
    echo "[ci] resolved experiment root=${PRETRAIN_EXPERIMENT_ROOT}" >&2
  fi

  if [[ -n "${PRETRAIN_MANIFEST:-}" ]]; then
    echo "[ci] resolved pretrain manifest path=${PRETRAIN_MANIFEST}" >&2
  fi

  if [[ -n "${PRETRAIN_STATE_FILE_CANONICAL:-}" ]]; then
    echo "[ci] resolved pretrain state canonical=${PRETRAIN_STATE_FILE_CANONICAL}" >&2
  fi

  if [[ -n "${PRETRAIN_STATE_FILE:-}" && "${PRETRAIN_STATE_FILE}" != "${PRETRAIN_STATE_FILE_CANONICAL:-}" ]]; then
    echo "[ci] active pretrain state path=${PRETRAIN_STATE_FILE}" >&2
  fi
}

ci_print_env_diag

# --- micromamba bootstrap ---
ensure_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    MMBIN="$(command -v micromamba)"
  elif [ -x "$MAMBA_ROOT_PREFIX/bin/micromamba" ]; then
    MMBIN="$MAMBA_ROOT_PREFIX/bin/micromamba"
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
  #  - Convert underscores to kebab-case flags (cache_dir -> --cache-dir)
  #  - Strings with spaces => wrap in double quotes
  #  - Env refs like ${VAR} or $VAR => keep for shell to expand (double-quote)
  #  - Lists => repeat the flag
  #  - true => boolean flag present; false => omitted
  # Determine a python interpreter (python, python3, etc.)
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
    
    if isinstance(v, (list, tuple)):
        # keys that accept multiple values with a single flag (nargs='+')
        multi = {
            "methods","mask_ratios","contiguities","hidden_dims","num_layers_list",
            "gnn_types","ema_decays","add_3d_options","pretrain_batch_sizes",
            "finetune_batch_sizes","pretrain_epochs_options","finetune_epochs_options",
            "learning_rates","seeds","aug_rotate_options","aug_mask_angle_options",
            "aug_dihedral_options","temperatures"
        }
        if k in multi:
            print(key)
            for item in v: print(item)
        else:
            for item in v: emit(k, item)
        return

    if v is None:
        return
    s = str(v)
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

expand_array_vars() {
  local -n _arr="$1"
  local i
  for i in "${!_arr[@]}"; do
    [[ "${_arr[$i]}" == --* ]] && continue
    _arr[$i]=$(eval "echo ${_arr[$i]}")
  done
}

# --- inject best grid search configuration ---
best_config_args() {
  # Usage: best_config_args <stage>
  # Reads best_grid_config.json and prints CLI args for the given stage
  local stage="$1"
  local cfg="$GRID_DIR/best_grid_config.json"
  if [[ ! -s "$cfg" ]]; then
    # fail fast for stages that require the winner
    case "$stage" in bench|pretrain|finetune|tox21)
      echo "[fatal] $stage needs winner but missing: $cfg" >&2; return 2;;
    esac
    return 0
  fi
  local py; py=$(python_bin) || { echo "python not found" >&2; return 127; }
  "$py" - "$cfg" "$stage" <<'PY'
import os, json, sys
path = sys.argv[1]
# normalize stage and support 'benchmark' alias
stage = (sys.argv[2].lower().strip() if len(sys.argv) > 2 else "bench")
if stage == "benchmark":
    stage = "bench"
raw = json.load(open(path))

# --- flatten possible W&B shapes ---
def _get(key, default=None):
    # 1) flat
    if isinstance(raw, dict) and key in raw:
        v = raw[key]
        if isinstance(v, dict) and "value" in v:  # sometimes already wrapped
            return v["value"]
        return v
    # 2) wandb-like: parameters.<key>.value
    p = raw.get("parameters") or raw.get("config") or {}
    if isinstance(p, dict) and key in p:
        v = p[key]
        if isinstance(v, dict) and "value" in v:
            return v["value"]
        return v
    return default

# Build a flat cfg the rest of the code expects
cfg = {}
for k in ["gnn_type","hidden_dim","num_layers","ema_decay","contiguity","add_3d",
          "lr","learning_rate","temperature","training_method","method","pretrain_epochs","finetune_epochs"]:
    v = _get(k)
    if v is not None:
        cfg[k] = v

# normalise: learning_rate → lr (benchmark/help only exposes --lr)
if "lr" not in cfg and "learning_rate" in cfg:
    cfg["lr"] = cfg["learning_rate"]

# normalise: method → contrastive flag
tm = cfg.get("training_method") or cfg.get("method")
if isinstance(tm, str) and tm.lower().startswith("con"):
    cfg["contrastive"] = True

# --- Unified, refactor-safe mappings (kebab-case) ---
# 1) Dataset / loader / device knobs that multiple stages accept
dataset_loader = {
    "dataset_dir": "--dataset-dir",
    "unlabeled_dir": "--unlabeled-dir",
    "labeled_dir": "--labeled-dir",
    "test_dir":    "--test-dir",
    "report_stem": "--report-stem",
    "csv": "--csv",
    "label_col": "--label-col",
    "smiles_col": "--smiles-col",
    "task_type": "--task-type",
    "cache_dir": "--cache-dir",
    "no_cache": "--no-cache",
    "num_workers": "--num-workers",
    "prefetch_factor": "--prefetch-factor",
    "persistent_workers": "--persistent-workers",
    "pin_memory": "--pin-memory",
    "bf16": "--bf16",
    "devices": "--devices",
    "device": "--device",
}

# 2) Model architecture & training hyperparams (shared)
model_common = {
    "gnn_types": "--gnn-types",
    "hidden_dims": "--hidden-dims",
    "num_layers_list": "--num-layers",
    "ema_decays": "--ema-decays",
    "contiguities": "--contiguities",
    "add_3d_options": "--add-3d-options",
    "lr": "--lr",

    # >>> add singular keys used by best_grid_config.json <<<
    # emit --lr universally so it isn't filtered by benchmark's allow-list
    "gnn_type": "--gnn-type",
    "hidden_dim": "--hidden-dim",
    "num_layers": "--num-layers",
    "ema_decay": "--ema-decay",
    "contiguity": "--contiguity",
    "add_3d": "--add-3d",
    "learning_rate": "--lr",

    # other common knobs
    "learning_rates": "--learning-rates",
    "use_scheduler": "--use-scheduler",
    "warmup_steps": "--warmup-steps",
    "temperature": "--temperature",
    "contrastive": "--contrastive",
}

# 3) Augmentations (JEPA vs. contrastive study toggles)
augment = {
    "aug_rotate_options": "--aug-rotate-options",
    "aug_mask_angle_options": "--aug-mask-angle-options",
    "aug_dihedral_options": "--aug-dihedral-options",
}

# 4) Logging / outputs
logging_io = {
    "out_csv": "--out-csv",
    "best_config_out": "--best-config-out",
    "ckpt_dir": "--ckpt-dir",
    "ckpt_every": "--ckpt-every",
    "output": "--output",
    "force_tqdm": "--force-tqdm",
    "use_wandb": "--use-wandb",
    "wandb_project": "--wandb-project",
    "wandb_tags": "--wandb-tags",
    "report_dir": "--report-dir",
}

# 5) Stage-specific knobs
grid_only = {
    "methods": "--methods",
    "mask_ratios": "--mask-ratios",
    "pretrain_batch_sizes": "--pretrain-batch-sizes",
    "finetune_batch_sizes": "--finetune-batch-sizes",
    "pretrain_epochs_options": "--pretrain-epochs-options",
    "finetune_epochs_options": "--finetune-epochs-options",
    "seeds": "--seeds",
    "sample_unlabeled": "--sample-unlabeled",
    "sample_labeled": "--sample-labeled",
    "n_rows_per_file": "--n-rows-per-file",
    "max_pretrain_batches": "--max-pretrain-batches",
    "target_pretrain_samples": "--target-pretrain-samples",
    "max_finetune_batches": "--max-finetune-batches",
    "time_budget_mins": "--time-budget-mins",
    "temperatures": "--temperatures",
    "force_refresh": "--force-refresh",
}

pretrain_only = {
    "mask_ratio": "--mask-ratio",
    "pretrain_batch_size": "--batch-size",
    "pretrain_epochs": "--epochs",
    "save_every": "--save-every",
}

finetune_only = {
    "finetune_batch_size": "--batch-size",
    "finetune_epochs": "--epochs",
    "metric": "--metric",
    "patience": "--patience",
    "jepa_encoder": "--jepa-encoder",
}

benchmark_only = {
    "jepa_encoder": "--jepa-encoder",
    "ft_ckpt": "--ft-ckpt",
    "dataset": "--dataset",
    "task": "--task",
}

tox21_only = {
    "task": "--task",
    "dataset": "--dataset",
    "pretrain_epochs": "--pretrain-epochs",
    "finetune_epochs": "--finetune-epochs",
    "pretrain_time_budget_mins": "--pretrain-time-budget-mins",
    "finetune_time_budget_mins": "--finetune-time-budget-mins",
}

# 6) Final per-stage maps (compose without duplication)
maps = {
    "grid": {
        **dataset_loader, **model_common, **augment, **logging_io, **grid_only
    },
    "pretrain": {
        **dataset_loader, **model_common, **augment, **logging_io, **pretrain_only
    },
    "finetune": {
        **dataset_loader, **model_common, **augment, **logging_io, **finetune_only
    },
    "bench": {
        **dataset_loader, **model_common, **augment, **logging_io, **benchmark_only
    },
    "tox21": {
        **dataset_loader, **model_common, **augment, **logging_io, **tox21_only
    },
}

# Build the skip set from env (accept space- or comma-separated)
_raw = os.environ.get("BESTCFG_SKIP", "")
# accept comma- or space-separated values and normalize
skip = {s.strip() for s in _raw.replace(",", " ").split() if s.strip()}

# treat 'learning_rate' as an alias for 'lr'
if "learning_rate" in skip:
  skip.add("lr")

# Add epochs if requested
if os.environ.get("BESTCFG_NO_EPOCHS") == "1":
  skip.update(["pretrain_epochs", "finetune_epochs"])

# Drop the keys (if present) - run this ALWAYS
for k in list(skip):
    cfg.pop(k, None)

mapping = maps.get(stage, {})
seen_flags = set()
for key, flag in mapping.items():
    if key not in cfg:
        continue
    if flag in seen_flags:
        continue
    val = cfg[key]
    if isinstance(val, bool):
        if val:
            print(flag)
            seen_flags.add(flag)
    else:
        print(flag)
        seen_flags.add(flag)
        print(val)
PY
}
