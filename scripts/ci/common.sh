#!/usr/bin/env bash
# Common helpers for Vast GPU CI stages
set -euo pipefail

normalize_bool() {
  local value="${1:-}" default="${2:-0}" result
  if [[ -z "$value" ]]; then
    printf '%s' "$default"
    return 0
  fi
  case "${value,,}" in
    1|true|yes|on) result=1 ;;
    0|false|no|off) result=0 ;;
    *) result="$default" ;;
  esac
  printf '%s' "$result"
}

: "${MJEPACI_STAGE:=}"
: "${PRETRAIN_STATE_FILE:=}"
: "${EXP_ID:=}"
: "${GRID_EXP_ID:=}"
: "${PRETRAIN_EXP_ID:=}"
: "${PRETRAIN_STATE_ID:=}"
: "${RUN_ID:=$(date +%s)}"
FORCE_UNFREEZE_GRID="$(normalize_bool "${FORCE_UNFREEZE_GRID:-}" 0)"
CI_FORCE_UNFREEZE_GRID="$(normalize_bool "${CI_FORCE_UNFREEZE_GRID:-}" "${FORCE_UNFREEZE_GRID}")"
FORCE_UNFREEZE_GRID="$CI_FORCE_UNFREEZE_GRID"
: "${ALLOW_CODE_DRIFT_WHEN_FROZEN:=1}"
STRICT_FROZEN="$(normalize_bool "${STRICT_FROZEN:-}" 0)"
export FORCE_UNFREEZE_GRID CI_FORCE_UNFREEZE_GRID STRICT_FROZEN

ci_refresh_freeze_state() {
  local marker_hint="${1:-}" freeze_id=""
  if [[ -z "$marker_hint" ]]; then
    if [[ -n "${PRETRAIN_EXP_ID:-}" ]]; then
      marker_hint="${EXPERIMENTS_ROOT%/}/${PRETRAIN_EXP_ID}/bench/encoder_frozen.ok"
    elif [[ -n "${PRETRAIN_STATE_ID:-}" ]]; then
      marker_hint="${EXPERIMENTS_ROOT%/}/${PRETRAIN_STATE_ID}/bench/encoder_frozen.ok"
    fi
  fi

  FREEZE_MARKER="$marker_hint"
  if [[ -n "$marker_hint" && -f "$marker_hint" ]]; then
    FROZEN=1
  else
    FROZEN=0
  fi

  if (( FROZEN )) && [[ "${CI_FORCE_UNFREEZE_GRID}" == "1" ]]; then
    FROZEN=0
  fi

  if (( FROZEN )); then
    freeze_id="${PRETRAIN_EXP_ID:-${PRETRAIN_STATE_ID:-}}"
    if [[ -n "$freeze_id" ]]; then
      if [[ -z "${ORIGINAL_PRETRAIN_EXP_ID:-}" ]]; then
        ORIGINAL_PRETRAIN_EXP_ID="$freeze_id"
      fi
      if [[ "${GRID_EXP_ID:-}" != "$freeze_id" ]]; then
        GRID_EXP_ID="$freeze_id"
      fi
      local freeze_grid="${EXPERIMENTS_ROOT%/}/${freeze_id}/grid"
      if [[ -d "$freeze_grid" ]]; then
        GRID_SOURCE_DIR="$freeze_grid"
        if [[ -z "${GRID_DIR:-}" || "${GRID_DIR%/}" == "${EXPERIMENTS_ROOT%/}/${EXP_ID:-}/grid" ]]; then
          GRID_DIR="$freeze_grid"
        fi
      fi
    fi
  fi

  export FREEZE_MARKER FROZEN GRID_EXP_ID GRID_SOURCE_DIR GRID_DIR ORIGINAL_PRETRAIN_EXP_ID
}

# --- centralised environment variables ---
: "${APP_DIR:=/srv/mjepa}"
: "${VENV_DIR:=/srv/mjepa/.venv}"
: "${LOG_DIR:=${APP_DIR}/logs}"
: "${PRETRAIN_EXP_ID:=}"
: "${PRETRAIN_EXPERIMENT_ROOT:=}"
: "${PRETRAIN_ARTIFACTS_DIR:=}"

mjepa_log_warn() {
  echo "[ci] warn: $*" >&2
}

mjepa_log_error() {
  echo "[ci] error: $*" >&2
}

mjepa_try_dir() {
  local path="$1"
  [[ -n "$path" ]] || return 1
  if mkdir -p "$path" 2>/dev/null && [[ -w "$path" ]]; then
    return 0
  fi
  return 1
}

mjepa_detect_data_root() {
  local requested_data="${DATA_ROOT-}"
  if [[ -n "$requested_data" ]]; then
    if mjepa_try_dir "$requested_data"; then
      printf '%s\n' "$requested_data"
      return 0
    fi
    mjepa_log_warn "DATA_ROOT=$requested_data not writable; ignoring"
  fi

  local env_root="${MJEPA_DATA_ROOT-}"
  if [[ -n "$env_root" ]]; then
    if mjepa_try_dir "$env_root"; then
      printf '%s\n' "$env_root"
      return 0
    fi
    mjepa_log_warn "MJEPA_DATA_ROOT=$env_root not writable; ignoring"
  fi

  local vast_root="/data/mjepa"
  if mjepa_try_dir "$vast_root"; then
    printf '%s\n' "$vast_root"
    return 0
  fi

  local runner_tmp="${RUNNER_TEMP:-/tmp}"
  local fallback="${runner_tmp%/}/mjepa"
  if mjepa_try_dir "$fallback"; then
    if [[ "$fallback" != "$requested_data" ]]; then
      mjepa_log_warn "falling back DATA_ROOT=$fallback"
    fi
    printf '%s\n' "$fallback"
    return 0
  fi

  mjepa_log_error "unable to detect writable DATA_ROOT (tried ${requested_data:-<unset>}, ${env_root:-<unset>}, /data/mjepa, $fallback)"
  return 1
}

mjepa_ensure_dir() {
  local var_name="$1"
  local path="$2"
  local label="${3:-$path}"

  if mjepa_try_dir "$path"; then
    printf -v "$var_name" '%s' "$path"
    return 0
  fi

  mjepa_log_error "unable to ensure $var_name=$label"
  return 1
}

if [[ -z "${APP_DIR:-}" ]]; then
  __ci_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  __ci_root="$(cd "${__ci_here}/../.." && pwd)"
  if [[ -f "${__ci_root}/scripts/train_jepa.py" ]]; then
    APP_DIR="${__ci_root}"
  fi
  unset __ci_here __ci_root
fi

requested_experiments="${EXPERIMENTS_ROOT-}"
if [[ -n "$requested_experiments" ]]; then
  if mjepa_try_dir "$requested_experiments"; then
    EXPERIMENTS_ROOT="$requested_experiments"
  else
    mjepa_log_warn "EXPERIMENTS_ROOT=$requested_experiments not writable; falling back"
    unset EXPERIMENTS_ROOT
  fi
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -z "${EXPERIMENTS_ROOT:-}" ]]; then
    if ! DATA_ROOT="$(mjepa_detect_data_root)"; then
      exit 1
    fi
  else
    parent_dir="$(dirname "${EXPERIMENTS_ROOT}")"
    if [[ -z "$parent_dir" || "$parent_dir" == "." ]]; then
      DATA_ROOT="$EXPERIMENTS_ROOT"
    else
      DATA_ROOT="$parent_dir"
    fi
  fi
fi

if [[ -n "${DATA_ROOT:-}" ]] && ! mjepa_try_dir "${DATA_ROOT}"; then
  mjepa_log_warn "DATA_ROOT=${DATA_ROOT} not writable; detecting fallback"
  if ! DATA_ROOT="$(mjepa_detect_data_root)"; then
    exit 1
  fi
fi

if [[ -z "${EXPERIMENTS_ROOT:-}" ]]; then
  EXPERIMENTS_ROOT="${DATA_ROOT%/}/experiments"
fi

if ! mjepa_try_dir "${EXPERIMENTS_ROOT}"; then
  runner_tmp_root="${RUNNER_TEMP:-/tmp}"
  fallback_experiments="${runner_tmp_root%/}/mjepa/experiments"
  if mjepa_try_dir "$fallback_experiments"; then
    mjepa_log_warn "falling back EXPERIMENTS_ROOT=$fallback_experiments"
    EXPERIMENTS_ROOT="$fallback_experiments"
    DATA_ROOT="$(dirname "$fallback_experiments")"
  else
    mjepa_log_error "unable to ensure writable EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT}"
    exit 1
  fi
  unset runner_tmp_root fallback_experiments
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
  derived_parent="$(dirname "${EXPERIMENTS_ROOT}")"
  if [[ -z "$derived_parent" || "$derived_parent" == "." ]]; then
    DATA_ROOT="$EXPERIMENTS_ROOT"
  else
    DATA_ROOT="$derived_parent"
  fi
fi

: "${MAMBA_ROOT_PREFIX:=${DATA_ROOT}/micromamba}"
: "${MAMBA_ROOT_PREFIX:=${DATA_ROOT}/micromamba}"
: "${PRETRAIN_STATE_FILE_LEGACY:=${EXPERIMENTS_ROOT}/pretrain_state.json}"

export DATA_ROOT
export APP_DIR
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

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
          PRETRAIN_STATE_ID="$value"
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

__read_pretrain_lineage_id() {
  local state_path="${1:-}"
  [[ -n "$state_path" && -f "$state_path" ]] || return 1

  local py
  py=$(python_bin 2>/dev/null) || return 1

  local value
  if ! value="$("$py" - "$state_path" <<'PY' 2>/dev/null
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle) or {}

candidate = data.get("pretrain_exp_id") or data.get("id")
if isinstance(candidate, str):
    candidate = candidate.strip()
if candidate:
    print(candidate)
PY
)"; then
    return 1
  fi

  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
    return 0
  fi

  return 1
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

__ci_stage_role="initiator"
case "${MJEPACI_STAGE}" in
  pretrain|grid|grid_search|phase1|phase2_sweep)
    __ci_stage_role="initiator"
    ;;
  *)
    __ci_stage_role="dependent"
    ;;
esac

if [[ "${MJEPACI_STAGE}" != "pretrain" ]]; then
  __load_pretrain_state || true
fi

if [[ -z "${PRETRAIN_EXP_ID:-}" ]]; then
  if id="$(__read_pretrain_lineage_id "${EXPERIMENTS_ROOT}/pretrain_state.json" 2>/dev/null)"; then
    PRETRAIN_EXP_ID="$id"
    if [[ -z "${PRETRAIN_STATE_ID:-}" ]]; then
      PRETRAIN_STATE_ID="$id"
    fi
  fi
fi

if [[ -z "${PRETRAIN_EXP_ID:-}" && -n "${EXP_ID:-}" ]]; then
  if id="$(__read_pretrain_lineage_id "${EXPERIMENTS_ROOT%/}/${EXP_ID}/pretrain_state.json" 2>/dev/null)"; then
    PRETRAIN_EXP_ID="$id"
    if [[ -z "${PRETRAIN_STATE_ID:-}" ]]; then
      PRETRAIN_STATE_ID="$id"
    fi
  fi
fi

if [[ -z "${PRETRAIN_STATE_ID:-}" && -n "${PRETRAIN_EXP_ID:-}" ]]; then
  PRETRAIN_STATE_ID="$PRETRAIN_EXP_ID"
fi

if [[ -z "${ORIGINAL_PRETRAIN_EXP_ID:-}" ]]; then
  ORIGINAL_PRETRAIN_EXP_ID="${PRETRAIN_EXP_ID:-}"
fi
ci_refresh_freeze_state "${FREEZE_MARKER:-}"

if [[ "$__ci_stage_role" == "initiator" && -z "${EXP_ID:-}" ]]; then
  EXP_ID="$RUN_ID"
fi

if [[ "${MJEPACI_STAGE}" == "pretrain" && -z "${PRETRAIN_EXP_ID:-}" && -n "${EXP_ID:-}" ]]; then
  PRETRAIN_EXP_ID="$EXP_ID"
fi

if [[ "${CI_FORCE_UNFREEZE_GRID}" == "1" ]]; then
  case "${MJEPACI_STAGE}" in
    pretrain|grid|grid_search|phase2_sweep)
      if [[ -z "${EXP_ID:-}" ]]; then
        EXP_ID="$RUN_ID"
      fi
      if [[ "${MJEPACI_STAGE}" == "pretrain" && -z "${PRETRAIN_EXP_ID:-}" ]]; then
        PRETRAIN_EXP_ID="$EXP_ID"
      fi
      FROZEN=0
      ;;
  esac
fi

if [[ "$__ci_stage_role" == "dependent" ]]; then
  if (( FROZEN )) && [[ -z "${EXP_ID:-}" ]]; then
    EXP_ID="$RUN_ID"
  elif [[ -z "${EXP_ID:-}" && -n "${PRETRAIN_STATE_ID:-}" ]]; then
    EXP_ID="$PRETRAIN_STATE_ID"
  elif [[ -z "${EXP_ID:-}" ]]; then
    EXP_ID="$RUN_ID"
  fi
fi

if [[ -z "${GRID_EXP_ID:-}" ]]; then
  case "${MJEPACI_STAGE}" in
    pretrain|phase1)
      if [[ -n "${EXP_ID:-}" ]]; then
        GRID_EXP_ID="$EXP_ID"
      fi
      ;;
  esac
fi

if [[ -z "${GRID_EXP_ID:-}" ]]; then
  if [[ -n "${PRETRAIN_EXP_ID:-}" ]]; then
    GRID_EXP_ID="$PRETRAIN_EXP_ID"
  elif [[ -n "${PRETRAIN_STATE_ID:-}" ]]; then
    GRID_EXP_ID="$PRETRAIN_STATE_ID"
  elif [[ -n "${EXP_ID:-}" ]]; then
    GRID_EXP_ID="$EXP_ID"
  fi
fi

if [[ -z "${PRETRAIN_EXP_ID:-}" && -n "${GRID_EXP_ID:-}" && "${MJEPACI_STAGE}" == "pretrain" ]]; then
  PRETRAIN_EXP_ID="$GRID_EXP_ID"
fi

_needs_pretrain_state=0
case "${MJEPACI_STAGE}" in
  finetune|bench|benchmark|tox21|report|phase2|phase2_recheck|phase2_export|grid_recheck|grid_export)
    _needs_pretrain_state=1
    ;;
esac

if (( _needs_pretrain_state )); then
  lineage_hint_primary="${EXPERIMENTS_ROOT}/pretrain_state.json"
  if [[ -n "${EXP_ID:-}" ]]; then
    lineage_hint_secondary="${EXPERIMENTS_ROOT}/${EXP_ID}/pretrain_state.json"
  else
    lineage_hint_secondary="${EXPERIMENTS_ROOT}/<EXP_ID>/pretrain_state.json"
  fi

  if [[ -z "${PRETRAIN_EXP_ID:-}" && -n "${EXP_ID:-}" ]]; then
    fallback_manifest="${EXPERIMENTS_ROOT%/}/${EXP_ID}/artifacts/encoder_manifest.json"
    if [[ -f "$fallback_manifest" ]]; then
      PRETRAIN_EXP_ID="$EXP_ID"
      PRETRAIN_ARTIFACTS_DIR="${EXPERIMENTS_ROOT%/}/${PRETRAIN_EXP_ID}/artifacts"
    fi
  fi

  if [[ -z "${PRETRAIN_EXP_ID:-}" ]]; then
    mjepa_log_error "missing pretrain lineage for ${MJEPACI_STAGE:-unknown}. Set PRETRAIN_EXP_ID=<id> to reuse an existing run or rerun pretrain to refresh pretrain_state.json."
    mjepa_log_error "checked: ${lineage_hint_primary}, ${lineage_hint_secondary}"
    exit 2
  fi

  if [[ -z "${PRETRAIN_ARTIFACTS_DIR:-}" ]]; then
    PRETRAIN_ARTIFACTS_DIR="${EXPERIMENTS_ROOT%/}/${PRETRAIN_EXP_ID}/artifacts"
  fi

  expected_manifest="${PRETRAIN_ARTIFACTS_DIR%/}/encoder_manifest.json"
  if [[ ! -f "$expected_manifest" ]]; then
    case "${MJEPACI_STAGE:-}" in
      bench|benchmark)
        echo "[fatal] missing encoder artifacts for bench: ${expected_manifest}" >&2
        echo "hint: set PRETRAIN_EXP_ID=<id> or rerun pretrain" >&2
        ;;
      *)
        mjepa_log_error "missing pretrain lineage for ${MJEPACI_STAGE:-unknown}. Set PRETRAIN_EXP_ID=<id> to reuse an existing run or rerun pretrain to refresh pretrain_state.json."
        mjepa_log_error "checked: ${lineage_hint_primary}, ${lineage_hint_secondary}"
        ;;
    esac
    exit 2
  fi
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
  local emergency_id="${EXP_ID:-}"
  case "$var_name" in
    GRID_DIR|FINETUNE_DIR|BENCH_DIR|TOX21_DIR|REPORTS_DIR)
      emergency_id="${GRID_EXP_ID:-${emergency_id}}"
      ;;
    ARTIFACTS_DIR|PRETRAIN_ARTIFACTS_DIR|PRETRAIN_DIR)
      emergency_id="${PRETRAIN_EXP_ID:-${emergency_id}}"
      ;;
  esac
  local emergency="${runner_tmp_base%/}/fallback/${suffix}"
  if [[ -n "$emergency_id" ]]; then
    emergency="${emergency}/${emergency_id}"
  fi
  add_candidate "$emergency" "emergency"

  local -a tried=()
  local idx
  for idx in "${!attempts[@]}"; do
    local path="${attempts[$idx]}"
    [[ -z "$path" ]] && continue
    if (( FROZEN )) && [[ "$var_name" =~ ^(PRETRAIN_DIR|PRETRAIN_ARTIFACTS_DIR|ARTIFACTS_DIR)$ ]]; then
      if [[ -n "${PRETRAIN_EXPERIMENT_ROOT:-}" && "$path" == ${PRETRAIN_EXPERIMENT_ROOT%/}* ]]; then
        if [[ -d "$path" ]]; then
          printf -v "$var_name" '%s' "$path"
          if (( idx > 0 )); then
            mjepa_log_warn "using existing read-only ${var_name}=$path"
          fi
          return 0
        fi
      fi
    fi
    if (( FROZEN )) && [[ "$var_name" =~ ^(GRID_DIR|FINETUNE_DIR|BENCH_DIR|TOX21_DIR|REPORTS_DIR)$ ]]; then
      if [[ -n "${GRID_EXPERIMENT_ROOT:-}" && "$path" == ${GRID_EXPERIMENT_ROOT%/}* ]]; then
        if [[ -d "$path" ]]; then
          printf -v "$var_name" '%s' "$path"
          if (( idx > 0 )); then
            mjepa_log_warn "using existing read-only ${var_name}=$path"
          fi
          return 0
        fi
        continue
      fi
    fi
    if mjepa_try_dir "$path"; then
      printf -v "$var_name" '%s' "$path"
      if (( idx > 0 )); then
        mjepa_log_warn "falling back ${var_name}=$path"
      fi
      return 0
    fi
    tried+=("$path")
  done

  mjepa_log_error "unable to create $var_name (attempted: ${tried[*]:-none})"
  return 1
}

if [[ -n "${EXP_ID:-}" ]]; then
  EXPERIMENT_DIR="${EXPERIMENTS_ROOT%/}/${EXP_ID}"
else
  EXPERIMENT_DIR="${EXPERIMENTS_ROOT%/}"
fi

if [[ -n "${PRETRAIN_EXP_ID:-}" ]]; then
  PRETRAIN_EXPERIMENT_ROOT="${EXPERIMENTS_ROOT%/}/${PRETRAIN_EXP_ID}"
else
  PRETRAIN_EXPERIMENT_ROOT=""
fi

if [[ -n "${GRID_EXP_ID:-}" ]]; then
  GRID_EXPERIMENT_ROOT="${EXPERIMENTS_ROOT%/}/${GRID_EXP_ID}"
else
  GRID_EXPERIMENT_ROOT="${EXPERIMENT_DIR}"
fi

EXP_ROOT="$EXPERIMENT_DIR"

if [[ -n "${PRETRAIN_EXPERIMENT_ROOT:-}" ]]; then
  PRETRAIN_ARTIFACTS_DIR="${EXPERIMENTS_ROOT%/}/${PRETRAIN_EXP_ID}/artifacts"
  if (( FROZEN )) && [[ "$__ci_stage_role" == "dependent" ]]; then
    artifacts_default="${EXPERIMENT_DIR}/artifacts"
    : "${ARTIFACTS_DIR:=$artifacts_default}"
  else
    artifacts_default="${PRETRAIN_EXPERIMENT_ROOT}/artifacts"
    : "${ARTIFACTS_DIR:=$artifacts_default}"
  fi
else
  artifacts_default="${EXPERIMENT_DIR}/artifacts"
  : "${ARTIFACTS_DIR:=$artifacts_default}"
  : "${PRETRAIN_ARTIFACTS_DIR:=${ARTIFACTS_DIR}}"
fi

if [[ -n "${MJEPACI_STAGE_SHIM:-}" ]]; then
  if [[ -z "${WANDB_DIR:-}" ]]; then
    WANDB_DIR="${EXPERIMENT_DIR}/wandb"
  fi
  if [[ -z "${CACHE_DIR:-}" ]]; then
    CACHE_DIR="${EXPERIMENT_DIR}/cache/graphs_50k"
  fi
  default_cache_root="${EXPERIMENT_DIR}/cache/graphs_50k"
  default_wandb_root="${EXPERIMENT_DIR}/wandb"
else
  if [[ -z "${CACHE_DIR:-}" ]]; then
    CACHE_DIR="${DATA_ROOT}/cache/graphs_50k"
  fi
  if [[ -z "${WANDB_DIR:-}" ]]; then
    WANDB_DIR="${DATA_ROOT}/wandb"
  fi
  default_cache_root="${DATA_ROOT}/cache/graphs_50k"
  default_wandb_root="${DATA_ROOT}/wandb"
fi

ensure_dir_var CACHE_DIR "$default_cache_root" "${EXP_ID:+experiments/${EXP_ID}/}cache"
ensure_dir_var WANDB_DIR "$default_wandb_root" "${EXP_ID:+experiments/${EXP_ID}/}wandb"

if [[ -z "${SWEEP_CACHE_DIR:-}" ]]; then
  SWEEP_CACHE_DIR="$CACHE_DIR"
fi

export CACHE_DIR
export SWEEP_CACHE_DIR

# Allow cache directories to be overridden by env vars supplied by the workflow. If Grid_Dir is not set in yaml it uses cache dir
GRID_DIR_DEFAULT="${EXPERIMENT_DIR}/grid"
: "${GRID_DIR:=${GRID_CACHE_DIR:-$GRID_DIR_DEFAULT}}"

ensure_dir_var GRID_DIR "$GRID_DIR_DEFAULT" "${EXP_ID:+experiments/${EXP_ID}/}grid"

if [[ -n "${GRID_EXP_ID:-}" && "${GRID_EXP_ID}" != "${EXP_ID:-}" ]]; then
  GRID_SOURCE_DIR="${EXPERIMENTS_ROOT%/}/${GRID_EXP_ID}/grid"
elif [[ -n "${GRID_DIR:-}" ]]; then
  GRID_SOURCE_DIR="$GRID_DIR"
else
  GRID_SOURCE_DIR="${EXPERIMENT_DIR}/grid"
fi

ensure_dir_var ARTIFACTS_DIR "${PRETRAIN_EXPERIMENT_ROOT}/artifacts" "experiments/${PRETRAIN_EXP_ID}/artifacts"
ensure_dir_var PRETRAIN_ARTIFACTS_DIR "${ARTIFACTS_DIR}" "experiments/${PRETRAIN_EXP_ID}/artifacts"

if [[ -n "${PRETRAIN_EXP_ID:-}" ]]; then
  PRETRAIN_STATE_FILE_CANONICAL="${EXPERIMENTS_ROOT}/${PRETRAIN_EXP_ID}/pretrain_state.json"
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
    PRETRAIN_DIR="${PRETRAIN_EXPERIMENT_ROOT}/pretrain"
  fi
fi

ensure_dir_var PRETRAIN_DIR "${PRETRAIN_EXPERIMENT_ROOT}/pretrain" "experiments/${PRETRAIN_EXP_ID}/pretrain"

if [[ -z "${PRETRAIN_MANIFEST:-}" ]]; then
  PRETRAIN_MANIFEST="${PRETRAIN_ARTIFACTS_DIR}/encoder_manifest.json"
fi

if [[ -z "${PRETRAIN_ENCODER_PATH:-}" ]]; then
  PRETRAIN_ENCODER_PATH="${PRETRAIN_DIR}/encoder.pt"
fi

if [[ -z "${PRETRAIN_TOX21_ENV:-}" ]]; then
  PRETRAIN_TOX21_ENV="${PRETRAIN_EXPERIMENT_ROOT}/tox21_gate.env"
fi

GRID_FINETUNE_DEFAULT="${EXPERIMENT_DIR}/finetune"
GRID_BENCH_DEFAULT="${EXPERIMENT_DIR}/bench"
GRID_TOX21_DEFAULT="${EXPERIMENT_DIR}/tox21"
GRID_REPORTS_DEFAULT="${EXPERIMENT_DIR}/report"

: "${FINETUNE_DIR:=${FINETUNE_CACHE_DIR:-$GRID_FINETUNE_DEFAULT}}"
: "${BENCH_DIR:=${BENCH_CACHE_DIR:-$GRID_BENCH_DEFAULT}}"
# Keep tox21 outputs under a dedicated subdirectory so stage stamps don't
# collide with pretrain's cache stamp when both default to EXPERIMENT_DIR.
: "${TOX21_DIR:=${TOX21_CACHE_DIR:-$GRID_TOX21_DEFAULT}}"
: "${REPORTS_DIR:=${REPORTS_CACHE_DIR:-$GRID_REPORTS_DEFAULT}}"

ensure_dir_var FINETUNE_DIR "$GRID_FINETUNE_DEFAULT" "${EXP_ID:+experiments/${EXP_ID}/}finetune"
ensure_dir_var BENCH_DIR "$GRID_BENCH_DEFAULT" "${EXP_ID:+experiments/${EXP_ID}/}bench"
ensure_dir_var TOX21_DIR "$GRID_TOX21_DEFAULT" "${EXP_ID:+experiments/${EXP_ID}/}tox21"
ensure_dir_var REPORTS_DIR "$GRID_REPORTS_DEFAULT" "${EXP_ID:+experiments/${EXP_ID}/}report"
ensure_dir_var LOG_DIR "${APP_DIR}/logs" "logs"

for dir_path in \
  "$CACHE_DIR" "$GRID_DIR" "$PRETRAIN_DIR" "$FINETUNE_DIR" "$BENCH_DIR" \
  "$TOX21_DIR" "$REPORTS_DIR" "$LOG_DIR" "$WANDB_DIR" "$ARTIFACTS_DIR" \
  "$PRETRAIN_ARTIFACTS_DIR" "$EXPERIMENT_DIR" "$PRETRAIN_EXPERIMENT_ROOT" \
  "$GRID_EXPERIMENT_ROOT"; do
  [[ -z "$dir_path" ]] && continue
  if (( FROZEN )) && [[ -n "${PRETRAIN_EXPERIMENT_ROOT:-}" ]] && [[ "$dir_path" == ${PRETRAIN_EXPERIMENT_ROOT%/}* ]]; then
    if [[ ! -d "$dir_path" ]]; then
      mjepa_log_error "expected frozen directory missing: $dir_path"
      exit 1
    fi
    continue
  fi
  if (( FROZEN )) && [[ -n "${GRID_EXPERIMENT_ROOT:-}" ]] && [[ "$dir_path" == ${GRID_EXPERIMENT_ROOT%/}* ]]; then
    if [[ ! -d "$dir_path" ]]; then
      mjepa_log_error "expected frozen directory missing: $dir_path"
      exit 1
    fi
    continue
  fi
  mkdir -p "$dir_path"
done

if (( FROZEN )) && [[ -n "${PRETRAIN_ARTIFACTS_DIR:-}" ]] && [[ ! -d "$PRETRAIN_ARTIFACTS_DIR" ]]; then
  mjepa_log_error "missing frozen artifacts directory: ${PRETRAIN_ARTIFACTS_DIR}"
  exit 1
fi
export GRID_DIR PRETRAIN_DIR FINETUNE_DIR BENCH_DIR TOX21_DIR REPORTS_DIR LOG_DIR \
  GRID_EXPERIMENT_ROOT PRETRAIN_EXPERIMENT_ROOT GRID_SOURCE_DIR
export EXPERIMENT_DIR ARTIFACTS_DIR EXP_ROOT EXPERIMENTS_ROOT
export PRETRAIN_MANIFEST PRETRAIN_EXP_ID PRETRAIN_ARTIFACTS_DIR \
  PRETRAIN_STATE_FILE PRETRAIN_STATE_FILE_CANONICAL PRETRAIN_STATE_FILE_LEGACY \
  PRETRAIN_ENCODER_PATH PRETRAIN_TOX21_ENV EXP_ID GRID_EXP_ID FREEZE_MARKER \
  FROZEN ORIGINAL_PRETRAIN_EXP_ID

if [[ -z "${MJEPACI_COMMIT_SHA:-}" ]]; then
  if git -C "${APP_DIR}" rev-parse HEAD >/dev/null 2>&1; then
    MJEPACI_COMMIT_SHA="$(git -C "${APP_DIR}" rev-parse HEAD 2>/dev/null)"
  else
    MJEPACI_COMMIT_SHA="unknown"
  fi
fi
export MJEPACI_COMMIT_SHA

ci_print_env_diag() {
  local stage_bin_value="${1:-${STAGE_BIN:-<unset>}}"
  local stage_name="${MJEPACI_STAGE:-<unset>}"
  local out_dir="<unset>"
  if [[ "$stage_name" != "<unset>" ]] && declare -F stage_dir >/dev/null 2>&1; then
    out_dir="$(stage_dir "$stage_name")"
  fi
  local grid_read="${GRID_SOURCE_DIR:-${GRID_DIR:-<unset>}}"
  echo "[ci] STAGE=${stage_name} EXP_ID=${EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>} GRID_EXP_ID=${GRID_EXP_ID:-<unset>} FROZEN=${FROZEN:-0} STAGE_BIN=${stage_bin_value}" >&2
  echo "     READ: ARTIFACTS_DIR=${PRETRAIN_ARTIFACTS_DIR:-<unset>} GRID_DIR=${grid_read}" >&2
  echo "     WRITE: OUT_DIR=${out_dir} EXPERIMENT_DIR=${EXPERIMENT_DIR:-<unset>}" >&2
}

# --- micromamba bootstrap ---
_bootstrap_micromamba() {
  local prefix="${1:-${MAMBA_ROOT_PREFIX:-$HOME/micromamba}}"
  local arch="$(uname -m)"
  local channel=""

  case "$arch" in
    x86_64|amd64) channel="linux-64" ;;
    aarch64|arm64) channel="linux-aarch64" ;;
    *)
      echo "[ensure_micromamba] unsupported architecture: $arch" >&2
      return 1
      ;;
  esac

  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    echo "[ensure_micromamba] curl or wget is required to download micromamba" >&2
    return 1
  fi

  if ! command -v tar >/dev/null 2>&1; then
    echo "[ensure_micromamba] tar is required to extract micromamba" >&2
    return 1
  fi

  mkdir -p "$prefix/bin"

  echo "[ensure_micromamba] bootstrapping micromamba into ${prefix}" >&2

  local tmp
  tmp="$(mktemp -d)"
  local archive="${tmp}/micromamba.tar.bz2"
  local url="https://micro.mamba.pm/api/micromamba/${channel}/latest"
  if command -v curl >/dev/null 2>&1; then
    if ! curl -fsSL "$url" -o "$archive"; then
      echo "[ensure_micromamba] failed to download micromamba payload" >&2
      rm -rf "$tmp"
      return 1
    fi
  else
    if ! wget -qO "$archive" "$url"; then
      echo "[ensure_micromamba] failed to download micromamba payload" >&2
      rm -rf "$tmp"
      return 1
    fi
  fi

  if ! tar -xjf "$archive" -C "$prefix"; then
    echo "[ensure_micromamba] failed to extract micromamba payload" >&2
    rm -rf "$tmp"
    return 1
  fi

  rm -rf "$tmp"

  if [[ ! -x "$prefix/bin/micromamba" ]]; then
    echo "[ensure_micromamba] micromamba binary missing after bootstrap" >&2
    return 1
  fi

  return 0
}

ensure_micromamba() {
  local prefix="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
  local candidate="${MMBIN:-}"

  if [[ -n "$candidate" && -x "$candidate" ]]; then
    :
  elif command -v micromamba >/dev/null 2>&1; then
    candidate="$(command -v micromamba)"
  else
    candidate="${prefix}/bin/micromamba"
    if [[ ! -x "$candidate" ]]; then
      if ! _bootstrap_micromamba "$prefix"; then
        echo "micromamba not found" >&2
        return 1
      fi
    fi
  fi

  MAMBA_ROOT_PREFIX="$prefix"
  export MAMBA_ROOT_PREFIX
  MMBIN="$candidate"
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
  local -a grid_roots=()
  if [[ -n "${GRID_SOURCE_DIR:-}" ]]; then
    grid_roots+=("${GRID_SOURCE_DIR%/}")
  fi
  if [[ -n "${GRID_DIR:-}" ]]; then
    local grid_dir_base="${GRID_DIR%/}"
    local already=0
    local existing_root
    for existing_root in "${grid_roots[@]}"; do
      if [[ "$existing_root" == "$grid_dir_base" ]]; then
        already=1
        break
      fi
    done
    if (( ! already )); then
      grid_roots+=("$grid_dir_base")
    fi
  fi
  if [[ ${#grid_roots[@]} -eq 0 ]]; then
    grid_roots+=("${EXPERIMENT_DIR%/}/grid")
  fi

  local grid_display="${grid_roots[0]}"
  [[ -n "$grid_display" ]] || grid_display='<unset>'
  local best_hint phase2_hint
  if [[ "$grid_display" == '<unset>' ]]; then
    best_hint='<unset>'
    phase2_hint='<unset>'
  else
    best_hint="${grid_display}/best_grid_config.json"
    phase2_hint="${grid_display}/phase2_export/best_grid_config.json"
  fi

  local -a winner_candidates=()
  if [[ -n "${BENCH_WINNER_JSON:-}" ]]; then
    winner_candidates+=("${BENCH_WINNER_JSON}")
  fi
  local grid_root
  for grid_root in "${grid_roots[@]}"; do
    [[ -n "$grid_root" ]] || continue
    winner_candidates+=("${grid_root}/best_grid_config.json")
    winner_candidates+=("${grid_root}/phase2_export/best_grid_config.json")
  done

  local cfg="" candidate=""
  for candidate in "${winner_candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ -r "$candidate" ]]; then
      cfg="$candidate"
      break
    fi
  done

  if [[ -z "$cfg" ]]; then
    case "$stage" in
      bench|benchmark)
        echo "[fatal] bench winner not found." >&2
        echo "checked: ${BENCH_WINNER_JSON:-<unset>}, ${best_hint}, ${phase2_hint}" >&2
        echo "hint: set GRID_EXP_ID=${GRID_EXP_ID:-<unset>} or run/export Phase-2" >&2
        return 2
        ;;
      pretrain|finetune|tox21)
        echo "[fatal] $stage needs winner but missing: ${best_hint}" >&2
        return 2
        ;;
      *)
        return 0
        ;;
    esac
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
_MISSING = object()

def _get(key, default=_MISSING):
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

def _lookup_with_aliases(key):
    """Return the first non-missing value for key or its underscore/hyphen aliases."""
    value = _get(key)
    if value is not _MISSING:
        return value
    candidates = []
    if "_" in key:
        candidates.append(key.replace("_", "-"))
    if "-" in key:
        candidates.append(key.replace("-", "_"))
    for alias in candidates:
        value = _get(alias)
        if value is not _MISSING:
            return value
    return _MISSING

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
    "sample_unlabeled": "--sample-unlabeled",
}

finetune_only = {
    "finetune_batch_size": "--batch-size",
    "finetune_epochs": "--epochs",
    "metric": "--metric",
    "patience": "--patience",
    "jepa_encoder": "--jepa-encoder",
    "max_finetune_batches": "--max-finetune-batches",
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

# Build a flat cfg the rest of the code expects. Start from all mapped keys so
# newly supported flags flow automatically, then ensure legacy singular knobs
# remain available even if a future refactor drops them from the stage maps.
cfg = {}
known_keys = set()
for mapping in maps.values():
    known_keys.update(mapping.keys())
known_keys.update({"training_method", "method", "learning_rate", "lr"})

for key in sorted(known_keys):
    value = _lookup_with_aliases(key)
    if value is _MISSING or value is None:
        continue
    cfg[key] = value

# Backfill historical singular fields that other tooling expects, regardless of
# whether they are still covered by the dynamic maps above.
legacy_keys = [
    "gnn_type",
    "hidden_dim",
    "num_layers",
    "ema_decay",
    "contiguity",
    "add_3d",
    "lr",
    "learning_rate",
    "temperature",
    "training_method",
    "method",
    "pretrain_epochs",
    "finetune_epochs",
]

for key in legacy_keys:
    if key in cfg:
        continue
    value = _lookup_with_aliases(key)
    if value is _MISSING or value is None:
        continue
    cfg[key] = value

# normalise: learning_rate → lr (benchmark/help only exposes --lr)
if "lr" not in cfg and "learning_rate" in cfg:
    cfg["lr"] = cfg["learning_rate"]

# normalise: method → contrastive flag
tm = cfg.get("training_method") or cfg.get("method")
if isinstance(tm, str) and tm.lower().startswith("con"):
    cfg["contrastive"] = True

# Build the skip set from env (accept space- or comma-separated)
_raw = os.environ.get("BESTCFG_SKIP", "")
# accept comma- or space-separated values and normalize
skip = {s.strip() for s in _raw.replace(",", " ").split() if s.strip()}

sorted_skip = sorted(skip)
joined_skip = ", ".join(sorted_skip)
print(f"[bestcfg] skip=[{joined_skip}]" if joined_skip else "[bestcfg] skip=[]", file=sys.stderr)

# Policy: structural winners (gnn_type, hidden_dim, num_layers) should not be
# skipped in normal flows. Use BESTCFG_NO_EPOCHS=1 to drop epochs when needed.
structural = {"hidden_dim", "num_layers", "gnn_type"}
structural_hits = sorted(structural.intersection(skip))
if structural_hits:
    print(
        "[bestcfg][warn] structural winner(s) skipped: "
        + ", ".join(structural_hits)
        + "; this can cause shape mismatches.",
        file=sys.stderr,
    )

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
