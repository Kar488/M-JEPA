#!/usr/bin/env bash
#set -x   # bash prints each command before executing
set -euo pipefail

ci_setup_vast_ssh_key() {
  local key="${SSH_KEY:-}"
  [[ -n "$key" ]] || return 0

  local home_dir="${HOME:-/root}"
  local ssh_dir="${home_dir%/}/.ssh"
  local key_path="${ssh_dir}/vast_key"

  mkdir -p "$ssh_dir"
  # Ensure the key has a trailing newline so ssh-add can parse it correctly.
  if [[ "$key" != *$'\n' ]]; then
    printf '%s\n' "$key" >"$key_path"
  else
    printf '%s' "$key" >"$key_path"
  fi
  chmod 600 "$key_path" 2>/dev/null || true

  if command -v ssh-agent >/dev/null 2>&1; then
    if [[ -z "${SSH_AUTH_SOCK:-}" || ! -S "${SSH_AUTH_SOCK}" ]]; then
      eval "$(ssh-agent -s)" >/dev/null 2>&1 || true
    fi
  fi

  if command -v ssh-add >/dev/null 2>&1; then
    if ! ssh-add -L 2>/dev/null | grep -F "$key_path" >/dev/null 2>&1; then
      SSH_ASKPASS="${SSH_ASKPASS:-/bin/true}" DISPLAY="${DISPLAY:-}" ssh-add "$key_path" >/dev/null 2>&1 || true
    fi
  fi

  export VAST_SSH_KEY_PATH="$key_path"
}

ci_phase2_refresh_lineage_bindings() {
  local new_pretrain="${1:-}"
  local new_grid="${2:-}"
  local old_pretrain="${3:-}"
  local old_grid="${4:-}"

  local root="${EXPERIMENTS_ROOT%/}"
  local old_pretrain_root=""
  local old_grid_root=""

  if [[ -n "$old_pretrain" ]]; then
    old_pretrain_root="${root}/${old_pretrain}"
  fi
  if [[ -n "$old_grid" ]]; then
    old_grid_root="${root}/${old_grid}/grid"
  fi

  if [[ -n "$new_pretrain" ]]; then
    local new_pretrain_root="${root}/${new_pretrain}"
    PRETRAIN_EXP_ID="$new_pretrain"
    export PRETRAIN_EXP_ID
    PRETRAIN_STATE_ID="$new_pretrain"
    export PRETRAIN_STATE_ID

    PRETRAIN_EXPERIMENT_ROOT="$new_pretrain_root"
    export PRETRAIN_EXPERIMENT_ROOT

    local default_state="${new_pretrain_root}/pretrain_state.json"
    PRETRAIN_STATE_FILE_CANONICAL="$default_state"
    export PRETRAIN_STATE_FILE_CANONICAL
    local current_state_file="${PRETRAIN_STATE_FILE:-}"
    if [[ -z "$current_state_file" || ( -n "$old_pretrain_root" && "${current_state_file%/}" == "${old_pretrain_root}/pretrain_state.json" ) ]]; then
      PRETRAIN_STATE_FILE="$default_state"
      export PRETRAIN_STATE_FILE
    fi

    local default_artifacts="${new_pretrain_root}/artifacts"
    local current_artifacts="${ARTIFACTS_DIR:-}"
    if [[ -z "$current_artifacts" || ( -n "$old_pretrain_root" && "${current_artifacts%/}" == "${old_pretrain_root}/artifacts" ) ]]; then
      ARTIFACTS_DIR="$default_artifacts"
      export ARTIFACTS_DIR
    fi
    local current_pretrain_artifacts="${PRETRAIN_ARTIFACTS_DIR:-}"
    if [[ -z "$current_pretrain_artifacts" || ( -n "$old_pretrain_root" && "${current_pretrain_artifacts%/}" == "${old_pretrain_root}/artifacts" ) ]]; then
      PRETRAIN_ARTIFACTS_DIR="$default_artifacts"
      export PRETRAIN_ARTIFACTS_DIR
    fi

    local default_pretrain_dir="${new_pretrain_root}/pretrain"
    local current_pretrain_dir="${PRETRAIN_DIR:-}"
    if [[ -z "$current_pretrain_dir" || ( -n "$old_pretrain_root" && "${current_pretrain_dir%/}" == "${old_pretrain_root}/pretrain" ) ]]; then
      PRETRAIN_DIR="$default_pretrain_dir"
      export PRETRAIN_DIR
    fi

    local default_manifest="${default_artifacts}/encoder_manifest.json"
    local current_manifest="${PRETRAIN_MANIFEST:-}"
    if [[ -z "$current_manifest" || ( -n "$old_pretrain_root" && "$current_manifest" == "${old_pretrain_root}/artifacts/encoder_manifest.json" ) ]]; then
      PRETRAIN_MANIFEST="$default_manifest"
      export PRETRAIN_MANIFEST
    fi

    local default_encoder="${default_pretrain_dir}/encoder.pt"
    local current_encoder="${PRETRAIN_ENCODER_PATH:-}"
    if [[ -z "$current_encoder" || ( -n "$old_pretrain_root" && "$current_encoder" == "${old_pretrain_root}/pretrain/encoder.pt" ) ]]; then
      PRETRAIN_ENCODER_PATH="$default_encoder"
      export PRETRAIN_ENCODER_PATH
    fi

    local default_tox="${new_pretrain_root}/tox21_gate.env"
    local current_tox21="${PRETRAIN_TOX21_ENV:-}"
    if [[ -z "$current_tox21" || ( -n "$old_pretrain_root" && "$current_tox21" == "${old_pretrain_root}/tox21_gate.env" ) ]]; then
      PRETRAIN_TOX21_ENV="$default_tox"
      export PRETRAIN_TOX21_ENV
    fi

    FREEZE_MARKER="${new_pretrain_root}/bench/encoder_frozen.ok"
    export FREEZE_MARKER
    if declare -F ci_refresh_freeze_state >/dev/null 2>&1; then
      ci_refresh_freeze_state "$FREEZE_MARKER"
    fi
    ORIGINAL_PRETRAIN_EXP_ID="$PRETRAIN_EXP_ID"
    export ORIGINAL_PRETRAIN_EXP_ID
  fi

  if [[ -n "$new_grid" ]]; then
    local new_grid_root="${root}/${new_grid}"
    GRID_EXP_ID="$new_grid"
    export GRID_EXP_ID
    GRID_EXPERIMENT_ROOT="$new_grid_root"
    export GRID_EXPERIMENT_ROOT

    local candidate_source="${new_grid_root}/grid"
    local current_grid_source="${GRID_SOURCE_DIR:-}"
    if [[ -z "$current_grid_source" || ( -n "$old_grid_root" && "${current_grid_source%/}" == "${old_grid_root%/}" ) ]]; then
      GRID_SOURCE_DIR="$candidate_source"
      export GRID_SOURCE_DIR
    fi
  fi
}

grace_marker() {
  local stage="${1:?stage}" log_dir="${2:-${LOG_DIR:-}}"
  if [[ -z "$log_dir" ]]; then
    log_dir="${APP_DIR:-.}/logs"
  fi
  echo "${log_dir}/${stage}.graceful_stop"
}
mark_graceful_stop() {
  local stage="${1:?stage}" log_dir="${2:-${LOG_DIR:-}}"
  if [[ -z "$log_dir" ]]; then
    log_dir="${APP_DIR:-.}/logs"
  fi
  mkdir -p "$log_dir"
  : >"$(grace_marker "$stage" "$log_dir")"
}
was_graceful_stop() {
  local stage="${1:?stage}" log_dir="${2:-${LOG_DIR:-}}"
  if [[ -z "$log_dir" ]]; then
    log_dir="${APP_DIR:-.}/logs"
  fi
  [[ -f "$(grace_marker "$stage" "$log_dir")" ]]
}

clear_graceful_stop() {
  local stage="${1:?stage}" log_dir="${2:-${LOG_DIR:-}}"
  local marker
  marker="$(grace_marker "$stage" "$log_dir")"
  if [[ -f "$marker" ]]; then
    rm -f "$marker"
  fi
}

declare -a MJEPACI_FORCE_RERUN_STAGES=()
if [[ -n "${FORCE_RERUN:-}" ]]; then
  IFS=',' read -r -a __mjepa_force_tokens <<<"${FORCE_RERUN}"
  for token in "${__mjepa_force_tokens[@]}"; do
    token="${token//[\"\']/}"
    token="${token//[[:space:]]/}"
    token="${token,,}"
    [[ -n "$token" ]] && MJEPACI_FORCE_RERUN_STAGES+=("$token")
  done
  unset __mjepa_force_tokens
fi

stage_is_forced() {
  local stage="${1,,}"
  for token in "${MJEPACI_FORCE_RERUN_STAGES[@]}"; do
    case "$token" in
      '' ) continue ;;
      all|\*) return 0 ;;
      *'\*')
        local pattern="${token}"; pattern="${pattern//\*/.*}"
        if [[ "$stage" =~ ^${pattern}$ ]]; then
          return 0
        fi
        ;;
      *)
        if [[ "$stage" == "$token" ]]; then
          return 0
        fi
        if [[ "$stage" == ${token}_* ]]; then
          return 0
        fi
        ;;
    esac
  done
  return 1
}

stage_state_file() {
  local dir="${1:?stage_dir}"; echo "${dir%/}/stage_state.json"
}

stage_compute_hash() {
  local stage="$1"; shift || true
  local py
  if py=$(python_bin 2>/dev/null); then
    "$py" - "$stage" "$@" <<'PY'
import hashlib
import json
import sys

stage = sys.argv[1]
items = sys.argv[2:]
payload = {"stage": stage, "items": items}
blob = json.dumps(payload, sort_keys=True).encode("utf-8")
print(hashlib.sha256(blob).hexdigest())
PY
  else
    local concat="$stage"
    local item
    for item in "$@"; do
      concat+="|$item"
    done
    printf '%s' "$concat" | sha256sum | awk '{print $1}'
  fi
}

stage_state_load() {
  local file="${1:?state_file}" output=""
  STAGE_STATE_STAGE=""
  STAGE_STATE_COMMIT=""
  STAGE_STATE_CONFIG_HASH=""
  STAGE_STATE_DATA_HASH=""
  STAGE_STATE_CREATED_AT=""
  [[ -f "$file" ]] || return 1
  local py
  py=$(python_bin 2>/dev/null) || return 1
  if ! output="$("$py" - "$file" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
except Exception:
    sys.exit(1)

for key in ("stage", "commit_sha", "config_hash", "data_hash", "created_at"):
    value = data.get(key)
    if value is None:
        value = ""
    if isinstance(value, (dict, list)):
        continue
    print(f"{key}={value}")
PY
  )"; then
    return 1
  fi
  local line key value
  while IFS='=' read -r key value; do
    case "$key" in
      stage) STAGE_STATE_STAGE="$value" ;;
      commit_sha) STAGE_STATE_COMMIT="$value" ;;
      config_hash) STAGE_STATE_CONFIG_HASH="$value" ;;
      data_hash) STAGE_STATE_DATA_HASH="$value" ;;
      created_at) STAGE_STATE_CREATED_AT="$value" ;;
    esac
  done <<<"$output"
  [[ -n "$STAGE_STATE_STAGE" ]] || return 1
  return 0
}

stage_state_write() {
  local stage="${1:?stage}" dir="${2:?dir}" config_hash="${3:-}" data_hash="${4:-}"
  local inputs_file="${5:-}" deps_file="${6:-}" outputs_file="${7:-}"
  local state_path
  state_path="$(stage_state_file "$dir")"
  local -a py_cmd=()
  if py=$(python_bin 2>/dev/null); then
    py_cmd=("$py")
  else
    ensure_micromamba
    py_cmd=("$MMBIN" run -n mjepa python)
  fi
  "${py_cmd[@]}" - "$state_path" "$stage" "${MJEPACI_COMMIT_SHA:-unknown}" \
    "$config_hash" "$data_hash" "${inputs_file:-}" "${deps_file:-}" \
    "${outputs_file:-}" "${EXP_ID:-}" "${GRID_EXP_ID:-}" "${PRETRAIN_EXP_ID:-}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    state_path,
    stage,
    commit,
    config_hash,
    data_hash,
    inputs_path,
    deps_path,
    outputs_path,
    exp_id,
    grid_id,
    pretrain_id,
) = sys.argv[1:12]


def read_lines(path: str) -> list[str]:
    if not path or path == "-":
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle if line.strip()]
    except FileNotFoundError:
        return []


payload = {
    "stage": stage,
    "commit_sha": commit,
    "config_hash": config_hash,
    "data_hash": data_hash or None,
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "inputs": {
        "args": read_lines(inputs_path),
        "dependencies": read_lines(deps_path),
    },
    "outputs": {
        "paths": read_lines(outputs_path),
        "stage_dir": os.path.abspath(os.path.dirname(state_path)),
    },
    "origin_ids": {
        "exp_id": exp_id or None,
        "grid_exp_id": grid_id or None,
        "pretrain_exp_id": pretrain_id or None,
    },
}

tmp_path = state_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp_path, state_path)
PY
}

python_inline() {
  local -a cmd=()
  if py=$(python_bin 2>/dev/null); then
    cmd=("$py")
  else
    ensure_micromamba
    cmd=("$MMBIN" run -n mjepa python)
  fi
  "${cmd[@]}" - "$@"
}

# requires: common.sh (ensure_micromamba, build_argv_from_yaml, expand_array_vars, best_config_args)
# ---------- stage → dirs / subcommands / dependencies ----------
stage_dir() {
  local stage="$1"
  case "$stage" in
    pretrain) echo "$PRETRAIN_DIR" ;;
    finetune) echo "$FINETUNE_DIR" ;;
    grid) echo "$GRID_DIR" ;;
    phase1) echo "${GRID_DIR}/phase1" ;;
    phase2_sweep|phase2_recheck|phase2_export) echo "${GRID_DIR}/$stage" ;;
    bench|benchmark) echo "$BENCH_DIR" ;;
    tox21) echo "$TOX21_DIR" ;;
    report) echo "$REPORTS_DIR" ;;
    *) echo "$EXP_ROOT/$stage" ;;
  esac
}
stage_subcmd() {
  case "$1" in
    grid) echo "grid-search" ;;
    bench) echo "benchmark" ;;
    *) echo "$1" ;;
  esac
}
stage_needs() {
  local s="$1"
  local base=(
    "$APP_DIR/scripts/train_jepa.py"
    "$APP_DIR/scripts/ci/train_jepa_ci.yml"
    "$APP_DIR/scripts/ci/common.sh"
    "$APP_DIR/scripts/ci/run-${s}.sh"
  )
  
  case "$s" in
    grid)   printf '%s\n' "${base[@]}" ;;
    pretrain) printf '%s\n' "${base[@]}" "${GRID_SOURCE_DIR}/best_grid_config.json" ;;
    finetune)
      local encoder_dep
      encoder_dep="$(resolve_encoder_checkpoint)"
      printf '%s\n' "${base[@]}" "${GRID_SOURCE_DIR}/best_grid_config.json" "$encoder_dep"
      ;;
    bench|benchmark) printf '%s\n' "${base[@]}" "${GRID_SOURCE_DIR}/best_grid_config.json" "$FINETUNE_DIR/seed_0/ft_best.pt" ;;
    tox21) printf '%s\n' "${base[@]}" "${GRID_SOURCE_DIR}/best_grid_config.json" "$APP_DIR/scripts/ci/data/tox21/data.csv" ;;
    phase2_sweep)
      printf '%s\n' \
        "${base[@]}" \
        "${GRID_SOURCE_DIR}/phase2_sweep_id.txt"
      ;;
    phase2_recheck)
      local sweep_stamp
      sweep_stamp="$(stage_dir phase2_sweep)"
      printf '%s\n' \
        "${base[@]}" \
        "$APP_DIR/scripts/ci/recheck_topk_from_wandb.py" \
        "${GRID_SOURCE_DIR}/phase2_sweep_id.txt" \
        "${sweep_stamp}/.stamp"
      ;;
    phase2_export)
      local recheck_stamp
      recheck_stamp="$(stage_dir phase2_recheck)"
      printf '%s\n' \
        "${base[@]}" \
        "${GRID_SOURCE_DIR}/best_grid_config.json" \
        "$GRID_DIR/recheck_summary.json" \
        "${recheck_stamp}/.stamp"
      ;;
    report)
      printf '%s\n' \
        "$APP_DIR/reports/build_wandb_report.py" \
        "$APP_DIR/reports/discover_schema.py" \
        "$APP_DIR/reports/wandb_utils.py" \
        "$APP_DIR/reports/plots_pretrain.py" \
        "$APP_DIR/reports/plots_regression.py" \
        "$APP_DIR/reports/plots_classification.py" \
        "$APP_DIR/reports/plots_repr.py" \
        "$APP_DIR/reports/plots_tox21.py" \
        "$APP_DIR/reports/plots_compare.py"
      ;;
    *) printf '%s\n' "${base[@]}" ;;
  esac
}

# ---------- phase-2 helpers ----------
phase2_step_diag() {
  local step="$1"
  echo "[ci] STEP=${step} EXP_ID=${EXP_ID:-<unset>} GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>} GRID_DIR=${GRID_DIR:-<unset>} GRID_SOURCE_DIR=${GRID_SOURCE_DIR:-<unset>} EXP_ROOT=${EXP_ROOT:-<unset>}" >&2
}

phase2_promote_local_grid() {
  local step_label="${1:-phase2_recheck}"
  local best_path="${GRID_DIR:-}/best_grid_config.json"
  [[ -n "${GRID_DIR:-}" && -f "$best_path" ]] || return 0

  local normalized_source="${GRID_SOURCE_DIR:-}"
  if [[ "${normalized_source%/}" != "${GRID_DIR%/}" ]]; then
    GRID_SOURCE_DIR="$GRID_DIR"
    export GRID_SOURCE_DIR
    if [[ -n "${EXP_ID:-}" ]]; then
      GRID_EXP_ID="${EXP_ID}"
      export GRID_EXP_ID
    fi
    echo "[${step_label}] promoting GRID_SOURCE_DIR=${GRID_SOURCE_DIR} GRID_EXP_ID=${GRID_EXP_ID:-<unset>}" >&2
  fi
}

phase2_sync_grid_artifacts() {
  local step_label="${1:-phase2}"
  local source="${GRID_SOURCE_DIR:-}"
  local dest="${GRID_DIR:-}"
  if [[ -z "$dest" ]]; then
    return 0
  fi

  local dest_dir="${dest%/}"
  mkdir -p "$dest_dir" || return 0

  if [[ -z "$source" ]]; then
    return 0
  fi

  local source_dir="${source%/}"
  local -a pairs=(
    "${source_dir}/phase2_sweep_id.txt" "${dest_dir}/phase2_sweep_id.txt"
    "${source_dir}/best_grid_config.json" "${dest_dir}/best_grid_config.json"
    "${source_dir}/recheck_summary.json" "${dest_dir}/recheck_summary.json"
  )

  local idx=0
  while (( idx < ${#pairs[@]} )); do
    local src="${pairs[$idx]}"
    local dst="${pairs[$((idx+1))]}"
    ((idx+=2))

    if [[ "$src" == "$dst" ]]; then
      continue
    fi
    if [[ ! -f "$src" ]]; then
      continue
    fi
    mkdir -p "$(dirname "$dst")" || continue
    if ! cmp -s "$src" "$dst" 2>/dev/null; then
      cp -f "$src" "$dst"
      echo "[${step_label}] synced $(basename "$src") -> ${dst}" >&2
    fi
  done

  local sentinel_src="${source_dir}/phase2_recheck/recheck_done.ok"
  local sentinel_dst="$(stage_dir phase2_recheck)/recheck_done.ok"
  if [[ -f "$sentinel_src" && "$sentinel_src" != "$sentinel_dst" ]]; then
    mkdir -p "$(dirname "$sentinel_dst")" || return 0
    if ! cmp -s "$sentinel_src" "$sentinel_dst" 2>/dev/null; then
      cp -f "$sentinel_src" "$sentinel_dst"
      echo "[${step_label}] synced $(basename "$sentinel_src") -> ${sentinel_dst}" >&2
    fi
  fi
}

phase2_promote_grid_artifacts_single() {
  local step_label="$1" dest="$2"
  shift 2 || true
  local candidate
  local chosen=""
  local chosen_mtime=-1

  for candidate in "$@"; do
    [[ -n "$candidate" && -f "$candidate" ]] || continue
    local candidate_mtime
    candidate_mtime=$(stat -c %Y "$candidate" 2>/dev/null) || candidate_mtime=0
    if [[ -z "$chosen" || $candidate_mtime -gt $chosen_mtime ]]; then
      chosen="$candidate"
      chosen_mtime=$candidate_mtime
    fi
  done

  [[ -n "$chosen" ]] || return 0

  if [[ -f "$dest" && cmp -s "$chosen" "$dest" 2>/dev/null ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$dest")" || return 0

  if cp -f "$chosen" "$dest"; then
    echo "[${step_label}] promoted $(basename "$chosen") -> ${dest}" >&2
  fi
  return 0
}

phase2_promote_grid_artifacts() {
  local step_label="${1:-phase2}"
  local grid_root="${GRID_DIR:-}"
  [[ -n "$grid_root" ]] || return 0
  grid_root="${grid_root%/}"

  phase2_promote_grid_artifacts_single "$step_label" \
    "${grid_root}/best_grid_config.json" \
    "${grid_root}/phase2_export/best_grid_config.json" \
    "${grid_root}/phase2_export/stage-outputs/best_grid_config.json" \
    "${grid_root}/phase2_recheck/stage-outputs/best_grid_config.json" \
    "${grid_root}/phase2_sweep/stage-outputs/best_grid_config.json"

  phase2_promote_grid_artifacts_single "$step_label" \
    "${grid_root}/recheck_summary.json" \
    "${grid_root}/phase2_export/recheck_summary.json" \
    "${grid_root}/phase2_export/stage-outputs/recheck_summary.json" \
    "${grid_root}/phase2_recheck/recheck_summary.json" \
    "${grid_root}/phase2_recheck/stage-outputs/recheck_summary.json"
}

restore_env_var() {
  local name="$1" previous="$2"
  if [[ -n "$previous" ]]; then
    printf -v "$name" '%s' "$previous"
    export "$name"
  else
    unset "$name"
  fi
}

run_phase2_sweep_stage() {
  local dir="$1" step="$2"

  phase2_step_diag "$step"

  local sweep_id_file="${GRID_SOURCE_DIR}/phase2_sweep_id.txt"
  if [[ ! -f "$sweep_id_file" ]]; then
    echo "[$step][fatal] sweep id file not found: $sweep_id_file (GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>}). Set GRID_EXP_ID=<id> to reuse an existing sweep or rerun phase2_sweep." >&2
    return 2
  fi

  local sweep_id
  sweep_id="$(<"$sweep_id_file")"
  export WANDB_SWEEP_ID2="$sweep_id"
  export SWEEP_ID="${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_SWEEP_ID2}"

  local prev_log_dir="${LOG_DIR:-}"
  local step_log_dir="${dir}/logs"
  mkdir -p "$step_log_dir"
  export LOG_DIR="$step_log_dir"

  local prev_wall="${HARD_WALL_MINS:-}"
  local sweep_wall="${PHASE2_SWEEP_WALL_MINS:-920}"
  export HARD_WALL_MINS="$sweep_wall"

  mapfile -t GRID_VISIBLE_GPUS < <(visible_gpu_ids)
  local gpu_count="${#GRID_VISIBLE_GPUS[@]}"

  if [[ -z "${WANDB_COUNT:-}" ]]; then
    export WANDB_COUNT=100
  fi
  local phase2_total_count="$WANDB_COUNT"

  : "${PHASE2_LABELED_DIR:=$APP_DIR/data/katielinkmoleculenet_benchmark/train}"
  : "${PHASE2_UNLABELED_DIR:=$APP_DIR/data/ZINC-canonicalized}"

  if [[ ! -d "$PHASE2_LABELED_DIR" ]]; then
    echo "[$step][fatal] not a dir: $PHASE2_LABELED_DIR" >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    restore_env_var HARD_WALL_MINS "$prev_wall"
    return 2
  fi
  if [[ ! -d "$PHASE2_UNLABELED_DIR" ]]; then
    echo "[$step][fatal] not a dir: $PHASE2_UNLABELED_DIR" >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    restore_env_var HARD_WALL_MINS "$prev_wall"
    return 2
  fi

  local agent_workers=1
  if [[ -n "${PHASE2_AGENT_COUNT:-}" ]]; then
    if [[ "$PHASE2_AGENT_COUNT" =~ ^[0-9]+$ ]]; then
      agent_workers="$PHASE2_AGENT_COUNT"
    else
      echo "[$step][warn] ignoring non-numeric PHASE2_AGENT_COUNT='${PHASE2_AGENT_COUNT}'" >&2
      agent_workers=1
    fi
  elif (( gpu_count > 1 )); then
    agent_workers="$gpu_count"
  fi

  if (( gpu_count > 0 && agent_workers > gpu_count )); then
    agent_workers="$gpu_count"
  fi
  if (( agent_workers < 1 )); then
    agent_workers=1
  fi

  local launched=0

  if (( agent_workers == 1 || gpu_count <= 1 )); then
    export WANDB_COUNT="$phase2_total_count"
    if ! run_with_timeout wandb_agent; then
      restore_env_var LOG_DIR "$prev_log_dir"
      restore_env_var HARD_WALL_MINS "$prev_wall"
      return 1
    fi
    launched=1
  else
    local -a gpu_splits=()
    split_gpu_ids gpu_splits "$agent_workers" "${GRID_VISIBLE_GPUS[@]}"

    local -a agent_counts=()
    local base=$(( phase2_total_count / agent_workers ))
    local remainder=$(( phase2_total_count % agent_workers ))
    for ((i=0; i<agent_workers; ++i)); do
      local count="$base"
      if (( i < remainder )); then
        count=$((count + 1))
      fi
      agent_counts+=("$count")
    done

    local -a pids=()
    local -a labels=()
    for ((i=0; i<agent_workers; ++i)); do
      local count="${agent_counts[$i]}"
      if (( count <= 0 )); then
        continue
      fi
      (
        export LOG_DIR="${step_log_dir}/agent_${i}"
        mkdir -p "$LOG_DIR"
        if [[ -n "${gpu_splits[$i]:-}" ]]; then
          export CUDA_VISIBLE_DEVICES="${gpu_splits[$i]}"
        else
          unset CUDA_VISIBLE_DEVICES
        fi
        export WANDB_COUNT="$count"
        run_with_timeout wandb_agent
      ) &
      pids+=($!)
      labels+=("agent#$i")
      ((launched++))
    done

    if (( launched == 0 )); then
      export WANDB_COUNT="$phase2_total_count"
      if ! run_with_timeout wandb_agent; then
        restore_env_var LOG_DIR "$prev_log_dir"
        restore_env_var HARD_WALL_MINS "$prev_wall"
        return 1
      fi
      launched=1
    else
      local fail_rc=0
      for idx in "${!pids[@]}"; do
        local pid="${pids[$idx]}"
        if wait "$pid"; then
          :
        else
          local rc=$?
          echo "[$step][error] ${labels[$idx]} failed (rc=$rc)" >&2
          fail_rc=$rc
        fi
      done
      if (( fail_rc != 0 )); then
        restore_env_var LOG_DIR "$prev_log_dir"
        restore_env_var HARD_WALL_MINS "$prev_wall"
        return "$fail_rc"
      fi
    fi
  fi

  if find "$step_log_dir" -name 'wandb_agent.graceful_stop' -print -quit >/dev/null 2>&1; then
    mark_graceful_stop "$step"
  fi

  local metadata="${dir}/stage-outputs/${step}.json"
  python_inline "$metadata" "$sweep_id" "$phase2_total_count" "$agent_workers" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, sweep_id, total, workers = sys.argv[1:5]

payload = {
    "sweep_id": sweep_id,
    "requested_trials": int(total),
    "agent_workers": int(workers),
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
import os
os.replace(tmp, path)
PY

  local grid_state_path="${GRID_DIR}/grid_state.json"
  python_inline "$grid_state_path" "$sweep_id" "${GRID_EXP_ID:-}" "${MJEPACI_COMMIT_SHA:-unknown}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, sweep_id, grid_id, commit = sys.argv[1:5]

payload = {
    "base_id": grid_id or None,
    "sweep_id": sweep_id,
    "commit_sha": commit,
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

tmp = path + ".tmp"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY

  restore_env_var LOG_DIR "$prev_log_dir"
  restore_env_var HARD_WALL_MINS "$prev_wall"

  return 0
}

run_phase2_recheck_stage() {
  local dir="$1" step="$2"

  phase2_step_diag "$step"

  : "${TOPK_RECHECK:=5}"
  : "${EXTRA_SEEDS:=3}"
  : "${PHASE2_METRIC:=val_rmse}"
  : "${PHASE2_DIRECTION:=min}"
  : "${PHASE2_UNLABELED_DIR:=$APP_DIR/data/ZINC-canonicalized}"
  : "${PHASE2_LABELED_DIR:=$APP_DIR/data/katielinkmoleculenet_benchmark/train}"
  : "${PHASE2_RECHECK_WALL_MINS:=300}"
  : "${PHASE2_SEED_WALL_MINS:=}"
  : "${PHASE2_RECHECK_GRACE_SECS:=120}"

  local sweep_id_file="${GRID_SOURCE_DIR}/phase2_sweep_id.txt"
  if [[ ! -f "$sweep_id_file" ]]; then
    echo "[$step][fatal] sweep id file not found: $sweep_id_file (GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>}). Set GRID_EXP_ID=<id> to reuse an existing sweep or rerun phase2_sweep." >&2
    return 2
  fi

  local sweep_id
  sweep_id="$(<"$sweep_id_file")"
  export WANDB_SWEEP_ID2="$sweep_id"

  local prev_log_dir="${LOG_DIR:-}"
  local step_log_dir="${dir}/logs"
  mkdir -p "$step_log_dir"
  export LOG_DIR="$step_log_dir"

  local recheck_dir
  recheck_dir="$(stage_dir phase2_recheck)"
  mkdir -p "$recheck_dir"
  local sentinel="${recheck_dir}/recheck_done.ok"
  local incomplete="${recheck_dir}/recheck_incomplete.ok"
  local heartbeat_path="${recheck_dir}/heartbeat"

  if [[ -f "$sentinel" ]]; then
    echo "[$step] sentinel present; skipping" >&2
    phase2_promote_local_grid "$step"
    phase2_sync_grid_artifacts "$step"
    restore_env_var LOG_DIR "$prev_log_dir"
    return 0
  fi

  rm -f "$sentinel"
  rm -f "$incomplete"

  local wall_mins_raw="${PHASE2_RECHECK_WALL_MINS:-300}"
  local wall_mins="$wall_mins_raw"
  if ! [[ "$wall_mins" =~ ^[0-9]+$ ]] || [[ "$wall_mins" -le 0 ]]; then
    echo "[$step][warn] invalid PHASE2_RECHECK_WALL_MINS=${wall_mins_raw}; defaulting to 300" >&2
    wall_mins=300
  fi
  local soft=$(( wall_mins * 60 ))

  local grace_raw="${PHASE2_RECHECK_GRACE_SECS:-120}"
  local grace="$grace_raw"
  if ! [[ "$grace" =~ ^[0-9]+$ ]] || [[ "$grace" -le 0 ]]; then
    echo "[$step][warn] invalid PHASE2_RECHECK_GRACE_SECS=${grace_raw}; defaulting to 120" >&2
    grace=120
  fi
  export PHASE2_RECHECK_GRACE_SECS="$grace"

  local seed_wall_raw="${PHASE2_SEED_WALL_MINS:-}"
  local seed_wall_secs=""
  local seed_wall_display="off"
  if [[ -n "$seed_wall_raw" ]]; then
    if [[ "$seed_wall_raw" =~ ^[0-9]+$ ]] && [[ "$seed_wall_raw" -gt 0 ]]; then
      seed_wall_secs=$(( seed_wall_raw * 60 ))
      seed_wall_display="$seed_wall_raw"
    else
      echo "[$step][warn] invalid PHASE2_SEED_WALL_MINS=${seed_wall_raw}; disabling per-seed timeout" >&2
    fi
  fi

  echo "[ci] EXP_ID=${EXP_ID:-<unset>} GRID_EXP_ID=${GRID_EXP_ID:-<unset>} GRID_DIR=${GRID_DIR:-<unset>} STEP=${step} WALL=${wall_mins} SEED_WALL=${seed_wall_display}" >&2

  if [[ -n "$seed_wall_secs" ]]; then
    export PHASE2_SEED_WALL_SECS="$seed_wall_secs"
  else
    unset PHASE2_SEED_WALL_SECS || true
  fi

  mapfile -t __RECHECK_VISIBLE_GPUS < <(visible_gpu_ids)
  local recheck_gpu_count="${#__RECHECK_VISIBLE_GPUS[@]}"
  if [[ -z "${PHASE2_RECHECK_AGENT_COUNT:-}" ]]; then
    if [[ -n "${PHASE2_AGENT_COUNT:-}" ]]; then
      if [[ "$PHASE2_AGENT_COUNT" =~ ^[0-9]+$ ]] && (( PHASE2_AGENT_COUNT > 0 )); then
        export PHASE2_RECHECK_AGENT_COUNT="$PHASE2_AGENT_COUNT"
      else
        echo "[$step][warn] ignoring non-numeric PHASE2_AGENT_COUNT='${PHASE2_AGENT_COUNT}' for recheck" >&2
      fi
    elif (( recheck_gpu_count > 1 )); then
      export PHASE2_RECHECK_AGENT_COUNT="$recheck_gpu_count"
    fi
  fi
  unset __RECHECK_VISIBLE_GPUS || true

  export PHASE2_RECHECK_HEARTBEAT="$heartbeat_path"
  export PHASE2_RECHECK_SENTINEL="$sentinel"
  export PHASE2_RECHECK_RESUME=1
  export PHASE2_RECHECK_INCOMPLETE="$incomplete"

  local log_path="${step_log_dir}/recheck_topk.log"

  echo "[$step] wall budget=${wall_mins}m (${soft}s), grace=${grace}s"

  timeout --signal=SIGTERM --kill-after="$grace" "$soft" \
    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
    python -u "$APP_DIR/scripts/ci/recheck_topk_from_wandb.py" \
      --sweep "${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_SWEEP_ID2}" \
      --metric "${PHASE2_METRIC}" \
      --direction "${PHASE2_DIRECTION}" \
      --topk "${TOPK_RECHECK}" \
      --extra_seeds "${EXTRA_SEEDS}" \
      --program "$APP_DIR/scripts/train_jepa.py" \
      --subcmd "sweep-run" \
      --unlabeled-dir "${PHASE2_UNLABELED_DIR}" \
      --labeled-dir   "${PHASE2_LABELED_DIR}" \
      --out "${GRID_DIR}/recheck_summary.json" \
      --resume \
    2>&1 | tee "$log_path"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    :
  elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
    echo "[$step][fatal] recheck timed out after ${wall_mins} minutes (rc=$rc); aborting." >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    return "$rc"
  else
    restore_env_var LOG_DIR "$prev_log_dir"
    return "$rc"
  fi

  if [[ ! -f "$sentinel" ]]; then
    echo "[$step][fatal] sentinel missing after successful recheck execution: $sentinel" >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    return 4
  fi

  rm -f "$incomplete" 2>/dev/null || true

  phase2_sync_grid_artifacts "$step"

  local metadata="${dir}/stage-outputs/${step}.json"
  python_inline "$metadata" "$sweep_id" "${GRID_DIR}/recheck_summary.json" "${GRID_DIR}/best_grid_config.json" "${PHASE2_METRIC}" "${PHASE2_DIRECTION}" "${TOPK_RECHECK}" "${EXTRA_SEEDS}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, sweep_id, summary_path, best_path, metric, direction, topk, extra = sys.argv[1:9]

payload = {
    "sweep_id": sweep_id,
    "summary_path": summary_path,
    "best_config_path": best_path,
    "metric": metric,
    "direction": direction,
    "topk": int(topk),
    "extra_seeds": int(extra),
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
PY

  restore_env_var LOG_DIR "$prev_log_dir"
  unset PHASE2_SEED_WALL_SECS || true
  unset PHASE2_RECHECK_HEARTBEAT PHASE2_RECHECK_SENTINEL PHASE2_RECHECK_RESUME PHASE2_RECHECK_INCOMPLETE || true
  phase2_promote_local_grid "$step"
  return 0
}

run_phase2_export_stage() {
  local dir="$1" step="$2"

  phase2_step_diag "$step"

  phase2_promote_local_grid "$step"
  phase2_sync_grid_artifacts "$step"
  phase2_promote_grid_artifacts "$step"

  local sweep_id_file="${GRID_SOURCE_DIR}/phase2_sweep_id.txt"
  if [[ ! -f "$sweep_id_file" ]]; then
    echo "[$step][fatal] sweep id file not found: $sweep_id_file (GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>}). Set GRID_EXP_ID=<id> to reuse an existing sweep or rerun phase2_sweep." >&2
    return 3
  fi

  local prev_log_dir="${LOG_DIR:-}"
  local step_log_dir="${dir}/logs"
  mkdir -p "$step_log_dir"
  export LOG_DIR="$step_log_dir"

  local recheck_dir
  recheck_dir="$(stage_dir phase2_recheck)"
  local sentinel="${recheck_dir}/recheck_done.ok"
  local incomplete="${recheck_dir}/recheck_incomplete.ok"
  if [[ -f "$incomplete" ]]; then
    echo "[$step][fatal] recheck incomplete marker present: $incomplete. Rerun phase2_recheck before exporting." >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    return 4
  fi
  if [[ ! -f "$sentinel" ]]; then
    echo "[$step][fatal] recheck sentinel missing: $sentinel. Complete phase2_recheck before export." >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    return 4
  fi

  local best_json="${GRID_DIR}/best_grid_config.json"
  local summary_json="${GRID_DIR}/recheck_summary.json"

  if [[ ! -f "$best_json" ]]; then
    echo "[$step][fatal] expected best_grid_config.json at $best_json but it is missing (GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>})." >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    return 3
  fi

  if [[ ! -f "$summary_json" ]]; then
    echo "[$step][fatal] expected recheck_summary.json at $summary_json but it is missing (GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>})." >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    return 3
  fi

  python_inline "$best_json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
if not isinstance(payload, dict) or not payload.get("config"):
    raise SystemExit(4)
PY

  local outputs_dir="${dir}/stage-outputs"
  mkdir -p "$outputs_dir"

  cp -f "$best_json" "${outputs_dir}/best_grid_config.json"
  if [[ -f "$summary_json" ]]; then
    cp -f "$summary_json" "${outputs_dir}/recheck_summary.json"
  fi

  local helper_dir="${outputs_dir}/helpers"
  mkdir -p "$helper_dir"

  while IFS= read -r rel; do
    [[ -z "$rel" || "$rel" == "." ]] && continue
    if [[ -f "${GRID_DIR}/${rel}" ]]; then
      cp -f "${GRID_DIR}/${rel}" "${helper_dir}/"
    fi
  done < <(cd "$GRID_DIR" && find . -maxdepth 1 -type f \( -name 'phase2_winner*' -o -name 'winner_*' -o -name 'phase2_cli*' \) -printf '%P\n' 2>/dev/null || true)

  local metadata="${dir}/stage-outputs/${step}.json"
  python_inline "$metadata" "$best_json" "$summary_json" "$helper_dir" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, best_path, summary_path, helper_dir = sys.argv[1:5]

helpers = []
if os.path.isdir(helper_dir):
    for entry in sorted(os.listdir(helper_dir)):
        helpers.append(os.path.join(helper_dir, entry))

payload = {
    "best_config_path": best_path,
    "summary_path": summary_path if os.path.exists(summary_path) else None,
    "helper_files": helpers,
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
PY

  printf '[%s] validated Phase-2 best configuration\n' "$step" | tee -a "${step_log_dir}/export.log" >/dev/null

  restore_env_var LOG_DIR "$prev_log_dir"
  return 0
}

# ---------- build & filter args ----------
build_stage_args() {
  local s="${1:?stage}"
  local section="$s"
  local subcmd="$s"
  local skip_best=0
  local skip_allowlist=0
  export TRAIN_JEPA_CI="$APP_DIR/scripts/ci/train_jepa_ci.yml"

  # YAML args
  #snake to kebab case
  if [ "$s" = "grid_search" ]; then
    section="grid_search"   # YAML key
    subcmd="grid-search"    # argparse subcommand
  fi

  if [ "$s" = "bench" ]; then
    section="bench"
    subcmd="benchmark"
  fi

  if [ "$s" = "report" ]; then
    skip_best=1
    skip_allowlist=1
  fi

  build_argv_from_yaml "$section"
  expand_array_vars ARGV
  prune_empty_args ARGV
  #echo "ARGV after expansion: ${ARGV[@]}"

  # Best (from grid) → append last so it overrides YAML
  local -a BEST=()
  if (( ! skip_best )); then
    mapfile -t BEST < <(best_config_args "$section")
    expand_array_vars BEST
    prune_empty_args BEST
  fi

  local -a COMBINED=( "${ARGV[@]}" "${BEST[@]}" )

  # Dynamic allowlist from tool help (supports nargs='+')
  local -a OUT=()
  if (( skip_allowlist )); then
    OUT=("${COMBINED[@]}")
  else
    local -a ALLOWED=()
    local help_output=""
    local fallback_to_micromamba=0

    if py=$(python_bin 2>/dev/null); then
      if help_output=$(PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$py" "$APP_DIR/scripts/train_jepa.py" "$subcmd" --help 2>&1
      ); then
        mapfile -t ALLOWED < <(printf '%s\n' "$help_output" |
          sed -n 's/.*\(--[a-z0-9-]\+\).*/\1/p' | sort -u)
      else
        local status=$?
        fallback_to_micromamba=1
        if [[ "$help_output" == *"ModuleNotFoundError"* || "$help_output" == *"ImportError"* ]]; then
          printf '[stage:%s] python (%s) is missing dependencies for %s --help; retrying via micromamba.\n' \
            "$s" "$py" "$subcmd" >&2
        else
          printf '[stage:%s] python (%s) failed to execute %s --help (exit %d); retrying via micromamba.\n' \
            "$s" "$py" "$subcmd" "$status" >&2
          printf '%s\n' "$help_output" >&2
        fi
      fi
    else
      fallback_to_micromamba=1
    fi

    if (( fallback_to_micromamba )); then
      ensure_micromamba
      if help_output=$(PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$MMBIN" run -n mjepa python "$APP_DIR/scripts/train_jepa.py" "$subcmd" --help 2>&1
      ); then
        mapfile -t ALLOWED < <(printf '%s\n' "$help_output" |
          sed -n 's/.*\(--[a-z0-9-]\+\).*/\1/p' | sort -u)
      else
        local status=$?
        printf '[stage:%s] failed to execute %s --help even via micromamba python (exit %d).\n' \
          "$s" "$subcmd" "$status" >&2
        printf '%s\n' "$help_output" >&2
        return "$status"
      fi
    fi

    local i=0 j=0
    while (( i < ${#COMBINED[@]} )); do
      local f="${COMBINED[$i]}"
      if [[ "$f" == --* ]] && printf '%s\n' "${ALLOWED[@]}" | grep -qx -- "$f"; then
        OUT+=("$f")
        j=$((i+1))
        while (( j < ${#COMBINED[@]} )) && [[ "${COMBINED[$j]}" != --* ]]; do
          OUT+=("${COMBINED[$j]}"); ((j++))
        done
        i=$j; continue
      fi
      ((i++))
    done
  fi

  prune_empty_args OUT

  if [ "$s" = "tox21" ]; then
    local enforce_flag="${TOX21_FULL_FINETUNE:-}"
    local have_flag=0
    local token
    for token in "${OUT[@]}"; do
      if [[ "$token" == "--full-finetune" || "$token" == "--no-full-finetune" ]]; then
        have_flag=1
        break
      fi
    done
    if (( ! have_flag )); then
      case "${enforce_flag,,}" in
        1|true|yes|on)
          OUT+=("--full-finetune")
          ;;
        0|false|no|off)
          OUT+=("--no-full-finetune")
          ;;
      esac
    fi
  fi

  if [ "$s" = "report" ]; then
    local -a FILTERED=()
    local idx=0
    while (( idx < ${#OUT[@]} )); do
      local token="${OUT[$idx]}"
      if [[ "$token" == --* ]]; then
        local next=$((idx + 1))
        if (( next < ${#OUT[@]} )) && [[ "${OUT[$next]}" != --* ]]; then
          local value="${OUT[$next]}"
          if [[ -z "$value" ]]; then
            idx=$((next + 1))
            continue
          fi
          FILTERED+=("$token" "$value")
          idx=$((next + 1))
          continue
        fi
      fi
      FILTERED+=("$token")
      ((idx++))
    done
    OUT=("${FILTERED[@]}")
  fi

  STAGE_ARGS=("${OUT[@]}")
}

# ---------- dataset preflight (harmless if flags absent) ----------
# Trim CR and whitespace
clean_path() {
  echo "$1" | tr -d '\r\n' | xargs
}

stage_dataset_preflight() {
  local -n arr="$1"
  local getv; getv() { local k="$1" n=0; while (( n < ${#arr[@]} )); do [[ "${arr[$n]}" == "$k" ]] && { echo "${arr[$((n+1))]}"; return 0; }; ((n++)); done; return 1; }
  local UL="$(getv --unlabeled-dir || true)"
  local LBL="$(getv --labeled-dir  || true)"
  local DS="$(getv --dataset-dir   || true)"
  local CSV="$(getv --csv          || true)"

  UL=$(clean_path "$UL")
  LBL=$(clean_path "$LBL")
  CSV=$(clean_path "$CSV")
  DS=$(clean_path "$DS")

  if [[ -n "$UL" && ! -d "$UL" ]]; then
    echo "[warn] unlabeled-dir not found: $UL"
  fi
  if [[ -n "$LBL" && ! -d "$LBL" ]]; then
    echo "[warn] labeled-dir not found: $LBL"
  fi
  if [[ -n "$DS" && ! -d "$DS" ]]; then
    echo "[warn] dataset-dir not found: $DS"
  fi
  if [[ -n "$CSV" && ! -f "$CSV" ]]; then
    echo "[warn] csv file not found: $CSV"
  fi
  }

# ---------- timeout + SIGINT ----------
run_with_timeout() {
  local s="$1"; shift
  local arr_name="${1:-}"

    # --- JEPA mode: run_stage passes an array name ---
  if [[ "$s" != "wandb_agent" ]]; then
    if [ "$s" = "report" ]; then
      local -a arr=()
      if [[ -n "$arr_name" ]]; then
        local -n arr_ref="$arr_name"
        arr=("${arr_ref[@]}")
      fi

      local BUDGET_MINS="${REPORT_TIME_BUDGET_MINS:-${HARD_WALL_MINS:-30}}"
      local SOFT=$((BUDGET_MINS*60))
      local GRACE="${KILL_AFTER_SECS:-60}"
      echo "[stage] wall budget=${BUDGET_MINS}m (${SOFT}s), grace=${GRACE}s"

      mkdir -p "$LOG_DIR"
      LOG="${LOG_DIR}/${s}.log"

      if [[ -z "${MPLBACKEND:-}" ]]; then
        if grep -q "matplotlib" "$APP_DIR"/reports/plots_*.py 2>/dev/null; then
          export MPLBACKEND="Agg"  # debug: enforce non-interactive MPL backend when plots import matplotlib
        fi
      fi

      if [[ -n "${OUT_DIR:-}" ]]; then
        mkdir -p "$OUT_DIR"
        local stat_output=""
        if stat_output=$(stat -c '%U:%G %a' "$OUT_DIR" 2>/dev/null); then
          echo "report: OUT_DIR=${OUT_DIR} perms=${stat_output}"
        else
          echo "report: unable to stat OUT_DIR=${OUT_DIR}" >&2
        fi
      fi

      local -a report_py_cmd=(python -m reports.build_wandb_report "${arr[@]}")
      local report_cmd_str=""
      printf -v report_cmd_str '%q ' "${report_py_cmd[@]}"
      report_cmd_str="${report_cmd_str% }"

      echo "report: starting at $(date -Is)"
      echo "report: invoking: ${report_cmd_str}"
      echo "report: WANDB_MODE=${WANDB_MODE:-<unset>} WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-<unset>}"

      local report_start_ts
      report_start_ts=$(date +%s)

      report_heartbeat() {
        while true; do
          echo "report: heartbeat $(date -Is)"
          sleep 30
        done
      }
      report_heartbeat &
      local heartbeat_pid=$!

      timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
        env PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
        "${report_py_cmd[@]}" \
        2>&1 | tee "$LOG_DIR/${s}.log"

      local -r timeout_rc=${PIPESTATUS[0]}

      if [[ -n "$heartbeat_pid" ]]; then
        kill "$heartbeat_pid" 2>/dev/null || true
        wait "$heartbeat_pid" 2>/dev/null || true
      fi

      echo "report: finished at $(date -Is) (rc=${timeout_rc})"

      local elapsed_ts
      elapsed_ts=$(($(date +%s) - report_start_ts))
      echo "report: elapsed ${elapsed_ts}s"

      if [[ -n "${OUT_DIR:-}" && -d "$OUT_DIR" ]]; then
        local -a report_artifacts=()
        mapfile -t report_artifacts < <(find "$OUT_DIR" -maxdepth 2 -mindepth 1 -type f | sort)
        echo "report: artifacts ${#report_artifacts[@]} files under ${OUT_DIR}"
        local artifact_path
        for artifact_path in "${report_artifacts[@]}"; do
          echo "report: artifact ${artifact_path}"
        done
      fi

      if [[ $timeout_rc -eq 0 ]]; then
        :
      elif [[ $timeout_rc -eq 124 || $timeout_rc -eq 130 || $timeout_rc -eq 143 || $timeout_rc -eq 137 ]]; then
        echo "[INFO][$s] graceful stop (rc=$timeout_rc); not marking stage done; outputs should be flushed."
        mark_graceful_stop "$s"
        return 0
      else
        echo "[ERROR][$s] build_wandb_report failed with exit code $timeout_rc" >&2
        exit $timeout_rc
      fi
      return 0
    fi

    if [[ -n "$arr_name" ]]; then
      local -n arr="$arr_name"; shift
    else
      local -a arr=()
    fi
    local subcmd; subcmd="$(stage_subcmd "$s")"

    #snake to kebab case  
    if [ "$s" = "grid_search" ]; then
      section="grid_search"   # YAML key
      subcmd="grid-search"    # argparse subcommand
    fi

    if [ "$s" = "bench" ]; then
      section="bench"
      subcmd="benchmark"
    fi

    # parsing flags from arr[@] inside stage.sh (e.g. --epochs 50, --batch-size 128),
    local getv; getv() {
      local k="$1" n=0
      while (( n < ${#arr[@]} )); do
        [[ "${arr[$n]}" == "$k" ]] && { echo "${arr[$((n+1))]}"; return 0; }
        ((n++))
      done
      return 1
    }

    # handle default timeouts differently
    local BUDGET_MINS="$(getv --time-budget-mins || true)"
    case "$s" in
      grid) : "${BUDGET_MINS:=${HARD_WALL_MINS:-240}}" ;;
      pretrain|finetune) : "${BUDGET_MINS:=${HARD_WALL_MINS:-120}}" ;;
      *) : "${BUDGET_MINS:=${HARD_WALL_MINS:-60}}" ;;
    esac
    local SOFT="$((BUDGET_MINS*60))"   # convert minutes → seconds
    local GRACE="${KILL_AFTER_SECS:-60}"
    echo "[stage] wall budget=${BUDGET_MINS}m (${SOFT}s), grace=${GRACE}s"

    mkdir -p "$LOG_DIR" 
    LOG="${LOG_DIR}/${s}.log"
    
    local -a launch_prefix=("$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1)
    local -a entrypoint=("$APP_DIR/scripts/train_jepa.py" "$subcmd" "${arr[@]}")
    local -a ddp_launcher=()
    local previous_world="${WORLD_SIZE-}"
    local previous_master_addr="${MASTER_ADDR-}"
    local previous_master_port="${MASTER_PORT-}"
    local ddp_env_modified=0

    if [[ "$s" == "finetune" ]]; then
      local raw_devices
      raw_devices="$(getv --devices || true)"
      if [[ -z "$raw_devices" && -n "${FINETUNE_DEVICES:-}" ]]; then
        raw_devices="${FINETUNE_DEVICES}"
      fi
      if [[ "$raw_devices" =~ ^[0-9]+$ && "$raw_devices" -gt 1 ]]; then
        local ddp_supported=0
        if "${launch_prefix[@]}" python - <<'PY' >/dev/null 2>&1; then
import sys
try:
    import torch  # noqa: F401
    import torch.distributed.run  # noqa: F401
except Exception:
    sys.exit(1)
PY
          ddp_supported=1
        fi
        if (( ! ddp_supported )); then
          echo "[stage:$s] torch.distributed.run unavailable; falling back to single-process execution" >&2
        fi
        if (( ddp_supported )); then
          local world="${WORLD_SIZE:-}"
          if [[ -z "$world" || "$world" -le 1 ]]; then
            local ddp_port="${MASTER_PORT:-}"
            if [[ -z "$ddp_port" ]]; then
              # Derive a semi-random port in the high ephemeral range to avoid clashes
              ddp_port=$(( (RANDOM % 20000) + 15000 ))
            fi
            export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
            export MASTER_PORT="$ddp_port"
            export WORLD_SIZE="$raw_devices"
            ddp_env_modified=1
            echo "[stage:$s] enabling torch.distributed.run (nproc_per_node=${raw_devices}, master_port=${MASTER_PORT})" >&2
          else
            echo "[stage:$s] WORLD_SIZE=${world}; assuming external launcher configured DDP" >&2
          fi
          ddp_launcher=(python -m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=${raw_devices}")
        fi
      fi
    fi

    local -a micromamba_cmd=("${launch_prefix[@]}")
    if (( ${#ddp_launcher[@]} )); then
      micromamba_cmd+=("${ddp_launcher[@]}" "${entrypoint[@]}")
    else
      micromamba_cmd+=(python -u "${entrypoint[@]}")
    fi

    timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
      env PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
      "${micromamba_cmd[@]}" \
      2>&1 | tee "$LOG_DIR/${s}.log"
    rc=${PIPESTATUS[0]}
    if (( ddp_env_modified )); then
      if [[ -n "$previous_world" ]]; then
        export WORLD_SIZE="$previous_world"
      else
        unset WORLD_SIZE
      fi
      if [[ -n "$previous_master_addr" ]]; then
        export MASTER_ADDR="$previous_master_addr"
      else
        unset MASTER_ADDR
      fi
      if [[ -n "$previous_master_port" ]]; then
        export MASTER_PORT="$previous_master_port"
      else
        unset MASTER_PORT
      fi
    fi
    # 0   = success
    # 124 = 'timeout' exceeded (we later sent SIGTERM/SIGKILL)
    # 143 = terminated by SIGTERM (128+15)
    # 137 = killed by SIGKILL   (128+9)
    if [[ $rc -eq 0 ]]; then
      :
    elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
      echo "[INFO][$s] graceful stop (rc=$rc); not marking stage done; outputs should be flushed."
      mark_graceful_stop "$s"
      return 0
    else
      echo "[ERROR][$s] train_jepa.py failed with exit code $rc" >&2
      exit $rc
    fi
  # --- WandB mode: run-grid passes a full cmd array --
  else
    ensure_micromamba

    # ensure the binary is callable in this subshell
    export PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"
    : "${MMBIN:=${MAMBA_ROOT_PREFIX}/bin/micromamba}"
    if [[ "$(basename "$MMBIN")" = "micromamba" && ! "$MMBIN" = /* ]]; then
      MMBIN="${MAMBA_ROOT_PREFIX}/bin/micromamba"
    fi
    if [[ ! -x "$MMBIN" ]]; then
      echo "[wandb_agent][fatal] micromamba not found at $MMBIN" >&2
      exit 1
    fi

   # --- ensure logging + ids are sane ---
    mkdir -p "$LOG_DIR"
    LOG="${LOG_DIR}/${s}.log"

    # build a full sweep path if only an id was provided
    SID="${SWEEP_ID}"
    if [[ -z "$SID" ]]; then echo "[wandb_agent][fatal] SWEEP_ID is empty" >&2; exit 1; fi

    if [[ "$SID" != */* ]]; then
      [[ -n "${WANDB_ENTITY:-}" && -n "${WANDB_PROJECT:-}" ]] \
        || { echo "[wandb_agent][fatal] SWEEP_ID not qualified and entity/project unset"; exit 1; }
      SID="${WANDB_ENTITY}/${WANDB_PROJECT}/${SID}"
    fi

    echo "[wandb_agent] using sweep: $SID"

    local -a cmd=("$@")
    local SOFT=$(( (${HARD_WALL_MINS:-240})*60 ))
    local GRACE="${KILL_AFTER_SECS:-60}"
    echo "[wandb_agent] wall budget=${SOFT}s, grace=${GRACE}s"

    timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
      "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -m wandb agent --count ${WANDB_COUNT:-50} "$SID" \
      2>&1 | tee "$LOG"

    rc=$?
    # If the agent “gracefully” exited 0 but clearly failed runs, force non-zero
    if grep -qE 'Detected [0-9]+ failed runs|error: the following arguments are required' "$LOG"; then
      echo "[ERROR][wandb_agent] runs failed; forcing non-zero exit"
      rc=1
    fi
    if [[ $rc -eq 0 ]]; then
      :
    elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
      echo "[INFO][wandb_agent] graceful stop (rc=$rc); letting agent flush."
      mark_graceful_stop "$s"
      return 0
    else
      echo "[ERROR][wandb_agent] wandb agent failed with exit code $rc" >&2
      exit $rc
    fi
  fi
}

# ---------- one-call entry ----------
run_stage() {
  local s="${1:?stage}"
  local stage="${s,,}"
  local dir
  dir="$(stage_dir "$stage")"
  local rc=0
  OUT_DIR="$dir"
  export OUT_DIR
  if declare -F ci_refresh_freeze_state >/dev/null 2>&1; then
    ci_refresh_freeze_state "${FREEZE_MARKER:-}"
  fi
  if declare -F ci_setup_vast_ssh_key >/dev/null 2>&1; then
    ci_setup_vast_ssh_key || true
  fi
  local grid_read="${GRID_SOURCE_DIR:-${GRID_DIR:-<unset>}}"
  echo "[ci] STAGE=${stage} EXP_ID=${EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>} GRID_EXP_ID=${GRID_EXP_ID:-<unset>} FROZEN=${FROZEN:-0}" >&2
  echo "     READ: ARTIFACTS_DIR=${PRETRAIN_ARTIFACTS_DIR:-<unset>} GRID_DIR=${grid_read}" >&2
  echo "     WRITE: OUT_DIR=${OUT_DIR:-<unset>} EXPERIMENT_DIR=${EXPERIMENT_DIR:-<unset>}" >&2

  if (( FROZEN )) && [[ "${CI_FORCE_UNFREEZE_GRID}" != "1" ]]; then
    case "$stage" in
      pretrain|grid|grid_search|phase1|phase2_sweep|phase2_recheck|phase2_export|finetune)
        echo "[ci] skip: frozen lineage (stage=${stage})" >&2
        return 0
        ;;
    esac
  fi

  local ignore_drift=0
  if (( FROZEN )) && [[ "${STRICT_FROZEN}" != "1" ]]; then
    if [[ "${ALLOW_CODE_DRIFT_WHEN_FROZEN}" != "0" ]]; then
      ignore_drift=1
    fi
  fi
  local stamp
  stamp="$(_stamp "$dir")"
  local state_path
  state_path="$(stage_state_file "$dir")"
  local allow_stale="${ALLOW_STALE_RUN:-}"
  local forced=0
  if stage_is_forced "$stage"; then
    forced=1
  fi
  local shim_mode=0
  if [[ -n "${MJEPACI_STAGE_SHIM:-}" && -x "${MJEPACI_STAGE_SHIM}" ]]; then
    shim_mode=1
    forced=1
  fi

  if [[ "$stage" == "pretrain" && $forced -eq 0 ]]; then
    phase2_sync_grid_artifacts "$stage"
    phase2_promote_grid_artifacts "$stage"
    local grid_root="${GRID_DIR:-}"
    local sweep_file="${grid_root%/}/phase2_sweep_id.txt"
    local best_json="${grid_root%/}/best_grid_config.json"
    local summary_json="${grid_root%/}/recheck_summary.json"
    local sentinel_path="$(stage_dir phase2_recheck)/recheck_done.ok"

    if [[ -z "$grid_root" ]]; then
      echo "[pretrain][fatal] GRID_DIR is unset; run phase2_sweep/recheck before pretraining." >&2
      return 3
    fi
    if [[ ! -f "$sweep_file" ]]; then
      echo "[pretrain][fatal] expected phase2 sweep output at $sweep_file" >&2
      return 3
    fi
    if [[ ! -f "$best_json" ]]; then
      echo "[pretrain][fatal] expected best_grid_config.json at $best_json" >&2
      return 3
    fi
    if [[ ! -f "$summary_json" ]]; then
      echo "[pretrain][fatal] expected recheck_summary.json at $summary_json" >&2
      return 3
    fi
    if [[ ! -f "$sentinel_path" ]]; then
      echo "[pretrain][fatal] missing Phase-2 recheck sentinel at $sentinel_path" >&2
      return 3
    fi
  fi

  local -a dependencies=()
  mapfile -t dependencies < <(stage_needs "$stage")

  local config_hash=""
  local data_hash=""
  local -a stage_args=()
  local stage_args_ready=0

  if (( shim_mode )); then
    local shim_meta="${MJEPACI_STAGE_SHIM}"
    if [[ -n "$MJEPACI_STAGE_SHIM" && -e "$MJEPACI_STAGE_SHIM" ]]; then
      shim_meta+="|$(stat -c '%s:%Y' "$MJEPACI_STAGE_SHIM" 2>/dev/null || echo unknown)"
    fi
    config_hash="$(stage_compute_hash "$stage" "shim=${shim_meta}" "stage=${stage}")"
  else
    case "$stage" in
      phase2_sweep)
        local sweep_id_file="${GRID_DIR}/phase2_sweep_id.txt"
        local sweep_id=""
        if [[ -f "$sweep_id_file" ]]; then
          sweep_id="$(<"$sweep_id_file")"
        fi
        config_hash="$(stage_compute_hash "$stage" \
          "sweep_id=${sweep_id}" \
          "wandb_count=${WANDB_COUNT:-}" \
          "agent_count=${PHASE2_AGENT_COUNT:-}" \
          "labeled=${PHASE2_LABELED_DIR:-}" \
          "unlabeled=${PHASE2_UNLABELED_DIR:-}" \
          "wall=${PHASE2_SWEEP_WALL_MINS:-}" \
        )"
        ;;
      phase2_recheck)
        local sweep_id_file="${GRID_SOURCE_DIR}/phase2_sweep_id.txt"
        local sweep_id=""
        if [[ -f "$sweep_id_file" ]]; then
          sweep_id="$(<"$sweep_id_file")"
        fi
        config_hash="$(stage_compute_hash "$stage" \
          "sweep_id=${sweep_id}" \
          "metric=${PHASE2_METRIC:-val_rmse}" \
          "direction=${PHASE2_DIRECTION:-min}" \
          "topk=${TOPK_RECHECK:-5}" \
          "extra=${EXTRA_SEEDS:-3}" \
          "labeled=${PHASE2_LABELED_DIR:-}" \
          "unlabeled=${PHASE2_UNLABELED_DIR:-}" \
        )"
        ;;
      phase2_export)
        local best_json="${GRID_SOURCE_DIR}/best_grid_config.json"
        local summary_json="${GRID_DIR}/recheck_summary.json"
        local best_sig="" summary_sig=""
        [[ -f "$best_json" ]] && best_sig="$(sha256sum "$best_json" 2>/dev/null | awk '{print $1}')"
        [[ -f "$summary_json" ]] && summary_sig="$(sha256sum "$summary_json" 2>/dev/null | awk '{print $1}')"
        local sweep_id_file="${GRID_SOURCE_DIR}/phase2_sweep_id.txt"
        local sweep_id=""
        if [[ -f "$sweep_id_file" ]]; then
          sweep_id="$(<"$sweep_id_file")"
        fi
        config_hash="$(stage_compute_hash "$stage" "sweep_id=${sweep_id}" "best=${best_sig}" "summary=${summary_sig}")"
        ;;
      *)
        build_stage_args "$stage"
        stage_dataset_preflight STAGE_ARGS
        stage_args_ready=1
        stage_args=("${STAGE_ARGS[@]}")
        config_hash="$(stage_compute_hash "$stage" "${stage_args[@]}")"
        ;;
    esac
  fi

  echo "[ci][stage=${stage}] commit=${MJEPACI_COMMIT_SHA:-unknown} config_hash=${config_hash:-<unset>} allow_stale=${allow_stale:-0} forced=${forced} FORCE_RERUN=${FORCE_RERUN:-<unset>} shim=${shim_mode}" >&2

  local skip=0
  local skip_reason=""
  local rerun_reason=""
  if (( forced )); then
    rerun_reason="forced"
  else
    if [[ -f "$stamp" ]]; then
      if ! stage_state_load "$state_path"; then
        rerun_reason="missing stage_state.json"
      else
        if [[ -n "$STAGE_STATE_COMMIT" && "$STAGE_STATE_COMMIT" != "${MJEPACI_COMMIT_SHA:-unknown}" ]]; then
          if (( ignore_drift )); then
            :
          elif [[ "$allow_stale" == "1" ]]; then
            echo "[ci][stage=${stage}] commit mismatch (${STAGE_STATE_COMMIT} -> ${MJEPACI_COMMIT_SHA:-unknown}) but ALLOW_STALE_RUN=1; reusing cache" >&2
          else
            rerun_reason="commit changed (${STAGE_STATE_COMMIT} -> ${MJEPACI_COMMIT_SHA:-unknown})"
          fi
        fi
        if [[ -z "$rerun_reason" && -n "$config_hash" && -n "$STAGE_STATE_CONFIG_HASH" && "$config_hash" != "$STAGE_STATE_CONFIG_HASH" ]]; then
          if (( ignore_drift )); then
            :
          else
            rerun_reason="config hash changed"
          fi
        fi
        if [[ -z "$rerun_reason" && -n "$data_hash" && -n "$STAGE_STATE_DATA_HASH" && "$data_hash" != "$STAGE_STATE_DATA_HASH" ]]; then
          if (( ignore_drift )); then
            :
          else
            rerun_reason="data hash changed"
          fi
        fi
        if [[ -z "$rerun_reason" && "$stage" == "phase2_recheck" ]]; then
          local sentinel_path
          sentinel_path="$(stage_dir phase2_recheck)/recheck_done.ok"
          if [[ ! -f "$sentinel_path" ]]; then
            rerun_reason="missing sentinel"
          fi
        fi
        if [[ -z "$rerun_reason" ]]; then
          if needs_stage "$dir" "${dependencies[@]}"; then
            rerun_reason="inputs updated"
          else
            skip=1
            skip_reason="cache hit (outputs newer than dependencies; stamp=${stamp})"
          fi
        fi
      fi
    else
      rerun_reason="missing stamp"
    fi
  fi

  if (( skip )); then
    echo "[${stage}] cache hit - skipping"
    [[ -z "$skip_reason" ]] && skip_reason="cache hit"
    echo "[ci][info] stage=${stage} skip_reason=${skip_reason}" >&2
    if [[ "$stage" == "bench" ]]; then
      echo "[ci][info] Benchmark reason: ${skip_reason}" >&2
    fi
    return 0
  fi

  if [[ -n "$rerun_reason" && -f "$stamp" ]]; then
    echo "[${stage}] rerun triggered: ${rerun_reason}" >&2
    rm -f "$stamp" "$state_path" 2>/dev/null || true
  fi

  mkdir -p "$dir" "$dir/stage-outputs"
  local inputs_tmp deps_tmp outputs_tmp
  inputs_tmp="$(mktemp)"
  deps_tmp="$(mktemp)"
  outputs_tmp="$(mktemp)"

  if (( stage_args_ready )); then
    printf '%s\n' "${stage_args[@]}" >"$inputs_tmp"
    STAGE_ARGS=("${stage_args[@]}")
  else
    STAGE_ARGS=()
    : >"$inputs_tmp"
  fi
  printf '%s\n' "${dependencies[@]}" >"$deps_tmp"
  printf '%s\n' "${dir}/stage-outputs" >"$outputs_tmp"
  printf '%s\n' "${dir}/logs" >>"$outputs_tmp"
  case "$stage" in
    pretrain)
      printf '%s\n' "${PRETRAIN_DIR}/encoder.pt" >>"$outputs_tmp"
      printf '%s\n' "${PRETRAIN_ARTIFACTS_DIR}/encoder_manifest.json" >>"$outputs_tmp"
      ;;
    finetune)
      printf '%s\n' "${FINETUNE_DIR}/seed_0/ft_best.pt" >>"$outputs_tmp"
      ;;
    grid)
      printf '%s\n' "${GRID_DIR}/best_grid_config.json" >>"$outputs_tmp"
      printf '%s\n' "${GRID_DIR}/paired_effect.json" >>"$outputs_tmp"
      ;;
    phase2_sweep)
      printf '%s\n' "${GRID_DIR}/phase2_sweep_id.txt" >>"$outputs_tmp"
      printf '%s\n' "${GRID_DIR}/grid_state.json" >>"$outputs_tmp"
      ;;
    phase2_recheck)
      printf '%s\n' "${GRID_DIR}/recheck_summary.json" >>"$outputs_tmp"
      printf '%s\n' "${dir}/recheck_done.ok" >>"$outputs_tmp"
      ;;
    phase2_export)
      printf '%s\n' "${GRID_DIR}/best_grid_config.json" >>"$outputs_tmp"
      ;;
  esac

  clear_graceful_stop "$stage"
  clear_graceful_stop "$stage" "${LOG_DIR:-}"
  clear_graceful_stop "$stage" "${dir}/logs"

  if (( shim_mode )); then
    echo "[${stage}] starting (shim)" >&2
    if ! "$MJEPACI_STAGE_SHIM" "$stage"; then
      rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
      return $?
    fi
    case "$stage" in
      phase2_sweep)
        if [[ -n "${GRID_DIR:-}" ]]; then
          local sweep_stub="${GRID_DIR}/phase2_sweep_id.txt"
          if [[ ! -f "$sweep_stub" ]]; then
            mkdir -p "${GRID_DIR}"
            printf 'shim-phase2-sweep\n' >"$sweep_stub"
          fi
        fi
        ;;
      phase2_recheck)
        local shim_sentinel="${dir}/recheck_done.ok"
        mkdir -p "$(dirname "$shim_sentinel")"
        : >"$shim_sentinel"
        phase2_sync_grid_artifacts "$stage"
        if [[ -n "${GRID_DIR:-}" ]]; then
          mkdir -p "${GRID_DIR}"
          local shim_best="${GRID_DIR}/best_grid_config.json"
          local shim_summary="${GRID_DIR}/recheck_summary.json"
          [[ -f "$shim_best" ]] || printf '{}\n' >"$shim_best"
          [[ -f "$shim_summary" ]] || printf '{}\n' >"$shim_summary"
        fi
        ;;
      phase2_export)
        phase2_sync_grid_artifacts "$stage"
        if [[ -n "${GRID_DIR:-}" ]]; then
          mkdir -p "${GRID_DIR}"
          local shim_best="${GRID_DIR}/best_grid_config.json"
          [[ -f "$shim_best" ]] || printf '{}\n' >"$shim_best"
        fi
        ;;
    esac
    stage_state_write "$stage" "$dir" "$config_hash" "$data_hash" "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
    mark_stage_done "$dir"
    rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
    return 0
  fi

  ensure_micromamba
  : "${WANDB_NAME:=$stage}"; export WANDB_NAME
  : "${WANDB_JOB_TYPE:=$stage}"; export WANDB_JOB_TYPE
  export WANDB_RUN_GROUP="${GITHUB_RUN_ID:-${WANDB_RUN_GROUP:-}}"

  echo "[${stage}] starting"

  local stage_log_dir="${LOG_DIR:-}"
  case "$stage" in
    phase2_sweep)
      stage_log_dir="${dir}/logs"
      clear_graceful_stop "$stage" "$stage_log_dir"
      if ! run_phase2_sweep_stage "$dir" "$stage"; then
        rc=$?
        rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
        return $rc
      fi
      ;;
    phase2_recheck)
      stage_log_dir="${dir}/logs"
      clear_graceful_stop "$stage" "$stage_log_dir"
      if ! run_phase2_recheck_stage "$dir" "$stage"; then
        rc=$?
        rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
        return $rc
      fi
      ;;
    phase2_export)
      stage_log_dir="${dir}/logs"
      clear_graceful_stop "$stage" "$stage_log_dir"
      if ! run_phase2_export_stage "$dir" "$stage"; then
        rc=$?
        rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
        return $rc
      fi
      ;;
    *)
      stage_log_dir="${LOG_DIR:-}"
      clear_graceful_stop "$stage" "$stage_log_dir"
      run_with_timeout "$stage" STAGE_ARGS
      ;;
  esac

  local -a grace_dirs=()
  if [[ -n "$stage_log_dir" ]]; then
    grace_dirs+=("$stage_log_dir")
  fi
  if [[ -n "${LOG_DIR:-}" && "${LOG_DIR:-}" != "$stage_log_dir" ]]; then
    grace_dirs+=("${LOG_DIR:-}")
  fi

  local candidate
  for candidate in "${grace_dirs[@]}"; do
    if [[ -n "$candidate" ]] && was_graceful_stop "$stage" "$candidate"; then
      echo "[${stage}] stopped gracefully; leaving cache unstamped so it can resume." >&2
      rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
      return 0
    fi
  done

  stage_state_write "$stage" "$dir" "$config_hash" "$data_hash" "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
  mark_stage_done "$dir"
  echo "[${stage}] completed"
  rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
}
