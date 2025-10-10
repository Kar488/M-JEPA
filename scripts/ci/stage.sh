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
    if ! ssh-add -l 2>/dev/null | grep -q "$key_path"; then
      ssh-add "$key_path" >/dev/null 2>&1 || true
    fi
  fi

  export VAST_SSH_KEY_PATH="$key_path"
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
  : "${PHASE2_RECHECK_WALL_MINS:=120}"
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
    restore_env_var LOG_DIR "$prev_log_dir"
    return 0
  fi

  rm -f "$sentinel"
  rm -f "$incomplete"

  local wall_mins_raw="${PHASE2_RECHECK_WALL_MINS:-120}"
  local wall_mins="$wall_mins_raw"
  if ! [[ "$wall_mins" =~ ^[0-9]+$ ]] || [[ "$wall_mins" -le 0 ]]; then
    echo "[$step][warn] invalid PHASE2_RECHECK_WALL_MINS=${wall_mins_raw}; defaulting to 120" >&2
    wall_mins=120
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
    echo "[$step] graceful stop (rc=$rc); letting outputs flush." >&2
    mark_graceful_stop "$step"
    restore_env_var LOG_DIR "$prev_log_dir"
    return 0
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

  local best_json="${GRID_SOURCE_DIR}/best_grid_config.json"
  if [[ ! -f "$best_json" && -f "${GRID_DIR}/best_grid_config.json" ]]; then
    best_json="${GRID_DIR}/best_grid_config.json"
    phase2_promote_local_grid "$step"
  fi
  local summary_json="${GRID_DIR}/recheck_summary.json"

  if [[ ! -f "$best_json" ]]; then
    echo "[$step][fatal] expected ${best_json} but it was not created (GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>}). Rerun phase2_export or ensure GRID_EXP_ID points at the sweep lineage." >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    return 3
  fi

  local canonical_best="${GRID_DIR}/best_grid_config.json"
  if [[ "$best_json" != "$canonical_best" ]]; then
    mkdir -p "${GRID_DIR}"
    cp -f "$best_json" "$canonical_best"
    best_json="$canonical_best"
  fi
  if [[ -f "$summary_json" && "$summary_json" != "${GRID_DIR}/recheck_summary.json" ]]; then
    cp -f "$summary_json" "${GRID_DIR}/recheck_summary.json"
    summary_json="${GRID_DIR}/recheck_summary.json"
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
  #echo "ARGV after expansion: ${ARGV[@]}"

  # Best (from grid) → append last so it overrides YAML
  local -a BEST=()
  if (( ! skip_best )); then
    mapfile -t BEST < <(best_config_args "$section")
    expand_array_vars BEST
  fi

  local -a COMBINED=( "${ARGV[@]}" "${BEST[@]}" )

  # Dynamic allowlist from tool help (supports nargs='+')
  local -a OUT=()
  if (( skip_allowlist )); then
    OUT=("${COMBINED[@]}")
  else
    local -a ALLOWED
    mapfile -t ALLOWED < <(
      PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$MMBIN" run -n mjepa python "$APP_DIR/scripts/train_jepa.py" "$subcmd" --help |
          sed -n 's/.*\(--[a-z0-9-]\+\).*/\1/p' | sort -u
    )

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

    # --- JEPA mode: run_stage passes an array name ---
  if [[ "$s" != "wandb_agent" ]]; then
    if [ "$s" = "report" ]; then
      local BUDGET_MINS="${REPORT_TIME_BUDGET_MINS:-${HARD_WALL_MINS:-30}}"
      local SOFT=$((BUDGET_MINS*60))
      local GRACE="${KILL_AFTER_SECS:-60}"
      echo "[stage] wall budget=${BUDGET_MINS}m (${SOFT}s), grace=${GRACE}s"

      mkdir -p "$LOG_DIR"
      LOG="${LOG_DIR}/${s}.log"

      timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
        env PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
        python -m reports.build_wandb_report "${arr[@]}" \
        2>&1 | tee "$LOG_DIR/${s}.log"

      rc=${PIPESTATUS[0]}
      if [[ $rc -eq 0 ]]; then
        :
      elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
        echo "[INFO][$s] graceful stop (rc=$rc); not marking stage done; outputs should be flushed."
        mark_graceful_stop "$s"
        return 0
      else
        echo "[ERROR][$s] build_wandb_report failed with exit code $rc" >&2
        exit $rc
      fi
      return 0
    fi

    local -n arr="$1"; shift
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
    
    timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
      env PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
      "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -u "$APP_DIR/scripts/train_jepa.py" "$subcmd" "${arr[@]}" \
      2>&1 | tee "$LOG_DIR/${s}.log"
    rc=${PIPESTATUS[0]}
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
  OUT_DIR="$dir"
  export OUT_DIR
  local grid_read="${GRID_SOURCE_DIR:-${GRID_DIR:-<unset>}}"
  echo "[ci] STAGE=${stage} EXP_ID=${EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>} GRID_EXP_ID=${GRID_EXP_ID:-<unset>} FROZEN=${FROZEN:-0}" >&2
  echo "     READ: ARTIFACTS_DIR=${PRETRAIN_ARTIFACTS_DIR:-<unset>} GRID_DIR=${grid_read}" >&2
  echo "     WRITE: OUT_DIR=${OUT_DIR:-<unset>} EXPERIMENT_DIR=${EXPERIMENT_DIR:-<unset>}" >&2

  if (( FROZEN )) && [[ "${FORCE_UNFREEZE_GRID}" != "1" ]]; then
    case "$stage" in
      pretrain|grid|grid_search|phase1|phase2_sweep)
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
    local sentinel_path="$(stage_dir phase2_recheck)/recheck_done.ok"
    local winner_hint=""
    if [[ -n "${GRID_SOURCE_DIR:-}" && -f "${GRID_SOURCE_DIR}/best_grid_config.json" ]]; then
      winner_hint="${GRID_SOURCE_DIR}/best_grid_config.json"
    elif [[ -n "${GRID_DIR:-}" && -f "${GRID_DIR}/best_grid_config.json" ]]; then
      winner_hint="${GRID_DIR}/best_grid_config.json"
    fi
    if [[ -z "$winner_hint" ]]; then
      echo "[pretrain] winner not ready; skipping until grid/best_grid_config.json exists" >&2
      return 0
    fi
    if [[ ! -f "$sentinel_path" ]]; then
      echo "[pretrain] waiting for Phase-2 recheck sentinel at $sentinel_path; skipping" >&2
      return 0
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
          fi
        fi
      fi
    else
      rerun_reason="missing stamp"
    fi
  fi

  if (( skip )); then
    echo "[${stage}] cache hit - skipping"
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
        rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
        return $?
      fi
      ;;
    phase2_recheck)
      stage_log_dir="${dir}/logs"
      clear_graceful_stop "$stage" "$stage_log_dir"
      if ! run_phase2_recheck_stage "$dir" "$stage"; then
        rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
        return $?
      fi
      ;;
    phase2_export)
      stage_log_dir="${dir}/logs"
      clear_graceful_stop "$stage" "$stage_log_dir"
      if ! run_phase2_export_stage "$dir" "$stage"; then
        rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
        return $?
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
