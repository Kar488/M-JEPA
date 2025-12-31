#!/usr/bin/env bash
#set -x   # bash prints each command before executing
set -euo pipefail

if [[ "${BESTCFG_NO_EPOCHS:-}" == "1" && -z "${PRETRAIN_FALLBACK_EPOCHS:-}" ]]; then
  # When epochs are stripped from the best config, keep pretrain from collapsing
  # to a single epoch by providing a sensible default.
  export PRETRAIN_FALLBACK_EPOCHS=5
fi

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

ci_touch_file_dir() {
  local path="${1:-}"
  [[ -n "$path" ]] || return 0

  local dir
  dir="${path%/*}"
  if [[ "$dir" != "$path" ]]; then
    mkdir -p "$dir"
  fi
}

ci_prepare_ddp_attempts_file() {
  local path="${DDP_ATTEMPTS_FILE:-}"
  [[ -n "$path" ]] || return 0

  ci_touch_file_dir "$path"
  if [[ ! -e "$path" ]]; then
    printf '0' >"$path"
  fi
}

ci_mark_ddp_attempt_if_empty() {
  local path="${DDP_ATTEMPTS_FILE:-}"
  [[ -n "$path" ]] || return 0

  ci_prepare_ddp_attempts_file

  ci_touch_file_dir "$path"
  local current=""
  if [[ -f "$path" ]]; then
    current="$(<"$path")"
  fi
  if [[ "$current" =~ ^[0-9]+$ ]]; then
    if (( current > 0 )); then
      return 0
    fi
  fi

  printf '1' >"$path"
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

phase2_candidate_grid_dirs() {
  local -a roots=()

  add_root() {
    local candidate="$1"
    [[ -n "$candidate" ]] || return 0
    candidate="${candidate%/}"
    [[ -n "$candidate" ]] || return 0

    # Skip roots that are not writable to avoid fatal mkdir errors when running
    # on GitHub-hosted runners without the Vast data volume mounted.  mjepa_try_dir
    # creates the directory when possible; otherwise it returns non-zero and we
    # silently ignore the candidate.
    if ! mjepa_try_dir "$candidate" 2>/dev/null; then
      return 0
    fi

    local existing
    for existing in "${roots[@]}"; do
      if [[ "$existing" == "$candidate" ]]; then
        return 0
      fi
    done
    roots+=("$candidate")
  }

  add_root "${GRID_DIR:-}"
  add_root "${GRID_SOURCE_DIR:-}"
  if [[ -n "${EXPERIMENTS_ROOT:-}" && -n "${GRID_EXP_ID:-}" ]]; then
    add_root "${EXPERIMENTS_ROOT%/}/${GRID_EXP_ID}/grid"
  fi
  if [[ -n "${GRID_CACHE_DIR:-}" ]]; then
    add_root "${GRID_CACHE_DIR%/}"
    if [[ -n "${GRID_EXP_ID:-}" ]]; then
      add_root "${GRID_CACHE_DIR%/}/${GRID_EXP_ID}"
    fi
  fi
  add_root "/data/mjepa/cache/grid"
  add_root "/cache/grid"
  if [[ -n "${GRID_EXP_ID:-}" ]]; then
    add_root "/data/mjepa/cache/grid/${GRID_EXP_ID}"
    add_root "/cache/grid/${GRID_EXP_ID}"
  fi

  printf '%s\n' "${roots[@]}"
}

phase2_log_grid_roots() {
  local step_label="${1:-phase2}"
  local -a roots=()
  mapfile -t roots < <(phase2_candidate_grid_dirs)
  echo "[${step_label}] candidate grid roots: ${roots[*]:-<none>}" >&2
}

phase2_copy_file_if_exists() {
  local step_label="$1" src="$2" dst="$3" label="$4"
  [[ -n "$src" && -n "$dst" ]] || return 0
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")" || return 0
    if cp -f "$src" "$dst"; then
      echo "[${step_label}] copied ${label:-$(basename "$src")} -> ${dst}" >&2
    fi
  fi
}

phase2_sync_stage_dirs() {
  local step_label="$1" stage_name="$2" source_root="$3" target_root="$4"
  local src_logs="${source_root%/}/${stage_name}/logs"
  local src_outputs="${source_root%/}/${stage_name}/stage-outputs"
  local dst_logs="${target_root%/}/${stage_name}/logs"
  local dst_outputs="${target_root%/}/${stage_name}/stage-outputs"

  if [[ "$src_logs" != "$dst_logs" && -d "$src_logs" ]]; then
    mkdir -p "$dst_logs"
    cp -a "$src_logs/." "$dst_logs/" || true
    echo "[${step_label}] mirrored logs ${src_logs} -> ${dst_logs}" >&2
  fi
  if [[ "$src_outputs" != "$dst_outputs" && -d "$src_outputs" ]]; then
    mkdir -p "$dst_outputs"
    cp -a "$src_outputs/." "$dst_outputs/" || true
    echo "[${step_label}] mirrored outputs ${src_outputs} -> ${dst_outputs}" >&2
  fi
}

phase2_publish_recheck_artifacts() {
  local step_label="$1" source_root="$2"
  local -a roots=()
  mapfile -t roots < <(phase2_candidate_grid_dirs)
  local src_root="${source_root%/}"
  local sentinel_rel="phase2_recheck/recheck_done.ok"
  local incomplete_rel="phase2_recheck/recheck_incomplete.ok"

  local rel
  for rel in "best_grid_config.json" "recheck_summary.json" "phase2_winner_config.csv" "phase2_winner.txt" "phase2_sweep_id.txt" "$sentinel_rel" "$incomplete_rel"; do
    local src_path="${src_root}/${rel}"
    local root
    for root in "${roots[@]}"; do
      root="${root%/}"
      [[ "$root" == "$src_root" || -z "$root" ]] && continue
      local dst_path="${root}/${rel}"
      phase2_copy_file_if_exists "$step_label" "$src_path" "$dst_path" "$rel"
    done
  done

  local stage_name
  for stage_name in phase2_recheck phase2_export phase2_sweep; do
    local root
    for root in "${roots[@]}"; do
      root="${root%/}"
      [[ "$root" == "$src_root" || -z "$root" ]] && continue
      phase2_sync_stage_dirs "$step_label" "$stage_name" "$src_root" "$root"
    done
  done
}

phase2_try_recheck_fallback() {
  local step_label="$1" primary_root="$2"
  local primary_sentinel="${primary_root%/}/phase2_recheck/recheck_done.ok"
  local primary_summary="${primary_root%/}/recheck_summary.json"
  local primary_best="${primary_root%/}/best_grid_config.json"

  if [[ -f "$primary_sentinel" && -f "$primary_summary" && -f "$primary_best" ]]; then
    return 0
  fi

  local -a roots=()
  mapfile -t roots < <(phase2_candidate_grid_dirs)
  local root
  for root in "${roots[@]}"; do
    root="${root%/}"
    [[ -z "$root" || "$root" == "${primary_root%/}" ]] && continue
    local sentinel="${root}/phase2_recheck/recheck_done.ok"
    local summary="${root}/recheck_summary.json"
    local best="${root}/best_grid_config.json"
    if [[ -f "$sentinel" ]]; then
      echo "[${step_label}] fallback: found sentinel under ${root}; syncing into ${primary_root}" >&2
      phase2_copy_file_if_exists "$step_label" "$sentinel" "$primary_sentinel" "recheck_done.ok"
      phase2_copy_file_if_exists "$step_label" "$summary" "$primary_summary" "recheck_summary.json"
      phase2_copy_file_if_exists "$step_label" "$best" "$primary_best" "best_grid_config.json"
      phase2_publish_recheck_artifacts "$step_label" "$root"
      return 0
    fi
  done

  echo "[${step_label}] fallback: no cache/grid root exposed a recheck sentinel" >&2
  return 1
}

phase2_cleanup_stale_sweep_stub() {
  local step_label="$1" root="$2"
  local sweep_stub="${root%/}/phase2_sweep_id.txt"
  if [[ -f "$sweep_stub" ]]; then
    echo "[${step_label}][warn] removing stale phase2_sweep_id stub at ${sweep_stub}" >&2
    rm -f "$sweep_stub" || true
  fi
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
  echo "[${step_label}] syncing grid artifacts source=${source_dir} dest=${dest_dir}" >&2

  sync_pair() {
    local src="$1" dst="$2" label="$3"
    if [[ "$src" == "$dst" ]]; then
      return 0
    fi
    local src_exists=0 dst_exists=0
    [[ -f "$src" ]] && src_exists=1
    [[ -f "$dst" ]] && dst_exists=1
    if (( !src_exists && !dst_exists )); then
      return 0
    fi

    local src_mtime=0 dst_mtime=0
    if (( src_exists )); then
      src_mtime=$(stat -c %Y "$src" 2>/dev/null || echo 0)
    fi
    if (( dst_exists )); then
      dst_mtime=$(stat -c %Y "$dst" 2>/dev/null || echo 0)
    fi

    local from="$src" to="$dst"
    if (( dst_exists && (!src_exists || dst_mtime > src_mtime) )); then
      from="$dst"
      to="$src"
    fi

    mkdir -p "$(dirname "$to")" || return 0
    if ! cmp -s "$from" "$to" 2>/dev/null; then
      cp -f "$from" "$to"
      echo "[${step_label}] synced ${label:-$(basename "$from")} -> ${to}" >&2
    fi
  }

  sync_pair "${source_dir}/phase2_sweep_id.txt" "${dest_dir}/phase2_sweep_id.txt" "phase2_sweep_id.txt"
  sync_pair "${source_dir}/best_grid_config.json" "${dest_dir}/best_grid_config.json" "best_grid_config.json"
  sync_pair "${source_dir}/recheck_summary.json" "${dest_dir}/recheck_summary.json" "recheck_summary.json"

  local sentinel_src="${source_dir}/phase2_recheck/recheck_done.ok"
  local sentinel_dst="$(stage_dir phase2_recheck)/recheck_done.ok"
  sync_pair "$sentinel_src" "$sentinel_dst" "recheck_done.ok"
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

  if [[ -f "$dest" ]]; then
    if cmp -s "$chosen" "$dest" 2>/dev/null; then
      return 0
    fi
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
  phase2_log_grid_roots "$step"

  echo "[${step}] starting (logs=${dir}/logs stage_dir=${dir} grid_root=${GRID_DIR:-<unset>} source=${GRID_SOURCE_DIR:-<unset>})" >&2

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
  local sweep_wall="${PHASE2_SWEEP_WALL_MINS:-1160}"
  export HARD_WALL_MINS="$sweep_wall"

  local sweep_state="" sweep_seen_runs=""
  read -r sweep_state sweep_seen_runs < <(
    WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" SID="$SWEEP_ID" \
      python_inline - "$SWEEP_ID" <<'PY' || true
import os, sys

try:
    import wandb
except Exception:
    sys.exit(0)

sid = os.environ.get("SID") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not sid:
    sys.exit(0)

entity = os.environ.get("WANDB_ENTITY")
project = os.environ.get("WANDB_PROJECT")
parts = sid.split("/")
if len(parts) == 3:
    entity, project, sid = parts
elif len(parts) != 1 or not (entity and project):
    sys.exit(0)

state = ""
seen = ""
try:
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sid}")
    state = getattr(sweep, "state", None) or ""
    try:
        runs = api.runs(f"{entity}/{project}", filters={"sweep": sid})
        seen = str(sum(1 for _ in runs))
    except Exception:
        seen = ""
except Exception:
    sys.exit(0)

print(state)
print(seen)
PY
  )

  local sweep_exhausted=0
  if [[ "$sweep_state" =~ ^(FINISHED|CANCELED|CANCELLED)$ ]]; then
    sweep_exhausted=1
    echo "[$step][info] sweep state=${sweep_state}; skipping agent launch and preflight checks" >&2
  fi

  local -a GRID_VISIBLE_GPUS=()
  local gpu_count=0
  if (( ! sweep_exhausted )); then
    mapfile -t GRID_VISIBLE_GPUS < <(visible_gpu_ids)
    gpu_count="${#GRID_VISIBLE_GPUS[@]}"
  fi

  if [[ -z "${WANDB_COUNT:-}" ]]; then
    export WANDB_COUNT=4
  fi
  local phase2_total_count="$WANDB_COUNT"
  if [[ -n "$sweep_seen_runs" && "$sweep_seen_runs" =~ ^[0-9]+$ ]]; then
    if (( sweep_seen_runs >= phase2_total_count )); then
      sweep_exhausted=1
      echo "[$step][info] sweep already has ${sweep_seen_runs} runs; target count=${phase2_total_count}. Skipping agent launch." >&2
    else
      phase2_total_count=$(( phase2_total_count - sweep_seen_runs ))
      echo "[$step][info] sweep has ${sweep_seen_runs} existing runs; launching ${phase2_total_count} more to honour WANDB_COUNT." >&2
    fi
  fi

  : "${PHASE2_LABELED_DIR:=$APP_DIR/data/katielinkmoleculenet_benchmark/train}"
  : "${PHASE2_UNLABELED_DIR:=${DATA_ROOT:-$APP_DIR}/data/ZINC-canonicalized}"

  if (( ! sweep_exhausted )); then
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
    local cache_unlabeled_dir="${DATA_ROOT:-$APP_DIR}/cache/graphs_10m"
    if [[ "$PHASE2_UNLABELED_DIR" == "$cache_unlabeled_dir" ]]; then
      echo "[$step][fatal] PHASE2_UNLABELED_DIR points at the cache (${cache_unlabeled_dir}); set it to the ZINC corpus instead (e.g., ${DATA_ROOT:-$APP_DIR}/data/ZINC-canonicalized)." >&2
      restore_env_var LOG_DIR "$prev_log_dir"
      restore_env_var HARD_WALL_MINS "$prev_wall"
      return 2
    fi
  fi

  local agent_workers=1
  if (( sweep_exhausted )); then
    agent_workers=0
  else
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
  fi

  local launched=0

  if (( sweep_exhausted )); then
    echo "[$step] sweep already exhausted; skipping agent launch" >&2
  elif (( agent_workers == 1 || gpu_count <= 1 )); then
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

  local primary_root="${GRID_DIR%/}"
  local -a roots=()
  mapfile -t roots < <(phase2_candidate_grid_dirs)
  local root
  for root in "${roots[@]}"; do
    root="${root%/}"
    [[ -z "$root" || "$root" == "$primary_root" ]] && continue
    phase2_sync_stage_dirs "$step" "$step" "$primary_root" "$root"
  done

  echo "[${step}] finished (logs=${dir}/logs stage_dir=${dir})" >&2

  restore_env_var LOG_DIR "$prev_log_dir"
  restore_env_var HARD_WALL_MINS "$prev_wall"

  return 0
}

run_phase2_recheck_stage() {
  local dir="$1" step="$2"

  phase2_step_diag "$step"
  phase2_log_grid_roots "$step"

  : "${TOPK_RECHECK:=5}"
  : "${EXTRA_SEEDS:=3}"
  : "${PHASE2_METRIC:=val_rmse}"
  : "${PHASE2_DIRECTION:=min}"
  : "${PHASE2_UNLABELED_DIR:=${DATA_ROOT:-$APP_DIR}/data/ZINC-canonicalized}"
  : "${PHASE2_LABELED_DIR:=$APP_DIR/data/katielinkmoleculenet_benchmark/train}"
  : "${PHASE2_RECHECK_WALL_MINS:=540}"
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

  echo "[$step] appending recheck trials to sweep ${sweep_id} (TOPK_RECHECK=${TOPK_RECHECK} EXTRA_SEEDS=${EXTRA_SEEDS}); total runs will exceed the sweep-stage WANDB_COUNT." >&2

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

  echo "[${step}] starting (logs=${step_log_dir} recheck_dir=${recheck_dir} grid_root=${GRID_DIR:-<unset>} source=${GRID_SOURCE_DIR:-<unset>})" >&2

  phase2_try_recheck_fallback "$step" "${GRID_DIR:-${dir%/}}" || true

  local runs_csv_expected="${GRID_DIR}/phase2_export/stage-outputs/phase2_runs.csv"
  if [[ -f "$sentinel" && ! -f "$runs_csv_expected" ]]; then
    echo "[$step][warn] sentinel present but missing ${runs_csv_expected}; rerunning recheck" >&2
    rm -f "$sentinel"
  fi

  if [[ -f "$sentinel" ]]; then
    echo "[$step] sentinel present; skipping" >&2
    phase2_promote_local_grid "$step"
    phase2_sync_grid_artifacts "$step"
    phase2_publish_recheck_artifacts "$step" "${GRID_DIR:-${dir%/}}"
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

  local recheck_devices_raw="${PHASE2_RECHECK_FORCE_DEVICES:-${PHASE2_FORCE_DEVICES:-}}"
  local recheck_devices_per_run=0
  if [[ -n "$recheck_devices_raw" && "$recheck_devices_raw" =~ ^[0-9]+$ ]] && (( recheck_devices_raw > 0 )); then
    recheck_devices_per_run=$recheck_devices_raw
  fi

  local recheck_max_parallel="$recheck_gpu_count"
  if (( recheck_devices_per_run > 1 )); then
    if (( recheck_gpu_count > 0 )); then
      recheck_max_parallel=$(( recheck_gpu_count / recheck_devices_per_run ))
      if (( recheck_max_parallel < 1 )); then
        recheck_max_parallel=1
      fi
    else
      recheck_max_parallel=1
    fi
  fi

  if [[ -z "${PHASE2_RECHECK_AGENT_COUNT:-}" ]]; then
    local auto_recheck_agents=""
    if [[ -n "${PHASE2_AGENT_COUNT:-}" ]]; then
      if [[ "$PHASE2_AGENT_COUNT" =~ ^[0-9]+$ ]] && (( PHASE2_AGENT_COUNT > 0 )); then
        auto_recheck_agents="$PHASE2_AGENT_COUNT"
      else
        echo "[$step][warn] ignoring non-numeric PHASE2_AGENT_COUNT='${PHASE2_AGENT_COUNT}' for recheck" >&2
      fi
    elif (( recheck_gpu_count > 1 )); then
      auto_recheck_agents="$recheck_gpu_count"
    fi

    if [[ -n "$auto_recheck_agents" ]]; then
      if [[ "$auto_recheck_agents" =~ ^[0-9]+$ ]] && (( recheck_max_parallel > 0 )); then
        local desired_agents="$auto_recheck_agents"
        local clamped_agents="$desired_agents"
        if (( recheck_devices_per_run > 1 )) && (( desired_agents > recheck_max_parallel )); then
          clamped_agents="$recheck_max_parallel"
          echo "[$step][warn] reducing recheck workers from ${desired_agents} to ${clamped_agents} to honour ${recheck_devices_per_run} GPU(s) per run" >&2
        fi
        export PHASE2_RECHECK_AGENT_COUNT="$clamped_agents"
      else
        export PHASE2_RECHECK_AGENT_COUNT="$auto_recheck_agents"
      fi
    fi
  fi
  unset __RECHECK_VISIBLE_GPUS || true

  export PHASE2_RECHECK_HEARTBEAT="$heartbeat_path"
  export PHASE2_RECHECK_SENTINEL="$sentinel"
  export PHASE2_RECHECK_RESUME=1
  export PHASE2_RECHECK_INCOMPLETE="$incomplete"
  if [[ -z "${PHASE2_RECHECK_FORCE_DEVICES:-}" && -n "${PHASE2_FORCE_DEVICES:-}" ]]; then
    export PHASE2_RECHECK_FORCE_DEVICES="$PHASE2_FORCE_DEVICES"
  fi

  local log_path="${step_log_dir}/recheck_topk.log"

  # Ensure the grid root exists and is writable so recheck outputs land in the
  # expected location (GRID_DIR).  When GRID_DIR is missing, the recheck helper
  # silently falls back to a temp directory, which leaves the core pipeline
  # without a recheck_summary.json to export or collect.
  if [[ -n "${GRID_DIR:-}" ]]; then
    mkdir -p "${GRID_DIR}" || true
  fi

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
      --runs-csv "${GRID_DIR}/phase2_export/stage-outputs/phase2_runs.csv" \
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
    phase2_try_recheck_fallback "$step" "${GRID_DIR:-${dir%/}}" || true
  fi

  if [[ ! -f "$sentinel" ]]; then
    echo "[$step][fatal] sentinel missing after successful recheck execution: $sentinel" >&2
    phase2_cleanup_stale_sweep_stub "$step" "${GRID_DIR:-${dir%/}}"
    restore_env_var LOG_DIR "$prev_log_dir"
    return 4
  fi

  rm -f "$incomplete" 2>/dev/null || true

  phase2_sync_grid_artifacts "$step"

  local outputs_dir="${dir}/stage-outputs"
  mkdir -p "$outputs_dir"

  local best_json="${GRID_DIR}/best_grid_config.json"
  local summary_json="${GRID_DIR}/recheck_summary.json"

  if [[ -f "$best_json" ]]; then
    cp -f "$best_json" "${outputs_dir}/best_grid_config.json"
  fi
  if [[ -f "$summary_json" ]]; then
    cp -f "$summary_json" "${outputs_dir}/recheck_summary.json"
  fi

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
  phase2_publish_recheck_artifacts "$step" "${GRID_DIR:-${dir%/}}"
  echo "[${step}] finished (logs=${step_log_dir} recheck_dir=${recheck_dir})" >&2
  return 0
}

run_phase2_export_stage() {
  local dir="$1" step="$2"

  phase2_step_diag "$step"
  phase2_log_grid_roots "$step"

  phase2_promote_local_grid "$step"
  phase2_sync_grid_artifacts "$step"
  phase2_promote_grid_artifacts "$step"

  echo "[${step}] starting (logs=${dir}/logs export_dir=${dir} grid_root=${GRID_DIR:-<unset>} source=${GRID_SOURCE_DIR:-<unset>})" >&2

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
  local primary_root="${GRID_DIR:-${dir%/}}"

  phase2_try_recheck_fallback "$step" "$primary_root" || true
  echo "[${step}] checking recheck artifacts (sentinel=${sentinel} summary=${GRID_DIR}/recheck_summary.json best=${GRID_DIR}/best_grid_config.json)" >&2

  if [[ -f "$incomplete" ]]; then
    echo "[$step][fatal] recheck incomplete marker present: $incomplete. Rerun phase2_recheck before exporting." >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    return 4
  fi
  if [[ ! -f "$sentinel" ]]; then
    echo "[$step][fatal] recheck sentinel missing: $sentinel. Complete phase2_recheck before export." >&2
    phase2_cleanup_stale_sweep_stub "$step" "$primary_root"
    restore_env_var LOG_DIR "$prev_log_dir"
    return 4
  fi

  local best_json="${GRID_DIR}/best_grid_config.json"
  local summary_json="${GRID_DIR}/recheck_summary.json"

  if [[ ! -f "$best_json" || ! -f "$summary_json" ]]; then
    phase2_try_recheck_fallback "$step" "$primary_root" || true
  fi

  if [[ ! -f "$best_json" ]]; then
    echo "[$step][fatal] expected best_grid_config.json at $best_json but it is missing (GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>})." >&2
    phase2_cleanup_stale_sweep_stub "$step" "$primary_root"
    restore_env_var LOG_DIR "$prev_log_dir"
    return 3
  fi

  if [[ ! -f "$summary_json" ]]; then
    echo "[$step][fatal] expected recheck_summary.json at $summary_json but it is missing (GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>})." >&2
    phase2_cleanup_stale_sweep_stub "$step" "$primary_root"
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

  local runs_csv_src="${GRID_DIR}/phase2_export/stage-outputs/phase2_runs.csv"
  if [[ -f "$runs_csv_src" ]]; then
    cp -f "$runs_csv_src" "${outputs_dir}/phase2_runs.csv"
  else
    echo "[$step][warn] expected phase2_runs.csv at ${runs_csv_src} but it is missing; rerun phase2_recheck to emit it" >&2
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

  local archived_sweep_id="${outputs_dir}/phase2_sweep_id.txt"
  local working_sweep_id="${GRID_DIR%/}/phase2_sweep_id.txt"
  local source_sweep_id=""
  if [[ -n "${GRID_SOURCE_DIR:-}" ]]; then
    source_sweep_id="${GRID_SOURCE_DIR%/}/phase2_sweep_id.txt"
  fi

  if [[ -f "$working_sweep_id" ]]; then
    mkdir -p "$(dirname "$archived_sweep_id")"
    cp -f "$working_sweep_id" "$archived_sweep_id"
    rm -f "$working_sweep_id"
    echo "[$step] archived sweep id to ${archived_sweep_id} and cleared ${working_sweep_id}" >&2
  fi

  if [[ -n "$source_sweep_id" && "$source_sweep_id" != "$working_sweep_id" && -f "$source_sweep_id" ]]; then
    if [[ ! -f "$archived_sweep_id" ]]; then
      mkdir -p "$(dirname "$archived_sweep_id")"
      cp -f "$source_sweep_id" "$archived_sweep_id"
    fi
    rm -f "$source_sweep_id"
    echo "[$step] cleared lineage sweep stub ${source_sweep_id} after export" >&2
  fi

  printf "[%s] validated Phase-2 best configuration\n" "$step" | tee -a "${step_log_dir}/export.log" >/dev/null

  phase2_publish_recheck_artifacts "$step" "$primary_root"
  echo "[${step}] finished (logs=${step_log_dir} export_dir=${dir})" >&2

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
    local allowlist_skipped=0

    if py=$(python_bin 2>/dev/null); then
      if help_output=$(PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$py" "$APP_DIR/scripts/train_jepa.py" "$subcmd" --help 2>&1
      ); then
        mapfile -t ALLOWED < <(printf '%s\n' "$help_output" |
          sed -n 's/.*\(--[A-Za-z0-9_-]\+\).*/\1/p' | sort -u)
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
          sed -n 's/.*\(--[A-Za-z0-9_-]\+\).*/\1/p' | sort -u)
      else
        local status=$?
        printf '[stage:%s] failed to execute %s --help even via micromamba python (exit %d).\n' \
          "$s" "$subcmd" "$status" >&2
        printf '%s\n' "$help_output" >&2
        if [[ "$s" == "tox21" ]]; then
          echo "[stage:$s] warning: skipping help/introspection preflight; proceeding without allowlist" >&2
          allowlist_skipped=1
        else
          echo "[diag] about to exit: stage help resolution failed (stage=${s} status=${status} subcmd=${subcmd})" >&2
          return "$status"
        fi
      fi
    fi

    if (( allowlist_skipped )); then
      OUT=("${COMBINED[@]}")
    else
      local -A ALLOWED_NORMALIZED=()
      local __allow_token
      for __allow_token in "${ALLOWED[@]}"; do
        [[ -z "${__allow_token}" ]] && continue
        local __norm="${__allow_token//_/-}"
        ALLOWED_NORMALIZED["${__norm}"]=1
      done

      local i=0 j=0
      while (( i < ${#COMBINED[@]} )); do
        local f="${COMBINED[$i]}"
        if [[ "$f" == --* ]]; then
          local check_flag="$f"
          local has_inline_value=0
          if [[ "$check_flag" == *=* ]]; then
            check_flag="${check_flag%%=*}"
            has_inline_value=1
          fi
          local canonical_flag="${check_flag//_/-}"
          if [[ -n "${ALLOWED_NORMALIZED[${canonical_flag}]:-}" ]]; then
            OUT+=("$f")
            j=$((i+1))
            if (( ! has_inline_value )); then
              while (( j < ${#COMBINED[@]} )) && [[ "${COMBINED[$j]}" != --* ]]; do
                OUT+=("${COMBINED[$j]}"); ((j++))
              done
            fi
            i=$j; continue
          fi
        fi
        ((++i))
      done
    fi
  fi

  prune_empty_args OUT

  dedupe_stage_args OUT

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

    local no_calib_requested=0
    for token in "${OUT[@]}"; do
      if [[ "$token" == "--no-calibrate" ]]; then
        no_calib_requested=1
        break
      fi
    done
    if (( ! no_calib_requested )); then
      local no_calib_env="${TOX21_NO_CALIBRATE:-}"
      local calibrate_env="${TOX21_CALIBRATE:-}"
      case "${no_calib_env,,}" in
        1|true|yes|on)
          no_calib_requested=1
          ;;
      esac
      if (( ! no_calib_requested )); then
        case "${calibrate_env,,}" in
          0|false|no|off)
            no_calib_requested=1
            ;;
        esac
      fi
      if (( no_calib_requested )); then
        OUT+=("--no-calibrate")
      fi
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
      ((idx+=1))
    done
    OUT=("${FILTERED[@]}")
  fi

  STAGE_ARGS=("${OUT[@]}")
}

dedupe_stage_args() {
  local arr_name="${1:?array name required}"
  local -n __arr="$arr_name"

  if (( ${#__arr[@]} == 0 )); then
    return 0
  fi

  local -a __dedup_entries=()
  declare -A __dedup_index=()
  local __sep=$'\037'
  local __idx=0

  while (( __idx < ${#__arr[@]} )); do
    local __token="${__arr[__idx]}"
    if [[ "${__token}" == --* ]]; then
      local __style="split"
      local __original_flag="${__token}"
      local __normalized_flag="${__original_flag//_/-}"
      local -a __values=()
      local __next=$((__idx + 1))

      if [[ "${__token}" == *=* ]]; then
        __style="joined"
        __original_flag="${__token%%=*}"
        __normalized_flag="${__original_flag//_/-}"
        local __joined_value="${__token#*=}"
        if [[ -n "${__joined_value}" ]]; then
          __values+=("${__joined_value}")
        fi
      else
        while (( __next < ${#__arr[@]} )) && [[ "${__arr[__next]}" != --* ]]; do
          __values+=("${__arr[__next]}")
          ((__next++))
        done
      fi

      if [[ -v __dedup_index["${__normalized_flag}"] ]]; then
        __dedup_entries[${__dedup_index["${__normalized_flag}"]}]=""
      fi

      local __serial="__FLAG__${__style}${__sep}${__original_flag}"
      local __part
      for __part in "${__values[@]}"; do
        __serial+="${__sep}${__part}"
      done

      __dedup_index["${__normalized_flag}"]=${#__dedup_entries[@]}
      __dedup_entries+=("${__serial}")

      if [[ "${__style}" == "split" ]]; then
        __idx=${__next}
      else
        ((__idx++))
      fi
      continue
    fi

    __dedup_entries+=("__POS__${__token}")
    ((__idx++))
  done

  local -a __flattened=()
  local __entry
  for __entry in "${__dedup_entries[@]}"; do
    [[ -z "${__entry}" ]] && continue
    if [[ "${__entry}" == __FLAG__* ]]; then
      local __payload="${__entry#__FLAG__}"
      IFS="${__sep}" read -r -a __parts <<<"${__payload}"
      local __style="${__parts[0]}"
      local __flag="${__parts[1]}"
      local -a __vals=("${__parts[@]:2}")
      if [[ "${__style}" == "joined" && ${#__vals[@]} -eq 1 ]]; then
        if [[ -n "${__vals[0]}" ]]; then
          __flattened+=("${__flag}=${__vals[0]}")
        else
          __flattened+=("${__flag}")
        fi
      else
        __flattened+=("${__flag}")
        if (( ${#__vals[@]} )); then
          __flattened+=("${__vals[@]}")
        fi
      fi
    elif [[ "${__entry}" == __POS__* ]]; then
      __flattened+=("${__entry#__POS__}")
    fi
  done

  __arr=("${__flattened[@]}")
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

ci_stage_resolve_python_runner() {
  local -n __out_env="$1"
  local -n __out_bin="$2"
  local -n __out_args="$3"
  local __stage_label="${4:-stage}"

  __out_env=()
  __out_bin=""
  __out_args=(-u)

  local prefer_system=0
  local system_bin="${MJEPACI_SYSTEM_PYTHON_BIN:-}"
  if [[ "${MJEPACI_FORCE_SYSTEM_PYTHON:-}" == "1" ]]; then
    prefer_system=1
  fi

  if (( ! prefer_system )); then
    if ensure_micromamba; then
      __out_env=("$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1)
      __out_bin="python"
      return 0
    fi
    echo "[stage:${__stage_label}] warn: micromamba unavailable; falling back to system python" >&2
    prefer_system=1
  fi

  if [[ -z "$system_bin" ]]; then
    if py=$(python_bin 2>/dev/null); then
      system_bin="$(command -v "$py" 2>/dev/null || true)"
      if [[ -z "$system_bin" ]]; then
        system_bin="$py"
      fi
    fi
  fi

  if [[ -z "$system_bin" ]]; then
    echo "[stage:${__stage_label}] error: unable to resolve system python interpreter" >&2
    return 1
  fi

  __out_env=(env PYTHONUNBUFFERED=1)
  __out_bin="$system_bin"
  return 0
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

      local BUDGET_MINS="${REPORT_TIME_BUDGET_MINS:-${HARD_WALL_MINS:-1500}}"
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
      grid) : "${BUDGET_MINS:=${HARD_WALL_MINS:-1500}}" ;;
      pretrain|finetune) : "${BUDGET_MINS:=${HARD_WALL_MINS:-1500}}" ;;
      *) : "${BUDGET_MINS:=${HARD_WALL_MINS:-1500}}" ;;
    esac
    local SOFT="$((BUDGET_MINS*60))"   # convert minutes → seconds
    local GRACE="${KILL_AFTER_SECS:-60}"
    echo "[stage] wall budget=${BUDGET_MINS}m (${SOFT}s), grace=${GRACE}s"

    mkdir -p "$LOG_DIR"
    LOG="${LOG_DIR}/${s}.log"
    local -a stage_launch_env=()
    local stage_python_bin=""
    local -a stage_python_args=()
    if ! ci_stage_resolve_python_runner stage_launch_env stage_python_bin stage_python_args "$s"; then
      echo "[diag] about to exit: unable to resolve python runner (stage=${s})" >&2
      exit 1
    fi
    local -a python_runner_cmd=("${stage_launch_env[@]}" "$stage_python_bin" "${stage_python_args[@]}")
    local -a ddp_launcher=()
    local previous_world="${WORLD_SIZE-}"
    local previous_master_addr="${MASTER_ADDR-}"
    local previous_master_port="${MASTER_PORT-}"
    local ddp_env_modified=0
    local -a entrypoint_args=("${arr[@]}")
    local -a entrypoint=()
    local ddp_enabled=0

    local replace_stage_flag
    replace_stage_flag() {
      local flag="$1" value="$2"
      local -a rebuilt=()
      local replaced=0
      local i=0
      while (( i < ${#entrypoint_args[@]} )); do
        local token="${entrypoint_args[$i]}"
        if [[ "$token" == "$flag" ]]; then
          if (( ! replaced )); then
            rebuilt+=("$flag" "$value")
            replaced=1
          fi
          if (( i + 1 < ${#entrypoint_args[@]} )); then
            ((i+=2))
          else
            ((i+=1))
          fi
          continue
        elif [[ "$token" == ${flag}=* ]]; then
          if (( ! replaced )); then
            rebuilt+=("$flag" "$value")
            replaced=1
          fi
          ((++i))
          continue
        fi
        rebuilt+=("$token")
        ((++i))
      done
      if (( ! replaced )); then
        rebuilt+=("$flag" "$value")
      fi
      entrypoint_args=("${rebuilt[@]}")
    }

    local set_devices_arg
    set_devices_arg() {
      replace_stage_flag "--devices" "$1"
    }

    local set_arg_value
    set_arg_value() {
      replace_stage_flag "$1" "$2"
    }

    local ddp_stage=""
    case "$s" in
      finetune|tox21) ddp_stage="$s" ;;
    esac

    if [[ -n "$ddp_stage" ]]; then
      ci_prepare_ddp_attempts_file
    fi

    local force_cpu_execution=0
    local preflight_forced_single=0
    local preflight_reason=""
    local preflight_marked=0
    local preflight_fallback_logged=0
    local requested_devices_numeric=0
    local fallback_devices_value=""

    if [[ -n "$ddp_stage" ]]; then
      local devices_idx=-1
      local devices_joined=0
      local requested_devices=""
      local i=0
      local detected_cuda_devices=""
      while (( i < ${#entrypoint_args[@]} )); do
        local token="${entrypoint_args[$i]}"
        if [[ "$token" == "--devices" ]]; then
          devices_idx=$i
          if (( i + 1 < ${#entrypoint_args[@]} )); then
            requested_devices="${entrypoint_args[$((i + 1))]}"
          fi
          break
        elif [[ "$token" == --devices=* ]]; then
          devices_idx=$i
          devices_joined=1
          requested_devices="${token#--devices=}"
          break
        fi
        ((++i))
      done

      if [[ -z "$requested_devices" ]]; then
        case "$ddp_stage" in
          finetune)
            requested_devices="${FINETUNE_DEVICES:-}"
            ;;
          tox21)
            requested_devices="${TOX21_DEVICES:-}"
            ;;
        esac
      fi

      if [[ "$requested_devices" =~ ^[0-9]+$ ]]; then
        requested_devices_numeric=$(( requested_devices + 0 ))
      fi

      if (( requested_devices_numeric > 1 )); then
        local probe_output=""
        local ddp_supported=0
        if probe_output=$("${python_runner_cmd[@]}" - <<'PY'
import os
import sys

try:
    import torch  # noqa: F401
    import torch.distributed.run  # noqa: F401
except Exception:
    print("bummed on torch error")
    sys.exit(1)

mask = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
cuda = getattr(torch, "cuda", None)
count = 0

is_available = False
if cuda is not None:
    try:
        is_available = bool(callable(getattr(cuda, "is_available", None)) and cuda.is_available())
    except Exception:
        is_available = False

if mask and is_available:
    devices = [entry.strip() for entry in mask.split(",") if entry.strip()]
    count = len(devices)
elif mask:
    count = 0
elif is_available:
    try:
        count = int(cuda.device_count())
    except Exception:
        count = 0
print(max(count, 0))
PY
        ); then
          ddp_supported=1
        fi

        local effective_devices=$requested_devices_numeric
        local available_devices=0
        if (( ddp_supported )); then
          available_devices="${probe_output:-0}"
          available_devices="${available_devices//$'\r'/}"
          available_devices="${available_devices//$'\n'/}"
          available_devices="${available_devices//[[:space:]]/}"
          if [[ -z "$available_devices" ]]; then
            available_devices=0
          elif [[ "$available_devices" =~ ^[0-9]+$ ]]; then
            available_devices=$(( available_devices + 0 ))
          else
            available_devices=0
          fi
          detected_cuda_devices="$available_devices"
          if (( available_devices <= 0 )); then
            echo "[stage:$s] no CUDA devices detected; falling back to single-process execution" >&2
            effective_devices=1
            force_cpu_execution=1
            preflight_forced_single=1
            preflight_reason="no_cuda_devices"
          elif (( available_devices < effective_devices )); then
            echo "[stage:$s] requested ${effective_devices} devices but only ${available_devices} visible; clamping" >&2
            effective_devices=$available_devices
            if (( available_devices <= 1 )); then
              preflight_forced_single=1
              preflight_reason="insufficient_cuda:${available_devices}"
            fi
          fi
        else
          echo "[stage:$s] torch.distributed.run unavailable; falling back to single-process execution" >&2
          effective_devices=1
          preflight_forced_single=1
          preflight_reason="ddp_unavailable"
        fi

        if (( effective_devices > 1 && ddp_supported )); then
          local world="${WORLD_SIZE:-}"
          if [[ -z "$world" || "$world" -le 1 ]]; then
            local ddp_port="${MASTER_PORT:-}"
            if [[ -z "$ddp_port" ]]; then
              ddp_port=$(( (RANDOM % 20000) + 15000 ))
            fi
            export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
            export MASTER_PORT="$ddp_port"
            export WORLD_SIZE="$effective_devices"
            ddp_env_modified=1
            echo "[stage:$s] enabling torch.distributed.run (nproc_per_node=${effective_devices}, master_port=${MASTER_PORT})" >&2
          else
            echo "[stage:$s] WORLD_SIZE=${world}; assuming external launcher configured DDP" >&2
          fi
          ddp_launcher=(-m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=${effective_devices}")
          ddp_enabled=1
          preflight_forced_single=0
          preflight_reason=""
        else
          effective_devices=1
        fi

        if (( effective_devices != requested_devices_numeric )); then
          if (( devices_idx >= 0 )); then
            if (( devices_joined )); then
              entrypoint_args[$devices_idx]="--devices=${effective_devices}"
            else
              entrypoint_args[$((devices_idx + 1))]="${effective_devices}"
            fi
          else
            entrypoint_args+=("--devices" "${effective_devices}")
          fi
        fi
      fi
    fi

    if [[ -n "$ddp_stage" ]]; then
      local requested_device=""
      local device_token=""
      local idx=0
      while (( idx < ${#entrypoint_args[@]} )); do
        device_token="${entrypoint_args[$idx]}"
        if [[ "$device_token" == "--device" ]]; then
          if (( idx + 1 < ${#entrypoint_args[@]} )); then
            requested_device="${entrypoint_args[$((idx + 1))]}"
          fi
          break
        elif [[ "$device_token" == --device=* ]]; then
          requested_device="${device_token#--device=}"
          break
        fi
        ((idx+=1))
      done
      local normalized_device="${requested_device,,}"
      if [[ -z "$normalized_device" ]]; then
        normalized_device="cuda"
      fi

      local cuda_probe_output=""
      local cuda_count=""
      if cuda_probe_output=$("${python_runner_cmd[@]}" - <<'PY'
import sys

try:
    import torch
except Exception:
    print("bummed on torch import")
    torch = None

count = 0
is_available = False
if torch is not None:
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            is_callable = callable(getattr(cuda, "is_available", None))
        except Exception:
            is_callable = False
        if is_callable:
            try:
                is_available = bool(cuda.is_available())
            except Exception:
                is_available = False
        if is_available:
            try:
                count = int(cuda.device_count())
            except Exception:
                count = 0
print(max(count, 0))
PY
      ); then
        cuda_count="${cuda_probe_output:-0}"
        cuda_count="${cuda_count//$'\r'/}"
        cuda_count="${cuda_count//$'\n'/}"
        cuda_count="${cuda_count//[[:space:]]/}"
      else
        cuda_count="0"
      fi

      if [[ -z "$cuda_count" ]]; then
        cuda_count="0"
      fi

      if [[ -n "$detected_cuda_devices" ]]; then
        cuda_count="$detected_cuda_devices"
      fi

      local numeric_cuda_count="$cuda_count"
      if [[ "$numeric_cuda_count" =~ ^[0-9]+$ ]]; then
        numeric_cuda_count=$(( numeric_cuda_count + 0 ))
      else
        numeric_cuda_count=0
      fi

      if [[ "$normalized_device" == cuda* && $numeric_cuda_count -eq 0 ]]; then
        force_cpu_execution=1
      fi

      if (( force_cpu_execution )); then
        ddp_launcher=()
        ddp_enabled=0
        set_devices_arg 1
        set_arg_value "--device" "cpu"
        set_arg_value "--bf16" "0"
        echo "[stage:$s] warn: CUDA unavailable; forcing CPU execution (devices=1, device=cpu, bf16=0)" >&2
        preflight_forced_single=1
        if [[ -z "$preflight_reason" ]]; then
          preflight_reason="no_cuda_devices"
        fi
      fi

      arr=("${entrypoint_args[@]}")

      if (( preflight_forced_single )) && (( !preflight_marked )) && (( requested_devices_numeric > 1 )); then
        ci_mark_ddp_attempt_if_empty
        preflight_marked=1
      fi

      if (( preflight_forced_single )); then
        local scan_idx=0
        fallback_devices_value=""
        while (( scan_idx < ${#entrypoint_args[@]} )); do
          local token="${entrypoint_args[$scan_idx]}"
          if [[ "$token" == "--devices" ]]; then
            if (( scan_idx + 1 < ${#entrypoint_args[@]} )); then
              fallback_devices_value="${entrypoint_args[$((scan_idx + 1))]}"
            fi
            break
          elif [[ "$token" == --devices=* ]]; then
            fallback_devices_value="${token#--devices=}"
            break
          fi
          ((scan_idx+=1))
        done
        if [[ -z "$fallback_devices_value" ]]; then
          fallback_devices_value="1"
        fi
      fi
    fi

    local build_entrypoint
    build_entrypoint() {
      entrypoint=("$APP_DIR/scripts/train_jepa.py" "$subcmd" "${entrypoint_args[@]}")
    }

    local fallback_attempted=0
    while true; do
      build_entrypoint

      local -a stage_cmd=("${python_runner_cmd[@]}")
      local using_ddp=0
      if (( ${#ddp_launcher[@]} )); then
        stage_cmd+=("${ddp_launcher[@]}" "${entrypoint[@]}")
        using_ddp=1
      else
        stage_cmd+=("${entrypoint[@]}")
      fi

      local stage_python_cmd_str=""
      if (( ${#stage_cmd[@]} )); then
        printf -v stage_python_cmd_str '%q ' "${stage_cmd[@]}"
        stage_python_cmd_str=${stage_python_cmd_str% }
      fi

      if (( preflight_forced_single )) && (( !preflight_fallback_logged )); then
        local fallback_msg="[stage:$s] warn: distributed launch failed"
        case "$preflight_reason" in
          no_cuda_devices)
            fallback_msg+=" (no CUDA devices detected)"
            ;;
          ddp_unavailable)
            fallback_msg+=" (torch.distributed.run unavailable)"
            ;;
          insufficient_cuda:*)
            local avail="${preflight_reason#insufficient_cuda:}"
            if (( requested_devices_numeric > 1 )); then
              fallback_msg+=" (requested ${requested_devices_numeric} but only ${avail} visible)"
            else
              fallback_msg+=" (only ${avail} CUDA devices visible)"
            fi
            ;;
          "")
            ;;
          *)
            fallback_msg+=" (${preflight_reason})"
            ;;
        esac
        if [[ -n "$fallback_devices_value" ]]; then
          fallback_msg+="; retrying with --devices ${fallback_devices_value}"
        else
          fallback_msg+="; retrying with single-process execution"
        fi
        echo "$fallback_msg" >&2
        preflight_fallback_logged=1
      fi
      echo "[diag] stage python command (stage=${s}): ${stage_python_cmd_str}" >&2
      if (( using_ddp )); then
        ci_prepare_ddp_attempts_file
      fi
      set +e
      timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
        env PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "${stage_cmd[@]}" \
        2>&1 | tee "$LOG_DIR/${s}.log"
      local timeout_rc=${PIPESTATUS[0]}
      set -e
      rc=$timeout_rc
      if (( ddp_env_modified && using_ddp )); then
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
        ddp_env_modified=0
      fi
      # 0   = success
      # 124 = 'timeout' exceeded (we later sent SIGTERM/SIGKILL)
      # 143 = terminated by SIGTERM (128+15)
      # 137 = killed by SIGKILL   (128+9)
      if [[ $rc -eq 0 ]]; then
        break
      elif [[ $rc -eq 2 ]]; then
        echo "[INFO][wandb_agent] no runs left for sweep (rc=2); treating as success."
        return 0
      elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
        echo "[INFO][$s] graceful stop (rc=$rc); not marking stage done; outputs should be flushed."
        mark_graceful_stop "$s"
        return 0
      elif (( using_ddp )) && (( ddp_enabled )) && (( ! fallback_attempted )); then
        fallback_attempted=1
        ddp_launcher=()
        set_devices_arg 1
        arr=("${entrypoint_args[@]}")
        ci_mark_ddp_attempt_if_empty
        echo "[stage:$s] warn: distributed launch failed (rc=$rc); retrying with --devices 1" >&2
        echo "[diag] stage ddp fallback (stage=${s} rc=${rc} command=${stage_python_cmd_str})" >&2
        continue
      else
        echo "[ERROR][$s] train_jepa.py failed with exit code $rc" >&2
        echo "[diag] about to exit: train_jepa execution failed (stage=${s} rc=${rc} command=${stage_python_cmd_str})" >&2
        exit $rc
      fi
    done
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
    local SOFT=$(( (${HARD_WALL_MINS:-480})*60 ))
    local GRACE="${KILL_AFTER_SECS:-60}"
    echo "[wandb_agent] wall budget=${SOFT}s, grace=${GRACE}s"

    local -a stage_launch_env=()
    local stage_python_bin=""
    local -a stage_python_args=()
    if ! ci_stage_resolve_python_runner stage_launch_env stage_python_bin stage_python_args "wandb_agent"; then
      echo "[diag] about to exit: unable to resolve python runner (stage=wandb_agent)" >&2
      exit 1
    fi
    local -a python_runner_cmd=("${stage_launch_env[@]}" "$stage_python_bin" "${stage_python_args[@]}")

    set +e
    timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
      "${python_runner_cmd[@]}" -m wandb agent --count ${WANDB_COUNT:-50} "$SID" \
      2>&1 | tee "$LOG"
    local -a wandb_agent_status=("${PIPESTATUS[@]}")
    set -e

    rc=${wandb_agent_status[0]:-$?}
    # --- FIX START: Immediately handle WandB "No runs" exit code ---
    if [[ $rc -eq 2 ]]; then
      echo "[INFO][wandb_agent] agent returned rc=2 (sweeps done); treating as success."
      return 0
    fi
    # --- FIX END ---
    
    local timeout_rc=0
    if [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
      timeout_rc=$rc
    fi
    local agent_runs_started=0
    if [[ -f "$LOG" ]]; then
      agent_runs_started=$(grep -c "About to run command" "$LOG" || true)
    fi

    local planned_runs="${WANDB_COUNT:-}"
    local meets_planned=0
    if [[ "$planned_runs" =~ ^[0-9]+$ ]] && (( planned_runs > 0 )) && (( agent_runs_started >= planned_runs )); then
      meets_planned=1
    fi

    local sweep_state=""
    if [[ -n "$SID" ]]; then
      sweep_state=$(
        WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" SID="$SID" \
          "${python_runner_cmd[@]}" - "$SID" <<'PY' || true
import os, sys

try:
    import wandb
except Exception:
    sys.exit(0)

sid = os.environ.get("SID") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not sid:
    sys.exit(0)

entity = os.environ.get("WANDB_ENTITY")
project = os.environ.get("WANDB_PROJECT")
parts = sid.split("/")
if len(parts) == 3:
    entity, project, sid = parts
elif len(parts) != 1 or not (entity and project):
    sys.exit(0)

try:
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sid}")
    state = getattr(sweep, "state", None) or ""
    print(state)
except Exception:
    sys.exit(0)
PY
)
    fi

    local log_has_no_runs=0
    local log_has_api_instability=0
    local log_has_rapid_failures=0
    local log_has_fatal_rapid_failures=0
    if [[ -f "$LOG" ]]; then
      if grep -qi "No runs found" "$LOG"; then
        log_has_no_runs=1
      fi
      if grep -qiE "Error while calling W&B API: An internal error occurred|Response \[500\]|Network error \(HTTPError\)" "$LOG"; then
        log_has_api_instability=1
      fi
      if grep -qiE "failed runs in the first [0-9]+ seconds" "$LOG"; then
        log_has_rapid_failures=1
      fi
    fi

    if (( log_has_rapid_failures )); then
      log_has_fatal_rapid_failures=1
      echo "[wandb_agent][error] agent aborted after repeated early failures; see ${LOG}" >&2
      if [[ $rc -eq 0 ]]; then
        rc=1
      fi
    fi

      if [[ $rc -ne 0 ]]; then
        local exhaustion_reason=""
        if (( timeout_rc )); then
          :
        elif (( log_has_fatal_rapid_failures )); then
          :
        elif (( log_has_no_runs )); then
          echo "[wandb_agent][debug] sweep_state=${sweep_state:-unknown} runs_started=${agent_runs_started} (rc=$rc)" >&2
          if [[ $agent_runs_started -gt 0 ]]; then
            exhaustion_reason="sweep_exhausted_after_runs"
          elif [[ "$sweep_state" =~ ^(FINISHED|CANCELED|CANCELLED)$ ]]; then
            exhaustion_reason="sweep_state_${sweep_state}"
          else
            echo "[wandb_agent][error] agent saw 'No runs found' before starting any runs (sweep_state=${sweep_state:-unknown}); failing" 
          fi
        elif [[ $agent_runs_started -gt 0 ]] && (( log_has_api_instability )); then
          exhaustion_reason="wandb_api_instability"
        elif [[ $agent_runs_started -gt 0 ]] && [[ "$sweep_state" =~ ^(FINISHED|CANCELED|CANCELLED)$ ]]; then
          exhaustion_reason="sweep_state_${sweep_state}"
        elif [[ $agent_runs_started -gt 0 ]]; then
          exhaustion_reason="post_run_nonzero"
        fi

        if [[ -n "$exhaustion_reason" ]]; then
          echo "[wandb_agent][warn] treating rc=$rc as sweep exhaustion (${exhaustion_reason}) after ${agent_runs_started} run(s)"
          rc=0
        fi
      fi
    # Treat exit 2 from timeout as sweep exhaustion, not failure
    if (( timeout_rc )); then
      if [[ $timeout_rc -eq 2 ]]; then
        echo "[INFO][wandb_agent] timeout wrapper returned rc=2 (No runs left); treating as success."
        return 0
      fi
      echo "[ERROR][wandb_agent] agent hit timeout/interrupt (rc=$timeout_rc); marking graceful stop and failing" >&2
      mark_graceful_stop "$s"
      exit $timeout_rc
    fi
    if [[ $rc -eq 0 ]]; then
      :
    elif [[ $rc -eq 2 ]]; then
      echo "[INFO][wandb_agent] no runs left for sweep (rc=2); treating as success."
      return 0
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
          "devices=${PHASE2_RECHECK_FORCE_DEVICES:-${PHASE2_FORCE_DEVICES:-}}" \
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

  if [[ "$stage" == "phase2_recheck" && -n "$rerun_reason" ]]; then
    local recheck_dir
    recheck_dir="$(stage_dir phase2_recheck)"
    rm -f "${recheck_dir}/recheck_done.ok" "${recheck_dir}/recheck_incomplete.ok" 2>/dev/null || true
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

  if ! ensure_micromamba; then
    if [[ "${MJEPACI_FORCE_SYSTEM_PYTHON:-}" == "1" ]]; then
      echo "[${stage}] warn: micromamba unavailable; proceeding with system python fallback" >&2
    else
      echo "[${stage}] error: unable to bootstrap micromamba environment" >&2
      rm -f "$inputs_tmp" "$deps_tmp" "$outputs_tmp"
      return 1
    fi
  fi
  # Preserve sweep-assigned run names during phase2 sweeps instead of forcing
  # the stage label (phase 1 clears WANDB_NAME similarly before launching).
  if [[ "$stage" == "phase2_sweep" ]]; then
    unset WANDB_NAME
  else
    : "${WANDB_NAME:=$stage}"; export WANDB_NAME
  fi
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
