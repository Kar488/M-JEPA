#!/usr/bin/env bash
set -euo pipefail

: "${SSH_KEY:?SSH_KEY secret required}"
: "${VAST_USER:?VAST_USER required}"
: "${VAST_HOST:?VAST_HOST required}"
: "${VAST_PORT:?VAST_PORT required}"
: "${EXP_ID:?EXP_ID required}"
: "${EXPERIMENTS_ROOT:?EXPERIMENTS_ROOT required}"
: "${PRETRAIN_EXP_ID:?PRETRAIN_EXP_ID required}"

DEST_ROOT="${1:-${RUNNER_TEMP:-phase2_artifacts}/phase2}"
mkdir -p "$DEST_ROOT" ~/.ssh

KEY_PATH=~/.ssh/vast_key
trap 'rm -f "$KEY_PATH"' EXIT
if [[ "$SSH_KEY" != *$'\n' ]]; then
  printf '%s\n' "$SSH_KEY" >"$KEY_PATH"
else
  printf '%s' "$SSH_KEY" >"$KEY_PATH"
fi
chmod 600 "$KEY_PATH"

if command -v ssh-agent >/dev/null 2>&1; then
  if [[ -z "${SSH_AUTH_SOCK:-}" || ! -S "${SSH_AUTH_SOCK}" ]]; then
    eval "$(ssh-agent -s)" >/dev/null 2>&1 || true
  fi
fi

if command -v ssh-add >/dev/null 2>&1; then
  if ! ssh-add -L 2>/dev/null | grep -F "$KEY_PATH" >/dev/null 2>&1; then
    SSH_ASKPASS="${SSH_ASKPASS:-/bin/true}" DISPLAY="${DISPLAY:-}" ssh-add "$KEY_PATH" >/dev/null 2>&1 || true
  fi
fi

REMOTE="${VAST_USER}@${VAST_HOST}"
SSH_OPTS=(-i "$KEY_PATH" -p "$VAST_PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
RSYNC=(rsync -avz --chmod=ugo=rwX -e "ssh ${SSH_OPTS[*]}")

discover_remote_phase2_lineage() {
  local target_dir="${GRID_DIR:-}" target_id="${GRID_EXP_ID:-}" need_lookup=0
  if [[ -z "$target_dir" ]]; then
    need_lookup=1
  else
    if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "test -f '${target_dir%/}/phase2_sweep_id.txt'" >/dev/null 2>&1; then
      need_lookup=1
    fi
  fi
  if [[ -z "$target_id" || "$target_id" == "${EXP_ID}" ]]; then
    need_lookup=1
  fi

  local app_dir="${APP_DIR:-/srv/mjepa}"
  local default_id="${target_id:-${PRETRAIN_EXP_ID:-${EXP_ID}}}"
  local payload=""

  if (( need_lookup )); then
    if ! payload="$(ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$app_dir" "$EXPERIMENTS_ROOT" "$default_id" <<'EOS' 2>/dev/null
set -euo pipefail
app_dir="${1:-/srv/mjepa}"
exp_root="${2:-/data/mjepa/experiments}"
default_id="${3:-}"
if [[ ! -d "$app_dir" ]]; then
  exit 0
fi
cd "$app_dir"
if command -v python3 >/dev/null 2>&1; then
  py=python3
elif command -v python >/dev/null 2>&1; then
  py=python
else
  exit 0
fi
"$py" scripts/ci/resolve_lineage_ids.py --root "$exp_root" --default-id "$default_id"
EOS
    )"; then
      payload=""
    fi

    if [[ -n "$payload" ]]; then
      local py_local=""
      if command -v python3 >/dev/null 2>&1; then
        py_local=python3
      elif command -v python >/dev/null 2>&1; then
        py_local=python
      fi

      if [[ -n "$py_local" ]]; then
        local -a resolved=()
        if mapfile -t resolved < <("$py_local" - "$payload" <<'PY'
import json
import sys

try:
    payload = json.loads(sys.argv[1])
except Exception:
    payload = {}

def emit(key):
    value = payload.get(key)
    if isinstance(value, str):
        return value.strip()
    return ""

print(emit("grid_exp_id"))
print(emit("grid_dir"))
PY
        ); then
          local remote_grid_id="${resolved[0]:-}" remote_grid_dir="${resolved[1]:-}"

          if [[ -n "$remote_grid_dir" ]]; then
            GRID_DIR="$remote_grid_dir"
            export GRID_DIR
            if [[ -z "${GRID_SOURCE_DIR:-}" ]]; then
              GRID_SOURCE_DIR="$remote_grid_dir"
              export GRID_SOURCE_DIR
            fi
          fi
          if [[ -n "$remote_grid_id" ]]; then
            GRID_EXP_ID="$remote_grid_id"
            export GRID_EXP_ID
          fi
        fi
      fi
    fi
  fi

  local need_dir_resolve=0
  if [[ -z "${GRID_DIR:-}" ]]; then
    need_dir_resolve=1
  elif ! ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d '${GRID_DIR%/}'" >/dev/null 2>&1; then
    need_dir_resolve=1
  fi

  if (( need_dir_resolve )); then
    local exp_root="${EXPERIMENTS_ROOT%/}"
    local probe=""
    probe=$(ssh "${SSH_OPTS[@]}" "$REMOTE" "find '$exp_root' -maxdepth 3 -path '*/grid/phase2_sweep_id.txt' -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1" 2>/dev/null || true)
    if [[ -n "$probe" ]]; then
      GRID_DIR="${probe#* }"
      export GRID_DIR
      GRID_EXP_ID="$(basename "$(dirname "${GRID_DIR%/}")")"
      export GRID_EXP_ID
      if [[ -z "${GRID_SOURCE_DIR:-}" ]]; then
        GRID_SOURCE_DIR="$GRID_DIR"
        export GRID_SOURCE_DIR
      fi
      echo "[ci][warn] fallback discovered Phase-2 grid at ${GRID_DIR} (GRID_EXP_ID=${GRID_EXP_ID:-<unset>})" >&2
    else
      echo "[ci][warn] unable to locate Phase-2 sweep directory under ${EXPERIMENTS_ROOT}" >&2
    fi
  fi

  if [[ -n "$GRID_DIR" || -n "$GRID_EXP_ID" ]]; then
    echo "[ci] discovered remote Phase-2 lineage GRID_EXP_ID=${GRID_EXP_ID:-<unset>} GRID_DIR=${GRID_DIR:-<unset>}" >&2
  fi

  if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
      [[ -n "$GRID_EXP_ID" ]] && echo "GRID_EXP_ID=$GRID_EXP_ID"
      [[ -n "$GRID_DIR" ]] && echo "GRID_DIR=$GRID_DIR"
    } >>"$GITHUB_ENV"
  fi
}

discover_remote_phase2_lineage

remote_lineage_id="${GRID_EXP_ID:-${PRETRAIN_EXP_ID:-}}"
remote_current_id="${GRID_EXP_ID:-${EXP_ID}}"

if [[ -n "${GRID_DIR:-}" ]]; then
  # GRID_DIR points directly at the grid used by phase2_export (e.g. /data/mjepa/experiments/1760284429/grid).
  remote_lineage_grid="${GRID_DIR%/}"
else
  # Fall back to constructing it from GRID_EXP_ID or PRETRAIN_EXP_ID.
  remote_lineage_grid="${EXPERIMENTS_ROOT%/}/${remote_lineage_id}/grid"
fi
remote_current_grid="${EXPERIMENTS_ROOT%/}/${remote_current_id}/grid"

resolve_remote_grid_root() {
  local primary="$1" grid_id="$2"
  shift 2 || true

  local -a candidates=()
  add_candidate() {
    local path="$1"
    [[ -z "$path" ]] && return
    local normalized="${path%/}"
    local existing
    for existing in "${candidates[@]}"; do
      if [[ "$existing" == "$normalized" ]]; then
        return
      fi
    done
    candidates+=("$normalized")
  }

  add_candidate "$primary"
  add_candidate "${GRID_SOURCE_DIR:-}"
  add_candidate "${GRID_DIR:-}"
  add_candidate "${SWEEP_CACHE_DIR:-}/grid"
  add_candidate "${SWEEP_CACHE_DIR:-}/grid/${grid_id}"
  add_candidate "${GRID_CACHE_DIR:-}"
  add_candidate "${GRID_CACHE_DIR:-}/${grid_id}"
  add_candidate "${CACHE_DIR:-}/grid"
  add_candidate "${CACHE_DIR:-}/grid/${grid_id}"
  if [[ -n "${RUNNER_TEMP:-}" && -n "$grid_id" ]]; then
    add_candidate "${RUNNER_TEMP%/}/mjepa/fallback/grid/${grid_id}"
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d '${candidate}'" >/dev/null 2>&1; then
      if [[ "$candidate" != "$primary" ]]; then
        echo "[ci][warn] using fallback grid root for ${grid_id:-unknown}: ${candidate}" >&2
      fi
      printf '%s' "$candidate"
      return 0
    fi
  done

  if [[ -n "$grid_id" ]]; then
    local probe=""
    if ! probe=$(
      ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$grid_id" "${EXPERIMENTS_ROOT:-}" "${GRID_CACHE_DIR:-}" "${SWEEP_CACHE_DIR:-}" "${CACHE_DIR:-}" 2>/dev/null <<'EOS'
set -euo pipefail
gid="$1"
shift
roots=("$@")
for root in "${roots[@]}"; do
  [[ -d "$root" ]] || continue
  if path=$(find "$root" -maxdepth 4 -type d -path "*/${gid}/grid" -print -quit 2>/dev/null); then
    printf '%s' "$path"
    exit 0
  fi
done
EOS
    ); then
      probe=""
    fi

    if [[ -n "$probe" ]]; then
      echo "[ci][warn] discovered grid root via search for ${grid_id}: ${probe}" >&2
      printf '%s' "$probe"
      return 0
    fi
  fi

  printf '%s' "$primary"
  return 1
}

remote_lineage_grid="$(resolve_remote_grid_root "$remote_lineage_grid" "${remote_lineage_id:-}")"
remote_current_grid="$(resolve_remote_grid_root "$remote_current_grid" "${remote_current_id:-}")"


# Ensure Phase‑2 sweep metadata is available under the current experiment.
# If the sweep ID and JSON files exist in the lineage grid but are missing in the
# current grid, copy them over on the Vast host.  This prevents the pretrain
# stage from failing due to a missing phase2_sweep_id.txt.
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "${remote_lineage_grid}" "${remote_current_grid}" <<'EOS'
set -euo pipefail
src_grid="$1"
dst_grid="$2"
if [[ -f "${src_grid}/phase2_sweep_id.txt" ]] && [[ ! -f "${dst_grid}/phase2_sweep_id.txt" ]]; then
  mkdir -p "${dst_grid}"
  for f in phase2_sweep_id.txt best_grid_config.json recheck_summary.json grid_state.json; do
    if [[ -f "${src_grid}/${f}" ]]; then
      cp -f "${src_grid}/${f}" "${dst_grid}/${f}"
    fi
  done
fi

if [[ "${src_grid%/}" != "${dst_grid%/}" ]]; then
  for step in phase2_sweep phase2_recheck phase2_export; do
    src_step="${src_grid}/${step}"
    dst_step="${dst_grid}/${step}"
    if [[ -d "$src_step" ]]; then
      mkdir -p "$dst_step"
      cp -a "${src_step}/." "$dst_step/" 2>/dev/null || true
    fi
  done
fi
EOS

collect_tree() {
  local remote_grid="$1"
  local dest_root="$2"
  local label="$3"
  shift 3 || true
  local -a steps=(phase2_sweep phase2_recheck phase2_export)
  find_remote_step_dir() {
    local grid_root="$1" step_name="$2"
    local primary_dir="${grid_root%/}/${step_name}"
    if ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d '$primary_dir'"; then
      printf '%s' "$primary_dir"
      return 0
    fi

    local alt_name="${step_name//_/-}" candidate=""
    if candidate=$(ssh "${SSH_OPTS[@]}" "$REMOTE" \
      "cd '${grid_root%/}' 2>/dev/null && find . -maxdepth 3 -type d \\\\( -name '${step_name}' -o -name '${alt_name}' \\\\) -print | head -n1" 2>/dev/null); then
      candidate="${candidate#./}"
    fi

    if [[ -n "$candidate" ]]; then
      printf '%s/%s' "${grid_root%/}" "$candidate"
      return 0
    fi
    return 1
  }
  for step in "${steps[@]}"; do
    local local_dir="${dest_root}/${step}"
    mkdir -p "$local_dir"
    local remote_dir
    if ! remote_dir="$(find_remote_step_dir "$remote_grid" "$step")" || [[ -z "$remote_dir" ]]; then
      echo "[ci][warn] remote step directory not found for $label: ${remote_grid}/${step}" >&2
      continue
    fi

    if ! "${RSYNC[@]}" "$REMOTE:${remote_dir%/}/logs/" "$local_dir/logs" 2>/dev/null; then
      echo "[ci][warn] missing or empty logs for $label/$step at ${remote_dir%/}/logs" >&2
    fi
    if ! "${RSYNC[@]}" "$REMOTE:${remote_dir%/}/stage-outputs/" "$local_dir/stage-outputs" 2>/dev/null; then
      echo "[ci][warn] missing stage outputs for $label/$step at ${remote_dir%/}/stage-outputs" >&2
    fi
  done

  mkdir -p "${dest_root}/grid"
  for name in best_grid_config.json recheck_summary.json grid_state.json; do
    if "${RSYNC[@]}" "$REMOTE:${remote_grid}/${name}" "${dest_root}/grid/" 2>/dev/null; then
      continue
    fi

    local fallback=""
    if fallback=$(ssh "${SSH_OPTS[@]}" "$REMOTE" \
      "cd '${remote_grid%/}' 2>/dev/null && find . -maxdepth 3 -type f -name '${name}' -print | head -n1" 2>/dev/null); then
      fallback="${fallback#./}"
    fi

    if [[ -n "$fallback" ]]; then
      "${RSYNC[@]}" "$REMOTE:${remote_grid%/}/${fallback}" "${dest_root}/grid/" 2>/dev/null || true
    else
      echo "[ci][warn] unable to copy ${name} from ${remote_grid} (${label})" >&2
    fi
  done

  local helper_output=""
  local helper_files=()
  if helper_output=$(ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "cd '${remote_grid}' 2>/dev/null && find . -maxdepth 1 -type f \\\\( -name 'phase2_winner*' -o -name 'winner_*' -o -name 'phase2_cli*' \\\\) -printf '%P\\n'" 2>/dev/null); then
    mapfile -t helper_files <<<"$helper_output"
  fi

  if (( ${#helper_files[@]} > 0 )); then
    mkdir -p "${dest_root}/grid/helpers"
    for rel in "${helper_files[@]}"; do
      [[ -z "$rel" ]] && continue
      "${RSYNC[@]}" "$REMOTE:${remote_grid}/${rel}" "${dest_root}/grid/helpers/" 2>/dev/null || true
    done
  fi
}

collect_tree "$remote_lineage_grid" "${DEST_ROOT}/lineage" "lineage"
collect_tree "$remote_current_grid" "${DEST_ROOT}/current" "current"

