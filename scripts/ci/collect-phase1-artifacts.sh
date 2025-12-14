#!/usr/bin/env bash
set -euo pipefail

: "${SSH_KEY:?SSH_KEY secret required}"
: "${VAST_USER:?VAST_USER required}"
: "${VAST_HOST:?VAST_HOST required}"
: "${VAST_PORT:?VAST_PORT required}"
: "${EXP_ID:?EXP_ID required}"
: "${EXPERIMENTS_ROOT:?EXPERIMENTS_ROOT required}"

DEST_ROOT="${1:-${RUNNER_TEMP:-phase1_artifacts}/phase1}"
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
SSH_OPTS=(-i "$KEY_PATH" -p "$VAST_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
copy_with_rsync=0
rsync_path=""
COPIED_ANY=0
if command -v rsync >/dev/null 2>&1; then
  rsync_path="$(command -v rsync)"
  if [[ -x "$rsync_path" ]] && rsync --version >/dev/null 2>&1; then
    copy_with_rsync=1
  else
    echo "[ci][warn] rsync present at ${rsync_path:-unknown} but unusable; falling back" >&2
  fi
else
  echo "[ci][warn] rsync unavailable; will use scp/ssh fallbacks" >&2
fi
RSYNC=(rsync -avz --chmod=ugo=rwX -e "ssh ${SSH_OPTS[*]}")
SCP=(scp -p -i "$KEY_PATH" -P "$VAST_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)

choose_local_python() {
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "python3"
  elif command -v python >/dev/null 2>&1; then
    printf '%s' "python"
  else
    printf '%s' ""
  fi
}

PY_LOCAL="$(choose_local_python)"
if [[ -z "$PY_LOCAL" ]]; then
  echo "[ci][warn] no local python interpreter available; lineage resolution helpers will be skipped" >&2
fi

check_remote_reachable() {
  if ssh "${SSH_OPTS[@]}" "$REMOTE" "echo ok" >/dev/null 2>&1; then
    return 0
  fi

  echo "[ci][fatal] unable to reach ${REMOTE} via SSH on port ${VAST_PORT}; phase1 collector cannot proceed" >&2
  echo "[ci][hint] verify VAST_HOST/VAST_PORT and that the runner can reach the Vast machine" >&2
  exit 1
}

check_remote_reachable

sync_remote_dir() {
  local remote_dir="$1"
  local local_dir="$2"
  local label="$3"

  if [[ -z "$remote_dir" ]]; then
    return 0
  fi
  if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d '${remote_dir}'" >/dev/null 2>&1; then
    echo "[ci][warn] ${label}: remote directory missing: ${remote_dir}" >&2
    return 0
  fi

  mkdir -p "$local_dir"

  if (( copy_with_rsync )); then
    if "${RSYNC[@]}" "$REMOTE:${remote_dir%/}/" "$local_dir/" >/dev/null 2>&1; then
      COPIED_ANY=1
      return 0
    fi
    echo "[ci][warn] ${label}: rsync failed for ${remote_dir}; attempting scp" >&2
  else
    echo "[ci][warn] ${label}: rsync unavailable; attempting scp/ssh" >&2
  fi

  if command -v scp >/dev/null 2>&1; then
    if "${SCP[@]}" -r "$REMOTE:${remote_dir%/}/." "$local_dir/" >/dev/null 2>&1; then
      COPIED_ANY=1
      return 0
    fi
    echo "[ci][warn] ${label}: scp failed for ${remote_dir}; attempting tar stream" >&2
  fi

  local parent="${remote_dir%/*}" basename="${remote_dir##*/}"
  [[ -z "$parent" || "$parent" == "$remote_dir" ]] && parent="/"
  if ssh "${SSH_OPTS[@]}" "$REMOTE" "cd '${parent}' && tar -cf - '${basename}'" | tar -xf - -C "$local_dir" --strip-components=1; then
    COPIED_ANY=1
    return 0
  fi

  echo "[ci][warn] ${label}: all copy strategies failed for ${remote_dir}" >&2
  return 0
}

sync_remote_file() {
  local remote_file="$1"
  local local_dir="$2"
  local label="$3"

  if [[ -z "$remote_file" ]]; then
    return 0
  fi
  if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "test -f '${remote_file}'" >/dev/null 2>&1; then
    echo "[ci][warn] ${label}: remote file missing: ${remote_file}" >&2
    return 0
  fi

  mkdir -p "$local_dir"

  if (( copy_with_rsync )); then
    if "${RSYNC[@]}" "$REMOTE:${remote_file}" "$local_dir/" >/dev/null 2>&1; then
      COPIED_ANY=1
      return 0
    fi
    echo "[ci][warn] ${label}: rsync failed for ${remote_file}; attempting scp" >&2
  fi

  if command -v scp >/dev/null 2>&1; then
    if "${SCP[@]}" "$REMOTE:${remote_file}" "$local_dir/" >/dev/null 2>&1; then
      COPIED_ANY=1
      return 0
    fi
    echo "[ci][warn] ${label}: scp failed for ${remote_file}; attempting ssh stream" >&2
  fi

  local dest_path="${local_dir}/$(basename "$remote_file")"
  if ssh "${SSH_OPTS[@]}" "$REMOTE" "cat '${remote_file}'" >"$dest_path"; then
    local mtime
    mtime="$(ssh "${SSH_OPTS[@]}" "$REMOTE" "stat -c '%y' '${remote_file}'" 2>/dev/null || true)"
    if [[ -n "$mtime" ]]; then
      touch -d "$mtime" "$dest_path" 2>/dev/null || true
    fi
    echo "[ci] copied ${label} via ssh stream from ${remote_file}" >&2
    COPIED_ANY=1
    return 0
  fi

  echo "[ci][warn] ${label}: all copy strategies failed for ${remote_file}" >&2
  return 0
}

discover_remote_phase1_lineage() {
  local app_dir="${APP_DIR:-/srv/mjepa}" default_id="${GRID_EXP_ID:-${EXP_ID}}"
  local payload=""
  if ! payload="$(
    ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$app_dir" "$EXPERIMENTS_ROOT" "$default_id" <<'EOS' 2>/dev/null
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

  if [[ -n "$payload" && -n "$PY_LOCAL" ]]; then
    local -a resolved=()
    if mapfile -t resolved < <("$PY_LOCAL" - "$payload" <<'PY'
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

print(emit("grid_dir"))
print(emit("grid_exp_id"))
PY
    ); then
      local resolved_grid="${resolved[0]:-}" resolved_id="${resolved[1]:-}"
      if [[ -n "$resolved_grid" ]]; then
        GRID_DIR="$resolved_grid"
        export GRID_DIR
      fi
      if [[ -n "$resolved_id" ]]; then
        GRID_EXP_ID="$resolved_id"
        export GRID_EXP_ID
      fi
    fi
  fi

  if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
      [[ -n "$GRID_DIR" ]] && echo "GRID_DIR=$GRID_DIR"
      [[ -n "$GRID_EXP_ID" ]] && echo "GRID_EXP_ID=$GRID_EXP_ID"
    } >>"$GITHUB_ENV"
  fi
}

remote_lineage_id="${GRID_EXP_ID:-${EXP_ID}}"
remote_current_id="${EXP_ID:-${GRID_EXP_ID:-}}"

if [[ -z "${GRID_DIR:-}" ]]; then
  discover_remote_phase1_lineage
fi

remote_lineage_grid="${GRID_DIR:-${EXPERIMENTS_ROOT%/}/${remote_lineage_id}/grid}"
remote_current_grid="${EXPERIMENTS_ROOT%/}/${remote_current_id}/grid"

resolve_remote_grid_root() {
  local primary="$1" grid_id="$2"
  local -a candidates=()
  local -a markers=(
    "phase1_export"
    "phase1"
    "phase1_export/stage-outputs"
    "phase1_export/stage-outputs/phase1_runs.csv"
    "grid_sweep_phase2.yaml"
    "phase1_sweep_id.txt"
  )
  local experiments_root="${EXPERIMENTS_ROOT:-}"
  local inferred_cache_root=""
  if [[ -n "$experiments_root" ]]; then
    experiments_root="${experiments_root%/}"
    inferred_cache_root="${experiments_root%/}/../cache"
  fi

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

  add_candidate_with_grid_variant() {
    local base="$1"
    [[ -z "$base" ]] && return
    add_candidate "$base"
    add_candidate "${base%/}/grid"
  }

  add_candidate_with_grid_variant "$primary"
  add_candidate_with_grid_variant "${GRID_DIR:-}"
  add_candidate_with_grid_variant "${SWEEP_CACHE_DIR:-}/grid"
  add_candidate_with_grid_variant "${SWEEP_CACHE_DIR:-}/grid/${grid_id}"
  add_candidate_with_grid_variant "${GRID_CACHE_DIR:-}"
  add_candidate_with_grid_variant "${GRID_CACHE_DIR:-}/${grid_id}"
  add_candidate_with_grid_variant "${CACHE_DIR:-}/grid"
  add_candidate_with_grid_variant "${CACHE_DIR:-}/grid/${grid_id}"
  add_candidate_with_grid_variant "${inferred_cache_root%/}/grid"
  add_candidate_with_grid_variant "${inferred_cache_root%/}/grid/${grid_id}"
  if [[ -n "${RUNNER_TEMP:-}" && -n "$grid_id" ]]; then
    add_candidate_with_grid_variant "${RUNNER_TEMP%/}/mjepa/fallback/grid/${grid_id}"
  fi

  local first_existing=""
  local candidate
  for candidate in "${candidates[@]}"; do
    if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d '${candidate}'" >/dev/null 2>&1; then
      continue
    fi

    if [[ -z "$first_existing" ]]; then
      first_existing="$candidate"
    fi

    local marker
    for marker in "${markers[@]}"; do
      if ssh "${SSH_OPTS[@]}" "$REMOTE" "test -e '${candidate%/}/${marker}'" >/dev/null 2>&1; then
        if [[ "$candidate" != "$primary" ]]; then
          echo "[ci][warn] using fallback grid root for ${grid_id:-unknown}: ${candidate}" >&2
        fi
        printf '%s' "$candidate"
        return 0
      fi
    done
  done

  if [[ -n "$first_existing" ]]; then
    if [[ "$first_existing" != "$primary" ]]; then
      echo "[ci][warn] using fallback grid root for ${grid_id:-unknown}: ${first_existing}" >&2
    fi
    printf '%s' "$first_existing"
    return 0
  fi

  if [[ -n "$grid_id" ]]; then
    local probe=""
    if ! probe=$(ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$grid_id" "${EXPERIMENTS_ROOT:-}" "${GRID_CACHE_DIR:-}" "${SWEEP_CACHE_DIR:-}" "${CACHE_DIR:-}" 2>/dev/null <<'EOS'
set -euo pipefail
gid="$1"
shift
roots=("$@")
for root in "${roots[@]}"; do
  [[ -d "$root" ]] || continue
  if path=$(find "$root" -maxdepth 6 \( -type d -path "*/${gid}/grid" -o -type d -path "*/${gid}/*/grid" -o -type f -path "*/${gid}/phase1_export/stage-outputs/phase1_runs.csv" -o -type d -path "*/${gid}/phase1_export" \) -print -quit 2>/dev/null); then
    if [[ -f "$path" ]]; then
      path="$(dirname "${path%/*}")"
    fi
    printf '%s' "${path%/}" | sed 's#/phase1_export/stage-outputs##; s#/phase1_export##'
    exit 0
  fi
done
EOS
    ); then
      probe=""
    fi

    if [[ -n "$probe" ]]; then
      echo "[ci][warn] using discovered grid root for ${grid_id}: ${probe}" >&2
      printf '%s' "$probe"
      return 0
    fi
  fi

  printf '%s' "$primary"
}

resolve_and_set_grid() {
  local label="$1" grid_path="$2" grid_id="$3"
  local resolved
  resolved="$(resolve_remote_grid_root "$grid_path" "$grid_id")"
  if [[ -n "$resolved" && "$resolved" != "$grid_path" ]]; then
    echo "[ci][warn] ${label}: resolved grid path to ${resolved}" >&2
  fi
  printf '%s' "${resolved:-$grid_path}"
}

remote_lineage_grid="$(resolve_and_set_grid lineage "$remote_lineage_grid" "$remote_lineage_id")"
remote_current_grid="$(resolve_and_set_grid current "$remote_current_grid" "$remote_current_id")"
if [[ -z "$remote_lineage_id" && -n "$remote_lineage_grid" ]]; then
  remote_lineage_id="$(basename "$(dirname "${remote_lineage_grid%/}")")"
fi
if [[ -z "$remote_current_id" && -n "$remote_current_grid" ]]; then
  remote_current_id="$(basename "$(dirname "${remote_current_grid%/}")")"
fi

collect_tree() {
  local remote_grid="$1" local_root="$2" label="$3" grid_id="$4"
  mkdir -p "$local_root"
  if [[ -z "$remote_grid" ]]; then
    echo "[ci][warn] ${label}: grid path is empty" >&2
    return 0
  fi

  sync_remote_dir "${remote_grid%/}/phase1_export" "$local_root/phase1_export" "${label} phase1_export"
  sync_remote_dir "${remote_grid%/}/phase1" "$local_root/phase1" "${label} phase1"

  local stage_outputs="${remote_grid%/}/phase1_export/stage-outputs"
  sync_remote_dir "$stage_outputs" "$local_root/phase1_export/stage-outputs" "${label} stage-outputs"

  local winner_candidates=(
    "${remote_grid%/}/phase1_export/stage-outputs/phase2_winner_config.csv"
    "${remote_grid%/}/phase2_winner_config.csv"
    "${remote_grid%/}/phase2_export/phase2_winner_config.csv"
  )
  local candidate
  for candidate in "${winner_candidates[@]}"; do
    sync_remote_file "$candidate" "$local_root/phase1_export" "${label} winner"
  done

  sync_remote_file "${remote_grid%/}/phase1_export/stage-outputs/phase1_runs.csv" "$local_root/phase1_export/stage-outputs" "${label} phase1_runs"

  local helper_output=""
  if helper_output=$(ssh "${SSH_OPTS[@]}" "$REMOTE" "cd '${remote_grid%/}' 2>/dev/null && find . -maxdepth 1 -type f \\( -name 'phase1_winner*' -o -name 'winner_*' \\) -print" 2>/dev/null); then
    local rel
    while IFS= read -r rel; do
      [[ -z "$rel" ]] && continue
      rel="${rel#./}"
      sync_remote_file "${remote_grid%/}/${rel}" "$local_root/phase1_export/helpers" "${label} helper ${rel}"
    done <<<"$helper_output"
  fi
}

collect_logs() {
  local local_root="$1"
  local log_root="${LOG_DIR:-${APP_DIR:-/srv/mjepa}/logs}"
  mkdir -p "$local_root"
  local name
  for name in phase1 phase1_jepa phase1_contrastive; do
    sync_remote_dir "${log_root%/}/${name}" "$local_root/${name}" "logs/${name}"
  done
}

collect_tree "$remote_lineage_grid" "${DEST_ROOT}/lineage" "lineage" "${remote_lineage_id}" || true
collect_tree "$remote_current_grid" "${DEST_ROOT}/current" "current" "${remote_current_id}" || true
collect_logs "${DEST_ROOT}/logs"

mkdir -p "${DEST_ROOT}/logs"

if [[ -d "${DEST_ROOT}/current/phase1_export" ]]; then
  mkdir -p "${DEST_ROOT}/phase1_export"
  rsync -a "${DEST_ROOT}/current/phase1_export/" "${DEST_ROOT}/phase1_export/" >/dev/null 2>&1 || true
fi

if (( COPIED_ANY == 0 )); then
  echo "[ci][fatal] no phase1 artifacts were copied from ${REMOTE} (lineage grid: ${remote_lineage_grid}, current grid: ${remote_current_grid})." >&2
  echo "[ci][hint] verify EXP_ID/GRID_EXP_ID and that phase1 artifacts exist under ${EXPERIMENTS_ROOT}" >&2
  exit 1
fi

echo "[ci] phase1 artifacts collected into ${DEST_ROOT}" >&2
