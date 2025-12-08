#!/usr/bin/env bash
set -euo pipefail

: "${SSH_KEY:?SSH_KEY secret required}"
: "${VAST_USER:?VAST_USER required}"
: "${VAST_HOST:?VAST_HOST required}"
: "${VAST_PORT:?VAST_PORT required}"
: "${EXPERIMENTS_ROOT:?EXPERIMENTS_ROOT required}"

stage_label="${1:-baseline}"
dest_root_input="${2:-${RUNNER_TEMP:-/tmp}/tox21_artifacts}"
if [[ -z "${dest_root_input}" ]]; then
  dest_root_input="${RUNNER_TEMP:-/tmp}/tox21_artifacts"
fi

dest_root="${dest_root_input%/}"
if [[ -n "${stage_label}" ]]; then
  dest_root="${dest_root}/${stage_label}"
fi
mkdir -p "$dest_root" ~/.ssh

key_path=~/.ssh/vast_key
trap 'rm -f "$key_path"' EXIT
if [[ "$SSH_KEY" != *$'\n' ]]; then
  printf '%s\n' "$SSH_KEY" >"$key_path"
else
  printf '%s' "$SSH_KEY" >"$key_path"
fi
chmod 600 "$key_path"

REMOTE="${VAST_USER}@${VAST_HOST}"
SSH_OPTS=(-i "$key_path" -p "$VAST_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
copy_with_rsync=0
rsync_path=""
if command -v rsync >/dev/null 2>&1; then
  rsync_path="$(command -v rsync)"
  if [[ -x "$rsync_path" ]] && rsync --version >/dev/null 2>&1; then
    copy_with_rsync=1
  else
    echo "[collect][warn] rsync present at ${rsync_path:-unknown} but unusable; falling back" >&2
  fi
else
  echo "[collect][warn] rsync unavailable; will use scp/ssh fallbacks" >&2
fi
RSYNC=(rsync -avz --chmod=ugo=rwX -e "ssh ${SSH_OPTS[*]}")
SCP=(scp -p -i "$key_path" -P "$VAST_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)

check_remote_reachable() {
  if ssh "${SSH_OPTS[@]}" "$REMOTE" "echo ok" >/dev/null 2>&1; then
    return 0
  fi

  echo "[collect][fatal] unable to reach ${REMOTE} via SSH on port ${VAST_PORT}; tox21 collector cannot proceed" >&2
  echo "[collect][hint] verify VAST_HOST/VAST_PORT and that the runner can reach the Vast machine" >&2
  exit 1
}

check_remote_reachable

remote_root="${EXP_ROOT:-}"
if [[ -z "$remote_root" ]]; then
  if [[ -n "${EXP_ID:-}" ]]; then
    remote_root="${EXPERIMENTS_ROOT%/}/${EXP_ID}"
  elif [[ -n "${PRETRAIN_EXP_ID:-}" ]]; then
    remote_root="${EXPERIMENTS_ROOT%/}/${PRETRAIN_EXP_ID}"
  else
    echo "[collect][fatal] unable to determine remote experiment root" >&2
    exit 2
  fi
fi
remote_root="${remote_root%/}"

remote_tox21="${TOX21_DIR:-}" 
if [[ -z "$remote_tox21" ]]; then
  remote_tox21="${remote_root}/tox21"
fi
remote_gate="${remote_root}/tox21_gate.env"
remote_logs="${LOG_DIR:-}" 
if [[ -z "$remote_logs" ]]; then
  remote_logs="${APP_DIR:-/srv/mjepa}/logs"
fi
remote_logs="${remote_logs%/}"

collect_dir() {
  local remote_path="$1"
  local local_path="$2"
  local label="$3"
  if [[ -z "$remote_path" ]]; then
    echo "[collect][warn] ${label}: remote path unset" >&2
    return 0
  fi
  if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d '$remote_path'" >/dev/null 2>&1; then
    echo "[collect][warn] ${label}: remote directory missing: $remote_path" >&2
    return 0
  fi
  mkdir -p "$local_path"
  if (( copy_with_rsync )); then
    if "${RSYNC[@]}" "$REMOTE:${remote_path%/}/" "$local_path/" >/dev/null 2>&1; then
      return 0
    fi
    echo "[collect][warn] ${label}: rsync failed for $remote_path; attempting scp" >&2
  else
    echo "[collect][warn] ${label}: rsync unavailable; attempting scp/ssh" >&2
  fi

  if command -v scp >/dev/null 2>&1; then
    if "${SCP[@]}" -r "$REMOTE:${remote_path%/}/." "$local_path/" >/dev/null 2>&1; then
      return 0
    fi
    echo "[collect][warn] ${label}: scp failed for $remote_path; attempting tar stream" >&2
  fi

  local parent="${remote_path%/*}" basename="${remote_path##*/}"
  [[ -z "$parent" || "$parent" == "$remote_path" ]] && parent="/"
  if ssh "${SSH_OPTS[@]}" "$REMOTE" "cd '$parent' && tar -cf - '$basename'" | tar -xf - -C "$local_path"; then
    return 0
  fi

  echo "[collect][warn] ${label}: all copy strategies failed for $remote_path" >&2
  return 0
}

collect_files_matching() {
  local remote_dir="$1"
  local local_dir="$2"
  shift 2
  local patterns=("$@")
  if [[ -z "$remote_dir" ]]; then
    return 0
  fi
  if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d '$remote_dir'" >/dev/null 2>&1; then
    return 0
  fi
  mkdir -p "$local_dir"
  local pattern
  for pattern in "${patterns[@]}"; do
    if [[ -z "$pattern" ]]; then
      continue
    fi
    local found=""
    if found=$(ssh "${SSH_OPTS[@]}" "$REMOTE" "cd '$remote_dir' && find . -maxdepth 1 -type f -name '$pattern' -printf '%P\n'" 2>/dev/null); then
      if [[ -n "$found" ]]; then
        local rel
        while IFS= read -r rel; do
          [[ -z "$rel" ]] && continue
          local dest_path="${local_dir}/${rel}"
          mkdir -p "$(dirname "$dest_path")"
          copy_file "${remote_dir}/${rel}" "$dest_path" "${remote_dir}/${rel}"
        done <<<"$found"
      fi
    fi
  done
}

copy_file() {
  local remote_path="$1"
  local local_path="$2"
  local label="$3"
  if [[ -z "$remote_path" ]]; then
    return 0
  fi
  if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "test -f '$remote_path'" >/dev/null 2>&1; then
    echo "[collect][warn] ${label}: remote file missing: $remote_path" >&2
    return 0
  fi
  mkdir -p "$(dirname "$local_path")"
  if (( copy_with_rsync )); then
    if "${RSYNC[@]}" "$REMOTE:${remote_path}" "$local_path" >/dev/null 2>&1; then
      return 0
    fi
    echo "[collect][warn] ${label}: rsync failed for $remote_path; attempting scp" >&2
  fi

  if command -v scp >/dev/null 2>&1; then
    if "${SCP[@]}" "$REMOTE:${remote_path}" "$local_path" >/dev/null 2>&1; then
      return 0
    fi
    echo "[collect][warn] ${label}: scp failed for $remote_path; attempting ssh stream" >&2
  fi

  if ssh "${SSH_OPTS[@]}" "$REMOTE" "cat '$remote_path'" >"$local_path"; then
    local mtime
    mtime="$(ssh "${SSH_OPTS[@]}" "$REMOTE" "stat -c '%y' '$remote_path'" 2>/dev/null || true)"
    if [[ -n "$mtime" ]]; then
      touch -d "$mtime" "$local_path" 2>/dev/null || true
    fi
    echo "[collect] copied ${label} from $remote_path via ssh stream" >&2
    return 0
  fi

  echo "[collect][warn] ${label}: all copy strategies failed for $remote_path" >&2
}

# Stage outputs (tox21_*.json) live under stage-outputs/
collect_dir "${remote_tox21}/stage-outputs" "$dest_root/stage-outputs" "stage-outputs"

# Integrated Gradients explanations live under tox21_dir/ig_explanations/<task>/.
collect_dir "${remote_tox21}/ig_explanations" "$dest_root/ig_explanations" "ig-explanations"

# Per-task CSV/JSON manifests and summary files at the root of tox21_dir.
collect_files_matching "$remote_tox21" "$dest_root" '*.json' '*.csv' '*.tsv'

# Copy logs directory within tox21_dir if present (e.g., calibrator or sweep traces).
collect_dir "${remote_tox21}/logs" "$dest_root/logs" "tox21-logs"

# Copy the tox21 gate env file for downstream steps.
copy_file "$remote_gate" "$dest_root/tox21_gate.env" "tox21-gate"

# Copy the aggregated tox21 log from the global LOG_DIR.
copy_file "${remote_logs}/tox21.log" "$dest_root/logs/tox21.log" "tox21-log"

# Emit summary for debugging.
if command -v find >/dev/null 2>&1; then
  echo "[collect] contents of $dest_root:" >&2
  find "$dest_root" -maxdepth 2 -mindepth 1 -print >&2 || true
fi

