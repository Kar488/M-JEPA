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
SSH_OPTS=(-i "$key_path" -p "$VAST_PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
RSYNC=(rsync -avz --chmod=ugo=rwX -e "ssh ${SSH_OPTS[*]}")

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
  if ! "${RSYNC[@]}" "$REMOTE:${remote_path}/" "$local_path/" >/dev/null 2>&1; then
    echo "[collect][warn] ${label}: rsync failed for $remote_path" >&2
    return 0
  fi
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
          if ! "${RSYNC[@]}" "$REMOTE:${remote_dir}/${rel}" "$dest_path" >/dev/null 2>&1; then
            echo "[collect][warn] failed to copy ${remote_dir}/${rel}" >&2
          fi
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
  if ! "${RSYNC[@]}" "$REMOTE:${remote_path}" "$local_path" >/dev/null 2>&1; then
    echo "[collect][warn] ${label}: rsync failed for $remote_path" >&2
  fi
}

# Stage outputs (tox21_*.json) live under stage-outputs/
collect_dir "${remote_tox21}/stage-outputs" "$dest_root/stage-outputs" "stage-outputs"

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

