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

remote_lineage_grid="${EXPERIMENTS_ROOT%/}/${PRETRAIN_EXP_ID}/grid"
remote_current_id="${GRID_EXP_ID:-${EXP_ID}}"
remote_current_grid="${EXPERIMENTS_ROOT%/}/${remote_current_id}/grid"

collect_tree() {
  local remote_grid="$1"
  local dest_root="$2"
  local label="$3"
  shift 3 || true
  local -a steps=(phase2_sweep phase2_recheck phase2_export)
  for step in "${steps[@]}"; do
    local local_dir="${dest_root}/${step}"
    mkdir -p "$local_dir"
    local remote_dir="${remote_grid}/${step}"
    if ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d '$remote_dir'"; then
      if ! "${RSYNC[@]}" "$REMOTE:$remote_dir/logs/" "$local_dir/logs" 2>/dev/null; then
        echo "[ci][warn] missing or empty logs for $label/$step at $remote_dir/logs" >&2
      fi
      if ! "${RSYNC[@]}" "$REMOTE:$remote_dir/stage-outputs/" "$local_dir/stage-outputs" 2>/dev/null; then
        echo "[ci][warn] missing stage outputs for $label/$step at $remote_dir/stage-outputs" >&2
      fi
    else
      echo "[ci][warn] remote step directory not found for $label: $remote_dir" >&2
    fi
  done

  mkdir -p "${dest_root}/grid"
  for name in best_grid_config.json recheck_summary.json grid_state.json; do
    if ! "${RSYNC[@]}" "$REMOTE:${remote_grid}/${name}" "${dest_root}/grid/" 2>/dev/null; then
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

