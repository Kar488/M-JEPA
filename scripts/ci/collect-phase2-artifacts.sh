#!/usr/bin/env bash
set -euo pipefail

: "${SSH_KEY:?SSH_KEY secret required}"
: "${VAST_USER:?VAST_USER required}"
: "${VAST_HOST:?VAST_HOST required}"
: "${VAST_PORT:?VAST_PORT required}"
: "${EXP_ID:?EXP_ID required}"
: "${EXPERIMENTS_ROOT:?EXPERIMENTS_ROOT required}"

DEST_ROOT="${1:-${RUNNER_TEMP:-phase2_artifacts}/phase2}"
mkdir -p "$DEST_ROOT" ~/.ssh

KEY_PATH=~/.ssh/vast_key
trap 'rm -f "$KEY_PATH"' EXIT
printf '%s' "$SSH_KEY" >"$KEY_PATH"
chmod 600 "$KEY_PATH"

REMOTE="${VAST_USER}@${VAST_HOST}"
RSYNC=(rsync -avz --chmod=ugo=rwX -e "ssh -i $KEY_PATH -p $VAST_PORT -o StrictHostKeyChecking=no")

remote_root="${EXPERIMENTS_ROOT%/}/${EXP_ID}"
remote_grid="${remote_root}/grid"

steps=(phase2_sweep phase2_recheck phase2_export)
for step in "${steps[@]}"; do
  local_dir="${DEST_ROOT}/${step}"
  mkdir -p "$local_dir"
  remote_dir="${remote_grid}/${step}"

  if ssh -i "$KEY_PATH" -p "$VAST_PORT" -o StrictHostKeyChecking=no "$REMOTE" "test -d '$remote_dir'"; then
    "${RSYNC[@]}" "$REMOTE:$remote_dir/logs/" "$local_dir/logs" 2>/dev/null || true
    "${RSYNC[@]}" "$REMOTE:$remote_dir/stage-outputs/" "$local_dir/stage-outputs" 2>/dev/null || true
  fi
done

mkdir -p "${DEST_ROOT}/grid"
for name in best_grid_config.json recheck_summary.json; do
  "${RSYNC[@]}" "$REMOTE:${remote_grid}/${name}" "${DEST_ROOT}/grid/" 2>/dev/null || true
 done

helper_output=""
if helper_output=$(ssh -i "$KEY_PATH" -p "$VAST_PORT" -o StrictHostKeyChecking=no "$REMOTE" \
  "cd '${remote_grid}' 2>/dev/null && find . -maxdepth 1 -type f \\\\( -name 'phase2_winner*' -o -name 'winner_*' -o -name 'phase2_cli*' \\\\) -printf '%P\\n'" 2>/dev/null); then
  mapfile -t helper_files <<<"$helper_output"
else
  helper_files=()
fi

if (( ${#helper_files[@]} > 0 )); then
  mkdir -p "${DEST_ROOT}/grid/helpers"
  for rel in "${helper_files[@]}"; do
    [[ -z "$rel" ]] && continue
    "${RSYNC[@]}" "$REMOTE:${remote_grid}/${rel}" "${DEST_ROOT}/grid/helpers/" 2>/dev/null || true
  done
fi

