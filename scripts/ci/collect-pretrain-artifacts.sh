#!/usr/bin/env bash
# Collect required pretrain artifacts from the Vast runner. The script ensures
# the remote directories exist, syncs the manifest, checkpoint, and state, and
# validates the required files locally for downstream jobs.
set -euo pipefail

: "${SSH_KEY:?SSH_KEY secret required}"
: "${VAST_USER:?VAST_USER required}"
: "${VAST_HOST:?VAST_HOST required}"
: "${VAST_PORT:?VAST_PORT required}"
if [[ -z "${PRETRAIN_EXPERIMENT_ROOT:-}" ]]; then
  echo "::warning::PRETRAIN_EXPERIMENT_ROOT not resolved; skipping rsync" >&2
  exit 0
fi

DEST_DIR="${1:-pretrain_artifacts}"
mkdir -p "$DEST_DIR" ~/.ssh

KEY_PATH=~/.ssh/vast_key
trap 'rm -f "$KEY_PATH"' EXIT
if [[ "$SSH_KEY" != *$'\n' ]]; then
  printf '%s\n' "$SSH_KEY" >"$KEY_PATH"
else
  printf '%s' "$SSH_KEY" >"$KEY_PATH"
fi
chmod 600 "$KEY_PATH"

REMOTE="${VAST_USER}@${VAST_HOST}"
SSH_CMD=(ssh -i "$KEY_PATH" -p "$VAST_PORT" -o StrictHostKeyChecking=no)
RSYNC=(rsync -avz --chmod=ugo=rwX -e "ssh -i $KEY_PATH -p $VAST_PORT -o StrictHostKeyChecking=no")

echo "[collect] PRETRAIN_EXPERIMENT_ROOT=$PRETRAIN_EXPERIMENT_ROOT" >&2

"${SSH_CMD[@]}" "$REMOTE" \
  "mkdir -p '${PRETRAIN_EXPERIMENT_ROOT}/artifacts' '${PRETRAIN_EXPERIMENT_ROOT}/pretrain/stage-outputs'"

sync_file() {
  local remote_path="$1"
  local label="$2"
  if [[ -z "$remote_path" ]]; then
    echo "::warning::skip $label because remote path is empty" >&2
    return 0
  fi
  if "${RSYNC[@]}" "$REMOTE:$remote_path" "$DEST_DIR"; then
    return 0
  fi
  echo "::warning::rsync failed for $label at $remote_path" >&2
  return 0
}

remote_file_exists() {
  local remote_path="$1"
  "${SSH_CMD[@]}" "$REMOTE" bash -s -- "$remote_path" <<'EOS' >/dev/null 2>&1
set -euo pipefail
path="${1:-}"
if [[ -f "$path" ]]; then
  exit 0
fi
exit 1
EOS
}

sync_optional_file() {
  local remote_path="$1"
  local label="$2"
  local dest_dir="$3"
  if [[ -z "$remote_path" ]]; then
    echo "[collect] info: $label remote path empty; skipping" >&2
    return 0
  fi
  if remote_file_exists "$remote_path"; then
    if "${RSYNC[@]}" "$REMOTE:$remote_path" "$dest_dir"; then
      echo "[collect] fetched optional $label from $remote_path" >&2
      return 0
    fi
    echo "::warning::rsync failed for optional $label at $remote_path" >&2
    return 0
  fi
  echo "[collect] info: $label not found at $remote_path; skipping" >&2
  return 0
}

sync_file "${PRETRAIN_ENCODER_PATH:-${PRETRAIN_EXPERIMENT_ROOT}/pretrain/encoder.pt}" "encoder"
sync_file "${PRETRAIN_MANIFEST:-${PRETRAIN_EXPERIMENT_ROOT}/artifacts/encoder_manifest.json}" "manifest"
sync_file "${PRETRAIN_EXPERIMENT_ROOT}/pretrain/stage-outputs/pretrain.json" "stage-outputs"
sync_file "${PRETRAIN_STATE_FILE:-${PRETRAIN_EXPERIMENT_ROOT}/pretrain_state.json}" "pretrain-state"

tox21_remote="${PRETRAIN_TOX21_ENV:-}"
if [[ -z "$tox21_remote" && -n "$PRETRAIN_EXPERIMENT_ROOT" ]]; then
  tox21_remote="${PRETRAIN_EXPERIMENT_ROOT%/}/tox21_gate.env"
fi
sync_optional_file "$tox21_remote" "tox21_gate.env" "$DEST_DIR"

required=(
  "$DEST_DIR/encoder.pt"
  "$DEST_DIR/encoder_manifest.json"
  "$DEST_DIR/pretrain.json"
  "$DEST_DIR/pretrain_state.json"
)

missing=()
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    missing+=("$path")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "::error::missing required pretrain artifacts: ${missing[*]}" >&2
  exit 1
fi

for path in "${required[@]}"; do
  if [[ -f "$path" ]]; then
    size=$(wc -c <"$path" 2>/dev/null || printf '0')
    echo "[collect] ready: $path (${size} bytes)" >&2
  fi
done
