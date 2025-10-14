#!/usr/bin/env bash
# Fetch the canonical pretrain_state.json from the Vast runner and emit
# EXP_ID-derived metadata for downstream GitHub Actions steps.
set -euo pipefail

: "${SSH_KEY:?SSH_KEY secret required}"
: "${VAST_USER:?VAST_USER required}"
: "${VAST_HOST:?VAST_HOST required}"
: "${VAST_PORT:?VAST_PORT required}"
: "${EXPERIMENTS_ROOT:?EXPERIMENTS_ROOT required}"
: "${GITHUB_ENV:?GITHUB_ENV must be set}"
: "${GITHUB_OUTPUT:?GITHUB_OUTPUT must be set}"

DEST_DIR="${1:-pretrain_artifacts}"
STATE_DEST="${DEST_DIR}/pretrain_state.json"

mkdir -p "$DEST_DIR" ~/.ssh
KEY_PATH=~/.ssh/vast_key
trap 'rm -f "$KEY_PATH"' EXIT
printf '%s' "$SSH_KEY" >"$KEY_PATH"
chmod 600 "$KEY_PATH"

REMOTE="${VAST_USER}@${VAST_HOST}"
REMOTE_STATE="${EXPERIMENTS_ROOT}/pretrain_state.json"
STATE_SOURCE=""

SCP_CMD=(scp -P "$VAST_PORT" -i "$KEY_PATH" -o StrictHostKeyChecking=no)
SSH_CMD=(ssh -i "$KEY_PATH" -p "$VAST_PORT" -o StrictHostKeyChecking=no)

download_state() {
  local remote_path="$1"
  [[ -n "$remote_path" ]] || return 1
  if "${SCP_CMD[@]}" "$REMOTE:$remote_path" "$STATE_DEST"; then
    STATE_SOURCE="$remote_path"
    echo "[collect] downloaded $remote_path" >&2
    return 0
  fi
  return 1
}

if ! download_state "$REMOTE_STATE"; then
  fallback_path=""
  if fallback_path="$("${SSH_CMD[@]}" "$REMOTE" bash -s -- "$EXPERIMENTS_ROOT" <<'EOS' 2>/dev/null)
set -euo pipefail
root="${1%/}"
if [[ -z "$root" || ! -d "$root" ]]; then
  exit 0
fi
shopt -s nullglob
latest=""
latest_mtime=0
for path in "$root"/*/pretrain_state.json; do
  [[ -f "$path" ]] || continue
  mtime=0
  # ``stat -c`` fails on BSD/macOS, so fall back to ``stat -f`` but keep
  # ``set -e`` intact by guarding failures inside conditionals.
  if ! mtime=$(stat -c '%Y' "$path" 2>/dev/null); then
    if ! mtime=$(stat -f '%m' "$path" 2>/dev/null); then
      mtime=0
    fi
  fi
  if [[ -z "$latest" || "$mtime" -gt "$latest_mtime" ]]; then
    latest="$path"
    latest_mtime="$mtime"
  fi
done
if [[ -n "$latest" ]]; then
  printf '%s' "$latest"
fi
EOS
)"; then
    fallback_path="${fallback_path:-}"
  else
    fallback_path=""
  fi

  if [[ -n "$fallback_path" ]]; then
    # Normalise whitespace just in case remote shells emit trailing newlines.
    fallback_path="${fallback_path//$'\r'/}"
    fallback_path="${fallback_path//$'\n'/}"
    echo "::warning::unable to download $REMOTE_STATE; retrying with discovered $fallback_path" >&2
    if ! download_state "$fallback_path"; then
      echo "::warning::failed to download fallback pretrain state at $fallback_path" >&2
      exit 0
    fi
  else
    echo "::warning::unable to download $REMOTE_STATE; downstream steps may fail" >&2
    exit 0
  fi
fi

if [[ ! -s "$STATE_DEST" ]]; then
  echo "::warning::downloaded state file empty: $STATE_DEST" >&2
  exit 0
fi

parse_output=$(python3 - "$STATE_DEST" <<'PY'
import os
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    data = json.load(fh) or {}

def emit(key, value):
    if isinstance(value, str):
        value = value.strip()
        if value:
            print(f"{key}={value}")

exp_id = data.get("id") or data.get("pretrain_exp_id")
if exp_id:
    print(f"exp_id={exp_id}")

root = data.get("experiment_root")
if root:
    emit("experiment_root", root)
    joined = os.path.join(root, "pretrain_state.json")
    print(f"state_path={joined}")

manifest = data.get("encoder_manifest")
if manifest:
    emit("manifest_path", manifest)

encoder = data.get("encoder_checkpoint")
if encoder:
    emit("encoder_path", encoder)

tox21_env = data.get("tox21_env")
if tox21_env:
    emit("tox21_env", tox21_env)
PY
)

exp_id=""
experiment_root=""
manifest_path=""
encoder_path=""
tox21_env=""
state_path=""

while IFS='=' read -r key value; do
  case "$key" in
    exp_id) exp_id="$value" ;;
    experiment_root) experiment_root="$value" ;;
    manifest_path) manifest_path="$value" ;;
    encoder_path) encoder_path="$value" ;;
    tox21_env) tox21_env="$value" ;;
    state_path) state_path="$value" ;;
  esac
done <<<"$parse_output"

if [[ -z "$experiment_root" && -n "$STATE_SOURCE" ]]; then
  experiment_root="${STATE_SOURCE%/pretrain_state.json}"
fi

if [[ -z "$state_path" && -n "$STATE_SOURCE" ]]; then
  state_path="$STATE_SOURCE"
fi

if [[ -n "$exp_id" ]]; then
  {
    echo "EXP_ID=$exp_id"
    echo "PRETRAIN_EXP_ID=$exp_id"
    echo "GRID_EXP_ID=$exp_id"
  } >>"$GITHUB_ENV"
fi

if [[ -n "$experiment_root" ]]; then
  {
    echo "PRETRAIN_EXPERIMENT_ROOT=$experiment_root"
    echo "GRID_EXPERIMENT_ROOT=$experiment_root"
    echo "EXP_ROOT=$experiment_root"
  } >>"$GITHUB_ENV"
fi

[[ -n "$manifest_path" ]] && echo "PRETRAIN_MANIFEST=$manifest_path" >>"$GITHUB_ENV"
[[ -n "$encoder_path" ]] && echo "PRETRAIN_ENCODER_PATH=$encoder_path" >>"$GITHUB_ENV"
[[ -n "$tox21_env" ]] && echo "PRETRAIN_TOX21_ENV=$tox21_env" >>"$GITHUB_ENV"

if [[ -n "$state_path" ]]; then
  {
    echo "PRETRAIN_STATE_FILE=$state_path"
    echo "PRETRAIN_STATE_FILE_CANONICAL=$state_path"
  } >>"$GITHUB_ENV"
fi

{
  [[ -n "$exp_id" ]] && echo "exp_id=$exp_id"
  [[ -n "$experiment_root" ]] && echo "experiment_root=$experiment_root"
  [[ -n "$manifest_path" ]] && echo "manifest_path=$manifest_path"
  [[ -n "$encoder_path" ]] && echo "encoder_path=$encoder_path"
  [[ -n "$tox21_env" ]] && echo "tox21_env=$tox21_env"
} >>"$GITHUB_OUTPUT"
