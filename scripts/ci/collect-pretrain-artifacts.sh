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
  if ! remote_file_exists "$remote_path"; then
    echo "::warning::remote $label missing at $remote_path" >&2
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

remote_path_exists() {
  local remote_path="$1"
  "${SSH_CMD[@]}" "$REMOTE" bash -s -- "$remote_path" <<'EOS' >/dev/null 2>&1
set -euo pipefail
path="${1:-}"
if [[ -e "$path" ]]; then
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

sync_optional_path() {
  local remote_path="$1"
  local label="$2"
  local dest_dir="$3"
  if [[ -z "$remote_path" ]]; then
    echo "[collect] info: $label remote path empty; skipping" >&2
    return 0
  fi
  if remote_path_exists "$remote_path"; then
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
stage_outputs_missing=1
stage_outputs_local="$DEST_DIR/pretrain.json"

stage_output_candidates=()
if [[ -n "${PRETRAIN_STAGE_OUTPUTS:-}" ]]; then
  stage_output_candidates+=("${PRETRAIN_STAGE_OUTPUTS%/}/pretrain.json")
fi
if [[ -n "${PRETRAIN_CACHE_DIR:-}" ]]; then
  stage_output_candidates+=("${PRETRAIN_CACHE_DIR%/}/stage-outputs/pretrain.json")
  stage_output_candidates+=("${PRETRAIN_CACHE_DIR%/}/pretrain/stage-outputs/pretrain.json")
fi
if [[ -n "${PRETRAIN_DIR:-}" ]]; then
  stage_output_candidates+=("${PRETRAIN_DIR%/}/stage-outputs/pretrain.json")
fi
if [[ -n "${PRETRAIN_EXPERIMENT_ROOT:-}" ]]; then
  stage_output_candidates+=("${PRETRAIN_EXPERIMENT_ROOT%/}/pretrain/stage-outputs/pretrain.json")
fi
if [[ -n "${PRETRAIN_MANIFEST:-}" ]]; then
  manifest_dir="${PRETRAIN_MANIFEST%/*}"
  if [[ "${manifest_dir##*/}" == "stage-outputs" ]]; then
    stage_output_candidates+=("${manifest_dir%/}/pretrain.json")
  fi
fi

seen_candidates=()
for candidate in "${stage_output_candidates[@]}"; do
  [[ -n "$candidate" ]] || continue
  skip=0
  for seen in "${seen_candidates[@]}"; do
    if [[ "$seen" == "$candidate" ]]; then
      skip=1
      break
    fi
  done
  (( skip )) && continue
  seen_candidates+=("$candidate")
  if remote_file_exists "$candidate"; then
    echo "[collect] found stage outputs at $candidate" >&2
    sync_file "$candidate" "stage-outputs"
    if [[ -f "$stage_outputs_local" ]]; then
      stage_outputs_missing=0
      break
    fi
  else
    echo "[collect] info: stage outputs not present at $candidate" >&2
  fi
done
if [[ -f "$stage_outputs_local" ]]; then
  stage_outputs_missing=0
fi
if (( stage_outputs_missing )); then
  echo "::notice::stage-outputs missing remotely; will attempt reconstruction" >&2
fi
sync_file "${PRETRAIN_STATE_FILE:-${PRETRAIN_EXPERIMENT_ROOT}/pretrain_state.json}" "pretrain-state"

# Rebuild stage outputs locally when the remote file is missing but the other
# artifacts were synced successfully.  This keeps downstream jobs working on
# Vast runs where ``pretrain.json`` was never materialised on the runner.
rebuild_stage_outputs() {
  local dest_dir="$1"
  local stage_json="$dest_dir/pretrain.json"
  local encoder_path="${2:-$dest_dir/encoder.pt}"
  local manifest_path="${3:-$dest_dir/encoder_manifest.json}"

  if [[ -f "$stage_json" ]]; then
    return 0
  fi

  if [[ ! -f "$manifest_path" ]]; then
    echo "::warning::cannot reconstruct pretrain stage outputs; missing manifest at ${manifest_path}" >&2
    return 0
  fi

  # Prefer the synced encoder checkpoint but fall back to a manifest hint when
  # that file is absent.  This is resilient to manifests that already embed an
  # absolute encoder path produced by earlier stages.
  local encoder_candidate="$encoder_path"
  if [[ ! -f "$encoder_candidate" ]]; then
    encoder_candidate="$(python - <<'PY'
import json
import os
import sys

manifest_path = sys.argv[1]
encoder = ""
try:
    with open(manifest_path, encoding="utf-8") as f:
        payload = json.load(f)
    encoder = payload.get("paths", {}).get("encoder", "")
except Exception:
    encoder = ""

if encoder and os.path.exists(encoder):
    print(os.path.abspath(encoder))
PY
"$manifest_path")"
  fi

  if [[ -z "$encoder_candidate" || ! -f "$encoder_candidate" ]]; then
    echo "::warning::cannot reconstruct pretrain stage outputs; missing encoder at ${encoder_path}" >&2
    return 0
  fi

  local abs_encoder abs_manifest
  abs_encoder="$(cd "$(dirname "$encoder_candidate")" && pwd)/$(basename "$encoder_candidate")"
  abs_manifest="$(cd "$(dirname "$manifest_path")" && pwd)/$(basename "$manifest_path")"

  cat >"$stage_json" <<EOF
{
  "encoder_checkpoint": "${abs_encoder}",
  "manifest_path": "${abs_manifest}",
  "manifest_updated": false
}
EOF

  echo "::warning::reconstructed missing pretrain stage outputs at ${stage_json}" >&2
}

tox21_remote="${PRETRAIN_TOX21_ENV:-}"
if [[ -z "$tox21_remote" && -n "$PRETRAIN_EXPERIMENT_ROOT" ]]; then
  tox21_remote="${PRETRAIN_EXPERIMENT_ROOT%/}/tox21_gate.env"
fi
sync_optional_file "$tox21_remote" "tox21_gate.env" "$DEST_DIR"

if [[ -n "${PRETRAIN_EXPERIMENT_ROOT:-}" ]]; then
  graphs_remote="${PRETRAIN_EXPERIMENT_ROOT%/}/graphs"
  reports_remote="${PRETRAIN_EXPERIMENT_ROOT%/}/reports"
  sync_optional_path "$graphs_remote" "graph visuals" "$DEST_DIR"
  sync_optional_path "$reports_remote" "reports" "$DEST_DIR"
fi

if (( stage_outputs_missing )); then
  rebuild_stage_outputs "$DEST_DIR"
fi

if [[ ! -f "$stage_outputs_local" ]]; then
  echo "::notice::pretrain stage outputs absent after reconstruction; continuing without pretrain.json" >&2
fi

required=(
  "$DEST_DIR/encoder.pt"
  "$DEST_DIR/encoder_manifest.json"
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
