#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config
export MJEPACI_STAGE="pretrain"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

echo "[ci] EXP_ID=${EXP_ID}" >&2
echo "[ci] EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT}" >&2
echo "[ci] DATA_ROOT=${DATA_ROOT:-<unset>}" >&2
echo "[ci] EXPERIMENT_DIR=${EXPERIMENT_DIR}" >&2
echo "[ci] ARTIFACTS_DIR=${ARTIFACTS_DIR}" >&2
echo "[ci] PRETRAIN_ARTIFACTS_DIR=${PRETRAIN_ARTIFACTS_DIR}" >&2

# ensure key directories are exported for shims
export PRETRAIN_DIR PRETRAIN_ARTIFACTS_DIR

export WANDB_NAME="pretrain"
export WANDB_JOB_TYPE="pretrain"

echo "[pretrain] using experiment id=${PRETRAIN_EXP_ID} root=${PRETRAIN_EXPERIMENT_ROOT}" >&2
echo "[pretrain] artifacts dir=${PRETRAIN_ARTIFACTS_DIR}" >&2
echo "[pretrain] manifest path=${PRETRAIN_MANIFEST}" >&2
echo "[pretrain] state path=${PRETRAIN_STATE_FILE} (legacy=${PRETRAIN_STATE_FILE_LEGACY:-n/a})" >&2
mkdir -p "$PRETRAIN_ARTIFACTS_DIR"

export STAGE_OUTPUTS_DIR="${PRETRAIN_DIR}/stage-outputs"
mkdir -p "$STAGE_OUTPUTS_DIR"

#ensure the parm matches train_jepa_ci.yml

if [[ -n "${STAGE_BIN:-}" ]]; then
  "$STAGE_BIN" pretrain
else
  run_stage pretrain
fi

encoder_ckpt="${PRETRAIN_DIR}/encoder.pt"
manifest_path="${PRETRAIN_MANIFEST}"
stage_outputs_json="${STAGE_OUTPUTS_DIR}/pretrain.json"

if [[ ! -f "$stage_outputs_json" ]]; then
  echo "[ci] error: expected pretrain.json at ${stage_outputs_json}" >&2
  exit 1
fi

# Resolve a Python interpreter for the parsing and bookkeeping steps below.
# Prefer an existing "python" binary, otherwise fall back to micromamba.
python_cmd=()
resolve_ci_python python_cmd

# If the training command emitted a stage outputs JSON, honour the absolute
# paths recorded there.  This guards against mismatched environment variables
# (e.g. stale PRETRAIN_* overrides) and gives us a concrete file to copy.
recorded_manifest=""
recorded_encoder=""

while IFS='=' read -r key value; do
  case "$key" in
    manifest_path)
      recorded_manifest="$value"
      ;;
    encoder_checkpoint)
      recorded_encoder="$value"
      ;;
  esac
done < <("${python_cmd[@]}" - "$stage_outputs_json" <<'PY'
import json
import os
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh) or {}
except Exception:
    sys.exit(1)

def emit(key, value):
    if isinstance(value, str) and value.strip():
        print(f"{key}={value.strip()}")

emit("manifest_path", data.get("manifest_path"))
emit("encoder_checkpoint", data.get("encoder_checkpoint"))
PY
)

if [[ -n "${recorded_encoder:-}" ]]; then
  encoder_ckpt="$recorded_encoder"
fi

if [[ -n "${recorded_manifest:-}" ]]; then
  manifest_path="$recorded_manifest"
fi

if [[ ! -f "$manifest_path" ]]; then
  echo "[ci] error: expected pretrain.json at ${stage_outputs_json} and manifest at ${manifest_path}" >&2
  exit 1
fi

export PRETRAIN_MANIFEST="$manifest_path"
export PRETRAIN_ENCODER_PATH="$encoder_ckpt"

if [[ ! -f "$encoder_ckpt" ]]; then
  echo "[pretrain] expected encoder checkpoint missing: $encoder_ckpt" >&2
  exit 1
fi

if [[ ! -f "$manifest_path" ]]; then
  echo "[pretrain] expected encoder manifest missing: $manifest_path" >&2
  exit 1
fi

mkdir -p "${PRETRAIN_ARTIFACTS_DIR}"

# Preserve key artifacts alongside checkpoints so downstream jobs can fetch them easily.
if [[ -f "$PRETRAIN_DIR/encoder.pt" ]]; then
  ln -sf "$PRETRAIN_DIR/encoder.pt" "$STAGE_OUTPUTS_DIR/encoder.pt"
fi

if [[ -f "$PRETRAIN_MANIFEST" ]]; then
  mkdir -p "${PRETRAIN_DIR}/artifacts"
  local_manifest="${PRETRAIN_DIR}/artifacts/encoder_manifest.json"
  if [[ "$PRETRAIN_MANIFEST" != "$local_manifest" ]]; then
    cp "$PRETRAIN_MANIFEST" "$local_manifest"
  fi
  ln -sf "$PRETRAIN_MANIFEST" "$STAGE_OUTPUTS_DIR/encoder_manifest.json"
fi

state_path="${PRETRAIN_STATE_FILE}"
state_legacy="${PRETRAIN_STATE_FILE_LEGACY:-}"
state_dir="$(dirname "$state_path")"
mkdir -p "$state_dir"

"${python_cmd[@]}" - "$state_path" "$state_legacy" "$PRETRAIN_EXP_ID" "$PRETRAIN_EXPERIMENT_ROOT" \
  "$PRETRAIN_ARTIFACTS_DIR" "$manifest_path" "$encoder_ckpt" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

state_path, legacy_path, exp_id, exp_root, artifacts_dir, manifest_path, encoder_path = sys.argv[1:8]

tox21_env = os.path.abspath(os.path.join(exp_root, "tox21_gate.env"))

payload = {
    "id": exp_id,
    "pretrain_exp_id": exp_id,
    "experiment_root": os.path.abspath(exp_root),
    "artifacts_dir": os.path.abspath(artifacts_dir),
    "encoder_manifest": os.path.abspath(manifest_path),
    "encoder_checkpoint": os.path.abspath(encoder_path),
    "tox21_env": tox21_env,
    "state_updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

tmp_path = state_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp_path, state_path)
print(f"[pretrain] wrote state to {state_path} (id={exp_id})")

legacy_path = legacy_path.strip()
if legacy_path and os.path.abspath(legacy_path) != os.path.abspath(state_path):
    legacy_dir = os.path.dirname(legacy_path)
    os.makedirs(legacy_dir, exist_ok=True)
    tmp_legacy = legacy_path + ".tmp"
    with open(tmp_legacy, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_legacy, legacy_path)
    print(f"[pretrain] synced legacy state to {legacy_path}")
PY

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages
