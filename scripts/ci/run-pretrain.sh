#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config
export MJEPACI_STAGE="pretrain"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

if [[ -n "${MJEPACI_STAGE_SHIM:-}" ]]; then
  STAGE_BIN="${MJEPACI_STAGE_SHIM}"
elif [[ -z "${STAGE_BIN:-}" ]]; then
  STAGE_BIN="run_stage"
fi

export STAGE_BIN
ci_print_env_diag "$STAGE_BIN"

export EXP_ID EXPERIMENTS_ROOT EXPERIMENT_DIR PRETRAIN_DIR ARTIFACTS_DIR PRETRAIN_ARTIFACTS_DIR

export WANDB_NAME="pretrain"
export WANDB_JOB_TYPE="pretrain"

mkdir -p "$PRETRAIN_ARTIFACTS_DIR"

export STAGE_OUTPUTS_DIR="${PRETRAIN_DIR}/stage-outputs"
mkdir -p "$STAGE_OUTPUTS_DIR"

"$STAGE_BIN" pretrain

expected_stage_outputs="${STAGE_OUTPUTS_DIR}/pretrain.json"
expected_manifest="${PRETRAIN_ARTIFACTS_DIR}/encoder_manifest.json"
expected_encoder="${PRETRAIN_DIR}/encoder.pt"

if [[ ! -f "$expected_stage_outputs" ]]; then
  echo "[ci] error: expected file ${expected_stage_outputs} not found" >&2
  exit 1
fi

python_cmd=()
resolve_ci_python python_cmd

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
done < <("${python_cmd[@]}" - "$expected_stage_outputs" <<'PY'
import json
import os
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh) or {}
except Exception:
    sys.exit(0)

def emit(key, value):
    if isinstance(value, str) and value.strip():
        print(f"{key}={value.strip()}")

emit("manifest_path", data.get("manifest_path"))
emit("encoder_checkpoint", data.get("encoder_checkpoint"))
PY
)

manifest_path="$expected_manifest"
encoder_ckpt="$expected_encoder"

if [[ -n "${recorded_manifest:-}" ]]; then
  manifest_path="$recorded_manifest"
fi

if [[ -n "${recorded_encoder:-}" ]]; then
  encoder_ckpt="$recorded_encoder"
fi

if [[ -f "$manifest_path" && "$manifest_path" != "$expected_manifest" ]]; then
  mkdir -p "$(dirname "$expected_manifest")"
  cp "$manifest_path" "$expected_manifest"
fi

if [[ -f "$encoder_ckpt" && "$encoder_ckpt" != "$expected_encoder" ]]; then
  mkdir -p "$(dirname "$expected_encoder")"
  ln -sf "$encoder_ckpt" "$expected_encoder"
fi

if [[ ! -f "$expected_manifest" ]]; then
  echo "[ci] error: expected file ${expected_manifest} not found" >&2
  exit 1
fi

if [[ ! -f "$expected_encoder" ]]; then
  echo "[ci] error: expected file ${expected_encoder} not found" >&2
  exit 1
fi

export PRETRAIN_MANIFEST="$expected_manifest"
export PRETRAIN_ENCODER_PATH="$expected_encoder"

mkdir -p "${PRETRAIN_DIR}/artifacts"
ln -sf "$expected_encoder" "$STAGE_OUTPUTS_DIR/encoder.pt"
ln -sf "$expected_manifest" "$STAGE_OUTPUTS_DIR/encoder_manifest.json"

state_path="${PRETRAIN_STATE_FILE}"
state_legacy="${PRETRAIN_STATE_FILE_LEGACY:-}"
state_dir="$(dirname "$state_path")"
mkdir -p "$state_dir"

"${python_cmd[@]}" - "$state_path" "$state_legacy" "$PRETRAIN_EXP_ID" "$PRETRAIN_EXPERIMENT_ROOT" \
  "$PRETRAIN_ARTIFACTS_DIR" "$expected_manifest" "$expected_encoder" <<'PY'
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
