#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config
export MJEPACI_STAGE="pretrain"

if [[ -n "${MJEPACI_STAGE_SHIM:-}" && -x "${MJEPACI_STAGE_SHIM}" ]]; then
  # When exercising the CI flow through a stage shim (used by unit tests) we
  # still rely on helpers such as ``resolve_ci_python`` from common.sh.  Source
  # it explicitly here because the early-return path below would otherwise skip
  # the later ``source" statements, leading to ``command not found`` failures
  # (exit 127) when the helpers are invoked.
  source "$(dirname "$0")/common.sh"

  if [[ -z "${EXP_ID:-}" || -z "${EXPERIMENTS_ROOT:-}" ]]; then
    echo "[ci] error: MJEPACI_STAGE_SHIM requires EXP_ID and EXPERIMENTS_ROOT" >&2
    exit 1
  fi

  EXPERIMENT_DIR="${EXPERIMENTS_ROOT%/}/${EXP_ID}"
  PRETRAIN_DIR="${PRETRAIN_DIR:-$EXPERIMENT_DIR}"
  ARTIFACTS_DIR="${ARTIFACTS_DIR:-${EXPERIMENT_DIR}/artifacts}"
  PRETRAIN_ARTIFACTS_DIR="${PRETRAIN_ARTIFACTS_DIR:-$ARTIFACTS_DIR}"

  export EXP_ID EXPERIMENTS_ROOT EXPERIMENT_DIR PRETRAIN_DIR ARTIFACTS_DIR PRETRAIN_ARTIFACTS_DIR

  STAGE_OUTPUTS_DIR="${PRETRAIN_DIR}/stage-outputs"
  export STAGE_OUTPUTS_DIR
  mkdir -p "${PRETRAIN_DIR}" "$STAGE_OUTPUTS_DIR" "${PRETRAIN_ARTIFACTS_DIR}"

  echo "[ci] (test) EXP_ID=${EXP_ID} EXPERIMENT_DIR=${EXPERIMENT_DIR} PRETRAIN_DIR=${PRETRAIN_DIR} PRETRAIN_ARTIFACTS_DIR=${PRETRAIN_ARTIFACTS_DIR} STAGE_BIN=${MJEPACI_STAGE_SHIM}" >&2

  "${MJEPACI_STAGE_SHIM}" pretrain

  expected_encoder="${PRETRAIN_DIR}/encoder.pt"
  expected_stage_outputs="${STAGE_OUTPUTS_DIR}/pretrain.json"
  expected_manifest="${PRETRAIN_ARTIFACTS_DIR}/encoder_manifest.json"

  python_cmd=()
  resolve_ci_python python_cmd

  for required in "$expected_encoder" "$expected_stage_outputs"; do
    if [[ ! -f "$required" ]]; then
      echo "[ci] error: expected ${required} not found" >&2
      exit 1
    fi
  done

  if [[ ! -f "$expected_manifest" ]]; then
    recorded_manifest=""
    recorded_encoder=""

    if [[ -f "$expected_stage_outputs" && ${#python_cmd[@]} -gt 0 ]]; then
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

      normalize_stage_path() {
        local raw="$1"
        shift || true
        local candidate
        if [[ -z "$raw" ]]; then
          return 1
        fi
        if [[ "$raw" == /* ]]; then
          if [[ -e "$raw" ]]; then
            printf '%s\n' "$raw"
            return 0
          fi
          return 1
        fi
        for candidate in "$@"; do
          [[ -z "$candidate" ]] && continue
          candidate="${candidate%/}/$raw"
          if [[ -e "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
          fi
        done
        return 1
      }

      if [[ -n "${recorded_manifest:-}" ]]; then
        if normalized=$(normalize_stage_path "$recorded_manifest" "$PRETRAIN_ARTIFACTS_DIR" "$PRETRAIN_DIR" "$EXPERIMENT_DIR" \
          "$PRETRAIN_EXPERIMENT_ROOT" "$EXPERIMENTS_ROOT"); then
          recorded_manifest="$normalized"
        fi
      fi

      if [[ -n "${recorded_encoder:-}" ]]; then
        if normalized=$(normalize_stage_path "$recorded_encoder" "$PRETRAIN_DIR" "$PRETRAIN_ARTIFACTS_DIR" "$EXPERIMENT_DIR" \
          "$PRETRAIN_EXPERIMENT_ROOT" "$EXPERIMENTS_ROOT"); then
          recorded_encoder="$normalized"
        fi
      fi
    fi

    if [[ -n "${recorded_manifest:-}" && -f "$recorded_manifest" ]]; then
      mkdir -p "$(dirname "$expected_manifest")"
      if [[ "$recorded_manifest" != "$expected_manifest" ]]; then
        cp "$recorded_manifest" "$expected_manifest"
      fi
    fi

    if [[ -n "${recorded_encoder:-}" && -f "$recorded_encoder" && "$recorded_encoder" != "$expected_encoder" ]]; then
      mkdir -p "$(dirname "$expected_encoder")"
      ln -sf "$recorded_encoder" "$expected_encoder"
    fi

    if [[ ! -f "$expected_manifest" ]]; then
      encoder_for_manifest="$expected_encoder"
      if [[ -n "${recorded_encoder:-}" && -f "$recorded_encoder" ]]; then
        encoder_for_manifest="$recorded_encoder"
      fi
      mkdir -p "$(dirname "$expected_manifest")"
      if [[ ${#python_cmd[@]} -gt 0 ]]; then
        "${python_cmd[@]}" - "$expected_manifest" "$encoder_for_manifest" "$EXP_ID" <<'PY'
import json
import os
import sys

manifest_path, encoder_path, exp_id = sys.argv[1:4]
payload = {
    "paths": {"encoder": os.path.abspath(encoder_path)},
}
if exp_id and exp_id.strip():
    payload["pretrain_exp_id"] = exp_id

with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
      else
        cat >"$expected_manifest" <<EOF
{
  "paths": {"encoder": "${encoder_for_manifest}"}
}
EOF
      fi
    fi
  fi

  if [[ ! -f "$expected_manifest" ]]; then
    echo "[ci] error: expected ${expected_manifest} not found" >&2
    exit 1
  fi

  state_path="${EXPERIMENT_DIR}/pretrain_state.json"
  state_legacy="${EXPERIMENTS_ROOT%/}/pretrain_state.json"
  pretrain_id="${EXP_ID}"
  pretrain_root="${EXPERIMENT_DIR}"
  python_cmd=()
  resolve_ci_python python_cmd

  if [[ ${#python_cmd[@]} -gt 0 ]]; then
    "${python_cmd[@]}" - "$state_path" "$state_legacy" "$pretrain_id" "$pretrain_root" \
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
os.makedirs(os.path.dirname(state_path), exist_ok=True)
with open(tmp_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp_path, state_path)

legacy_path = legacy_path.strip()
if legacy_path:
    legacy_dir = os.path.dirname(legacy_path)
    os.makedirs(legacy_dir, exist_ok=True)
    tmp_legacy = legacy_path + ".tmp"
    with open(tmp_legacy, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_legacy, legacy_path)
PY
  else
    echo "[ci] warn: unable to write pretrain_state.json because no python interpreter was resolved" >&2
  fi

  exit 0
fi

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

# Phase-2 sweeps may have been executed under a different experiment ID.
# When this script runs in a fresh CI shell the GRID_* variables might be
# empty (or still pointing at the current EXP_ID).  Backfill them from the
# latest lineage metadata so the pretrain stage can locate the sweep
# outputs created during run-grid-phase2.
ci_pretrain_backfill_phase2_bindings() {
  local grid_hint="${GRID_DIR:-}" sweep_file=""
  if [[ -n "$grid_hint" ]]; then
    sweep_file="${grid_hint%/}/phase2_sweep_id.txt"
  fi

  if [[ -n "$grid_hint" && -f "$sweep_file" ]]; then
    return 0
  fi

  local root="${EXPERIMENTS_ROOT%/}"
  local default_id="${GRID_EXP_ID:-${PRETRAIN_EXP_ID:-${EXP_ID:-}}}"

  resolve_ci_python python_cmd
  local payload=""
  if ! payload="$("${python_cmd[@]}" scripts/ci/resolve_lineage_ids.py --root "$root" --default-id "$default_id" 2>/dev/null)"; then
    return 0
  fi

  local _resolved=()
  if ! mapfile -t _resolved < <(python_inline "$payload" <<'PY'
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

print(emit("grid_exp_id"))
print(emit("grid_dir"))
print(emit("pretrain_exp_id"))
PY
  ); then
    return 0
  fi

  local new_grid_id="${_resolved[0]:-}" new_grid_dir="${_resolved[1]:-}" new_pretrain_id="${_resolved[2]:-}"
  unset _resolved || true

  if [[ -z "$new_grid_dir" ]]; then
    return 0
  fi

  local prev_grid_id="${GRID_EXP_ID:-}"
  local prev_pretrain_id="${PRETRAIN_EXP_ID:-}"

  GRID_DIR="$new_grid_dir"
  export GRID_DIR

  if [[ -n "$new_grid_id" ]]; then
    GRID_EXP_ID="$new_grid_id"
    export GRID_EXP_ID
  fi

  if [[ -n "$new_pretrain_id" ]]; then
    PRETRAIN_EXP_ID="$new_pretrain_id"
    export PRETRAIN_EXP_ID
  fi

  if [[ -z "${GRID_SOURCE_DIR:-}" ]]; then
    GRID_SOURCE_DIR="$GRID_DIR"
    export GRID_SOURCE_DIR
  fi

  if declare -F ci_phase2_refresh_lineage_bindings >/dev/null 2>&1; then
    ci_phase2_refresh_lineage_bindings "${PRETRAIN_EXP_ID:-}" "${GRID_EXP_ID:-}" "$prev_pretrain_id" "$prev_grid_id"
  fi

  echo "[pretrain] discovered Phase-2 lineage GRID_EXP_ID=${GRID_EXP_ID:-<unset>} GRID_DIR=${GRID_DIR:-<unset>}" >&2
}

ci_pretrain_backfill_phase2_bindings || true

if (( FROZEN )); then
  echo "[pretrain] encoder lineage ${PRETRAIN_EXP_ID:-<unset>} is frozen; skipping pretrain." >&2
  if [[ ! -f "${PRETRAIN_DIR}/encoder.pt" ]]; then
    echo "[pretrain][warn] expected encoder checkpoint missing: ${PRETRAIN_DIR}/encoder.pt" >&2
  fi
  exit 0
fi

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

# Resolve relative paths emitted by the stage shim.
normalize_stage_path() {
  local raw="$1"
  shift || true
  local candidate
  if [[ -z "$raw" ]]; then
    return 1
  fi
  if [[ "$raw" == /* ]]; then
    if [[ -e "$raw" ]]; then
      printf '%s\n' "$raw"
      return 0
    fi
    return 1
  fi
  for candidate in "$@"; do
    [[ -z "$candidate" ]] && continue
    candidate="${candidate%/}/$raw"
    if [[ -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ -n "${recorded_manifest:-}" ]]; then
  if normalized=$(normalize_stage_path "$recorded_manifest" "$PRETRAIN_ARTIFACTS_DIR" "$PRETRAIN_DIR" "$EXPERIMENT_DIR" "$PRETRAIN_EXPERIMENT_ROOT" "$EXPERIMENTS_ROOT"); then
    recorded_manifest="$normalized"
  fi
fi

if [[ -n "${recorded_encoder:-}" ]]; then
  if normalized=$(normalize_stage_path "$recorded_encoder" "$PRETRAIN_DIR" "$PRETRAIN_ARTIFACTS_DIR" "$EXPERIMENT_DIR" "$PRETRAIN_EXPERIMENT_ROOT" "$EXPERIMENTS_ROOT"); then
    recorded_encoder="$normalized"
  fi
fi

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

# If the stage skipped manifest creation, synthesise a minimal manifest so
# downstream stages still find the encoder checkpoint.
if [[ ! -f "$expected_manifest" && -f "$expected_encoder" ]]; then
  "${python_cmd[@]}" - "$expected_manifest" "$expected_encoder" "$PRETRAIN_EXP_ID" <<'PY'
import json
import os
import sys

manifest_path, encoder_path, exp_id = sys.argv[1:4]
manifest_dir = os.path.dirname(manifest_path)
os.makedirs(manifest_dir, exist_ok=True)

payload = {
    "paths": {"encoder": os.path.abspath(encoder_path)},
}

if exp_id and exp_id.strip():
    payload["pretrain_exp_id"] = exp_id

with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
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
