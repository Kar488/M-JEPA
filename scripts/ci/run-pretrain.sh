#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

unset BESTCFG_NO_EPOCHS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PRETRAIN_SAMPLE_UNLABELED:-}" ]]; then
  BESTCFG_SKIP="sample_unlabeled ${BESTCFG_SKIP:-}"
else
  BESTCFG_KEEP="sample_unlabeled ${BESTCFG_KEEP:-}"
fi
export BESTCFG_SKIP BESTCFG_KEEP

ensure_tox21_gate_stub() {
  # Ensure downstream artifact syncs never fail due to a missing optional
  # tox21 gate file.  When the gate has not run yet seed a conservative default.
  local root="${1:-${PRETRAIN_EXPERIMENT_ROOT:-}}"
  [[ -n "$root" ]] || return 0
  local gate_path="${root%/}/tox21_gate.env"
  if [[ -f "$gate_path" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$gate_path")"
  {
    echo "# Seeded by run-pretrain.sh to stabilise artifact collection"
    echo "TOX21_MET_GATE=false"
  } >"$gate_path"
}

normalize_dataset_dir() {
  local var_name="$1" default_dir="$2"
  local current="${!var_name:-}" data_root="${DATA_ROOT:-${APP_DIR:-/srv/mjepa}}"
  if [[ -z "$current" ]]; then
    current="$default_dir"
  fi
  for cache_hint in "${data_root%/}/cache/graphs_10m" "${data_root%/}/cache/graphs_250k"; do
    if [[ -n "$cache_hint" && "$current" == "$cache_hint" ]]; then
      echo "[ci] info: ${var_name} points to a cache (${current}); switching to dataset corpus ${default_dir}" >&2
      current="$default_dir"
      break
    fi
  done
  printf -v "$var_name" '%s' "$current"
}

run_graph_visuals_helper() {
  local helper_path
  helper_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-pretrain-graph-visuals.sh"
  if [[ ! -f "$helper_path" ]]; then
    helper_path="scripts/ci/run-pretrain-graph-visuals.sh"
  fi
  if [[ ! -f "$helper_path" ]]; then
    echo "[pretrain] info: graph visuals helper missing at $helper_path" >&2
    return 1
  fi
  if bash "$helper_path"; then
    return 0
  fi
  return 1
}

create_graph_visuals_placeholder() {
  local reason="${1:-graph visuals helper failed}"
  local output_dir="${PRETRAIN_EXPERIMENT_ROOT:-}"
  if [[ -z "${output_dir}" ]]; then
    return 1
  fi

  output_dir="${output_dir%/}/graphs"
  if [[ -n "$output_dir" && "$output_dir" != "/" ]]; then
    rm -rf "$output_dir"
  fi
  mkdir -p "$output_dir"

  local summary_path="${output_dir}/summary.json"
  if [[ ! -f "$summary_path" ]]; then
    cat >"$summary_path" <<EOF
{
  "dataset_path": "${DATASET_DIR:-<unset>}",
  "fallback_reason": "${reason}",
  "output_dir": "${output_dir}",
  "num_graphs": 0,
  "num_rendered": 0,
  "loader": "placeholder",
  "fallback_forced": false,
  "rdkit_available": false,
  "rdkit_installed": false,
  "rdkit_import_error": "${reason}",
  "png_renderers": {"placeholder": 1}
}
EOF
  fi

  : >"${output_dir}/placeholder.png"
  cat >"${output_dir}/index.html" <<EOF
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Graph Visuals (placeholder)</title></head>
<body>
<p>Graph visualisation generation failed (${reason}); placeholder assets materialised by run-pretrain.sh.</p>
</body>
</html>
EOF
}

pretrain_graph_visuals_enabled() {
  local default="${1:-1}" path_sink="${2:-}"
  local cfg_path="${TRAIN_JEPA_CI:-}"

  if [[ -z "$cfg_path" && -n "${APP_DIR:-}" ]]; then
    cfg_path="${APP_DIR%/}/scripts/ci/train_jepa_ci.yml"
  fi
  if [[ -z "$cfg_path" ]]; then
    cfg_path="${SCRIPT_DIR}/train_jepa_ci.yml"
  fi
  if [[ -n "$path_sink" ]]; then
    printf -v "$path_sink" '%s' "$cfg_path"
  fi

  local override="${PRETRAIN_GENERATE_GRAPH_VISUALS:-${GENERATE_GRAPH_VISUALS:-}}"
  case "${override,,}" in
    1|true|yes|on)
      echo "1"
      return 0
      ;;
    0|false|no|off)
      echo "0"
      return 0
      ;;
  esac

  if [[ ! -f "$cfg_path" ]]; then
    echo "$default"
    return 0
  fi

  local py_bin=""
  if command -v python >/dev/null 2>&1; then
    py_bin="python"
  elif command -v python3 >/dev/null 2>&1; then
    py_bin="python3"
  fi

  if [[ -z "$py_bin" ]]; then
    echo "$default"
    return 0
  fi

  local parsed
  parsed="$("$py_bin" - "$cfg_path" <<'PY'
import os
import sys

import yaml

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}

pretrain = data.get("pretrain") or {}
value = pretrain.get("generate_graph_visuals", pretrain.get("generate_graphs"))

if isinstance(value, str):
    text = value.strip()
    if text.startswith("${") and text.endswith("}"):
        value = os.path.expandvars(text)


def normalise(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return bool(val)
    text = str(val).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


result = normalise(value)
if result is None:
    sys.exit(1)

print("1" if result else "0")
PY
)"
  local status=$?
  if [[ $status -eq 0 && "$parsed" =~ ^[01]$ ]]; then
    echo "$parsed"
  else
    echo "$default"
  fi
}

GRAPH_VISUALS_CONFIG_PATH=""
GRAPH_VISUALS_ENABLED="$(pretrain_graph_visuals_enabled 0 GRAPH_VISUALS_CONFIG_PATH)"

ci_pretrain_materialize_manifest() {
  local stage_outputs="$1"
  local expected_manifest="$2"
  local expected_encoder="$3"
  local artifacts_dir="${4:-}"
  local pretrain_dir="${5:-}"
  local experiment_dir="${6:-}"
  local experiments_root="${7:-}"
  local pretrain_experiment_root="${8:-}"

  local recorded_manifest="" recorded_encoder=""
  local python_cmd=()

  resolve_ci_python python_cmd

  if [[ -f "$stage_outputs" && ${#python_cmd[@]} -gt 0 ]]; then
    while IFS='=' read -r key value; do
      case "$key" in
        manifest_path)
          recorded_manifest="$value"
          ;;
        encoder_checkpoint)
          recorded_encoder="$value"
          ;;
      esac
    done < <("${python_cmd[@]}" - "$stage_outputs" <<'PY'
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
  fi

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
    if normalized=$(normalize_stage_path "$recorded_manifest" "$artifacts_dir" "$pretrain_dir" "$experiment_dir" \
      "$pretrain_experiment_root" "$experiments_root"); then
      recorded_manifest="$normalized"
    fi
  fi

  if [[ -n "${recorded_encoder:-}" ]]; then
    if normalized=$(normalize_stage_path "$recorded_encoder" "$pretrain_dir" "$artifacts_dir" "$experiment_dir" \
      "$pretrain_experiment_root" "$experiments_root"); then
      recorded_encoder="$normalized"
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
    local encoder_for_manifest="$expected_encoder"
    if [[ -n "${recorded_encoder:-}" && -f "$recorded_encoder" ]]; then
      encoder_for_manifest="$recorded_encoder"
    fi
    mkdir -p "$(dirname "$expected_manifest")"
    if [[ ${#python_cmd[@]} -gt 0 ]]; then
      "${python_cmd[@]}" - "$expected_manifest" "$encoder_for_manifest" "${EXP_ID:-}" <<'PY'
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

  [[ -f "$expected_manifest" ]]
}

export MJEPACI_STAGE="pretrain"

if [[ -n "${MJEPACI_STAGE_SHIM:-}" && -x "${MJEPACI_STAGE_SHIM}" ]]; then
  # When exercising the CI flow through a stage shim (used by unit tests) we
  # still rely on helpers such as ``resolve_ci_python`` from common.sh.  Source
  # it explicitly here because the early-return path below would otherwise skip
  # the later ``source" statements, leading to ``command not found`` failures
  # (exit 127) when the helpers are invoked.
  source "$(dirname "$0")/common.sh"

  DEFAULT_DATASET_DIR="${APP_DIR:-/srv/mjepa}/data/ZINC-canonicalized"
  : "${DATASET_DIR:=${DEFAULT_DATASET_DIR}}"
  normalize_dataset_dir DATASET_DIR "$DEFAULT_DATASET_DIR"
  export DATASET_DIR

  if [[ -z "${EXP_ID:-}" || -z "${EXPERIMENTS_ROOT:-}" ]]; then
    echo "[ci] error: MJEPACI_STAGE_SHIM requires EXP_ID and EXPERIMENTS_ROOT" >&2
    exit 1
  fi

  EXPERIMENT_DIR="${EXPERIMENTS_ROOT%/}/${EXP_ID}"
  PRETRAIN_DIR="${EXPERIMENT_DIR}/pretrain"
  ARTIFACTS_DIR="${EXPERIMENT_DIR}/artifacts"
  PRETRAIN_ARTIFACTS_DIR="${ARTIFACTS_DIR}"
  PRETRAIN_EXPERIMENT_ROOT="${EXPERIMENT_DIR}"

  export EXP_ID EXPERIMENTS_ROOT EXPERIMENT_DIR PRETRAIN_DIR ARTIFACTS_DIR PRETRAIN_ARTIFACTS_DIR PRETRAIN_EXPERIMENT_ROOT

  STAGE_OUTPUTS_DIR="${PRETRAIN_DIR}/stage-outputs"
  export STAGE_OUTPUTS_DIR
  mkdir -p "${PRETRAIN_DIR}" "$STAGE_OUTPUTS_DIR" "${PRETRAIN_ARTIFACTS_DIR}"

  echo "[ci] (test) EXP_ID=${EXP_ID} EXPERIMENT_DIR=${EXPERIMENT_DIR} PRETRAIN_DIR=${PRETRAIN_DIR} PRETRAIN_ARTIFACTS_DIR=${PRETRAIN_ARTIFACTS_DIR} STAGE_BIN=${MJEPACI_STAGE_SHIM}" >&2

  "${MJEPACI_STAGE_SHIM}" pretrain

  expected_encoder="${PRETRAIN_DIR}/encoder.pt"
  expected_stage_outputs="${STAGE_OUTPUTS_DIR}/pretrain.json"
  expected_manifest="${PRETRAIN_ARTIFACTS_DIR}/encoder_manifest.json"

  for required in "$expected_encoder" "$expected_stage_outputs"; do
    if [[ ! -f "$required" ]]; then
      echo "[ci] error: expected ${required} not found" >&2
      exit 1
    fi
  done

  if ! ci_pretrain_materialize_manifest "$expected_stage_outputs" "$expected_manifest" "$expected_encoder" \
    "$PRETRAIN_ARTIFACTS_DIR" "$PRETRAIN_DIR" "$EXPERIMENT_DIR" "$EXPERIMENTS_ROOT" "${PRETRAIN_EXPERIMENT_ROOT:-}"; then
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

  if [[ "$GRAPH_VISUALS_ENABLED" == "0" ]]; then
    echo "[pretrain] graph visualisations disabled via config (${GRAPH_VISUALS_CONFIG_PATH:-<unset>}); seeding placeholders" >&2
    create_graph_visuals_placeholder "graph visuals disabled via config" || true
  elif ! run_graph_visuals_helper; then
    echo "::warning::graph visualisation generation failed" >&2
    create_graph_visuals_placeholder || true
  else
    graphs_dir="${PRETRAIN_EXPERIMENT_ROOT%/}/graphs"
    if [[ ! -d "$graphs_dir" || ! -f "${graphs_dir}/summary.json" ]]; then
      echo "::warning::graph visuals helper completed but outputs missing; seeding placeholders" >&2
      create_graph_visuals_placeholder || true
    fi
  fi

  ensure_tox21_gate_stub "$pretrain_root"

  exit 0
fi

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

DEFAULT_DATASET_DIR="${APP_DIR:-/srv/mjepa}/data/ZINC-canonicalized"
: "${DATASET_DIR:=${DEFAULT_DATASET_DIR}}"
normalize_dataset_dir DATASET_DIR "$DEFAULT_DATASET_DIR"
export DATASET_DIR

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

pretrain_hidden_dim=""
pretrain_num_layers=""
phase2_resolve_structural_defaults pretrain_hidden_dim pretrain_num_layers
if [[ -z "${PRETRAIN_HIDDEN_DIM:-}" ]]; then
  PRETRAIN_HIDDEN_DIM="$pretrain_hidden_dim"
fi
if [[ -z "${PRETRAIN_NUM_LAYERS:-}" ]]; then
  PRETRAIN_NUM_LAYERS="$pretrain_num_layers"
fi
export PRETRAIN_HIDDEN_DIM PRETRAIN_NUM_LAYERS

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

if ! ci_pretrain_materialize_manifest "$expected_stage_outputs" "$expected_manifest" "$expected_encoder" \
  "$PRETRAIN_ARTIFACTS_DIR" "$PRETRAIN_DIR" "$EXPERIMENT_DIR" "$EXPERIMENTS_ROOT" "${PRETRAIN_EXPERIMENT_ROOT:-}"; then
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

if [[ "$GRAPH_VISUALS_ENABLED" == "0" ]]; then
  echo "[pretrain] graph visualisations disabled via config (${GRAPH_VISUALS_CONFIG_PATH:-<unset>}); seeding placeholders" >&2
  create_graph_visuals_placeholder "graph visuals disabled via config" || true
elif ! run_graph_visuals_helper; then
  echo "::warning::graph visualisation generation failed" >&2
fi

ensure_tox21_gate_stub "$PRETRAIN_EXPERIMENT_ROOT"

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages
