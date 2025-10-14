#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config
export MJEPACI_STAGE="tox21"

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

export WANDB_NAME="tox21"
export WANDB_JOB_TYPE="tox21"

: "${GITHUB_ENV:=${PRETRAIN_TOX21_ENV}}"
mkdir -p "$(dirname "$GITHUB_ENV")"
: >"$GITHUB_ENV"
export GITHUB_ENV

SOURCE="${TOX21_ENCODER_SOURCE:-pretrain_frozen}"
MANIFEST_DEFAULT="${PRETRAIN_MANIFEST:-${PRETRAIN_ARTIFACTS_DIR}/encoder_manifest.json}"
MANIFEST_PATH="${TOX21_ENCODER_MANIFEST:-$MANIFEST_DEFAULT}"

echo "[tox21] using pretrain experiment id=${PRETRAIN_EXP_ID}" >&2
echo "[tox21] tox21 env path=${GITHUB_ENV}" >&2
echo "[tox21] encoder checkpoint source=${SOURCE}" >&2
echo "[tox21] manifest path=${MANIFEST_PATH}" >&2
echo "[tox21] state path=${PRETRAIN_STATE_FILE} (canonical=${PRETRAIN_STATE_FILE_CANONICAL:-n/a})" >&2
echo "[tox21] expected tox21 env seed=${PRETRAIN_TOX21_ENV}" >&2

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[ci] error: expected ${MANIFEST_PATH} but it was not found. Set PRETRAIN_EXP_ID=<id> to reuse an existing run or rerun pretrain." >&2
  exit 1
fi

python_cmd=()
if ensure_micromamba >/dev/null 2>&1; then
  python_cmd=("$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 python -u)
elif [[ -n "${MJEPACI_STAGE_SHIM:-}" ]]; then
  resolve_ci_python python_cmd
else
  ensure_micromamba
fi

MET_ENV_FILE="${PRETRAIN_EXPERIMENT_ROOT}/met_benchmark.env"
mkdir -p "$(dirname "$MET_ENV_FILE")"
if [[ "$SOURCE" == "fine_tuned" && -f "$MET_ENV_FILE" ]]; then
  while IFS='=' read -r key value; do
    [[ -z "$key" ]] && continue
    export "$key"="$value"
  done <"$MET_ENV_FILE"
  if [[ "${MET_BENCHMARK_BASELINE:-false}" == "true" ]]; then
    echo "[tox21] Baseline met benchmark; skipping fine-tuned evaluation."
    exit 0
  fi
fi

ensure_dir() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[ci] error: required file missing: $path" >&2
    exit 1
  fi
}

if [[ "$SOURCE" == "pretrain_frozen" ]]; then
  ensure_dir "$MANIFEST_PATH"
  TOX21_ENCODER_CHECKPOINT=$("${python_cmd[@]}" - "$MANIFEST_PATH" <<'PY'
import json, os, sys
manifest = json.load(open(sys.argv[1]))
paths = manifest.get("paths") if isinstance(manifest, dict) else {}
candidate = None
if isinstance(paths, dict):
    candidate = paths.get("encoder") or paths.get("encoder_symlink")
if candidate is None and isinstance(manifest, dict):
    candidate = manifest.get("encoder_checkpoint")
if candidate:
    print(os.path.abspath(candidate))
PY
  )
  TOX21_ENCODER_CHECKPOINT=${TOX21_ENCODER_CHECKPOINT:-}
  if [[ -z "$TOX21_ENCODER_CHECKPOINT" ]]; then
    echo "[tox21] could not extract encoder path from $MANIFEST_PATH" >&2
    exit 1
  fi
  export TOX21_ENCODER_CHECKPOINT
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
elif [[ "$SOURCE" == "fine_tuned" ]]; then
  TOX21_ENCODER_CHECKPOINT="${FINETUNE_DIR}/seed_0/ft_best.pt"
  export TOX21_ENCODER_CHECKPOINT
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
else
  TOX21_ENCODER_CHECKPOINT="${TOX21_ENCODER_CHECKPOINT:-$MANIFEST_PATH}"
  export TOX21_ENCODER_CHECKPOINT
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
fi

ensure_dir "$TOX21_ENCODER_CHECKPOINT"

env_file="$MET_ENV_FILE"
echo "[tox21] writing summary env to ${env_file}" >&2

export TOX21_ENCODER_SOURCE="$SOURCE"
export TOX21_ENCODER_CHECKPOINT

# ensure the parm matches train_jepa_ci.yml
SECONDS=0
"$STAGE_BIN" tox21
elapsed="${SECONDS}" 

stage_file="${TOX21_DIR}/stage-outputs/tox21_${SOURCE}.json"
"${python_cmd[@]}" - <<'PY' "$stage_file" "$SOURCE" "$env_file" "$elapsed"
import json, os, sys
stage_path, source, env_path, elapsed = sys.argv[1:5]
data = {}
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                data[key] = value
try:
    with open(stage_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
except Exception:
    payload = {}

def _coerce_bool(val):
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        return "true" if val.lower() in {"true", "1", "yes"} else "false"
    if isinstance(val, (int, float)):
        return "true" if val else "false"
    return None

met_flag = payload.get("met_benchmark")
if met_flag is None:
    met_flag = payload.get("met_benchmark_selected")
selected_path = payload.get("selected_path")

if source == "pretrain_frozen":
    if met_flag is not None:
        data["MET_BENCHMARK_BASELINE"] = _coerce_bool(met_flag) or "false"
    if selected_path:
        data["TOX21_SELECTED_BASELINE"] = str(selected_path)
else:
    if met_flag is not None:
        data["MET_BENCHMARK_FINAL"] = _coerce_bool(met_flag) or "false"
    if selected_path:
        data["TOX21_SELECTED_FINAL"] = str(selected_path)

with open(env_path, "w", encoding="utf-8") as fh:
    for key in sorted(data):
        fh.write(f"{key}={data[key]}\n")

diagnostics = {}
if isinstance(payload, dict):
    diagnostics = payload.get("diagnostics") or {}

def _resolve_batches(name):
    bucket = diagnostics.get("batch_counts") if isinstance(diagnostics, dict) else {}
    if not isinstance(bucket, dict):
        return None
    entry = bucket.get(name)
    if isinstance(entry, dict):
        return entry.get("batches")
    return None

encoder_path = diagnostics.get("encoder_checkpoint") if isinstance(diagnostics, dict) else None
if not encoder_path:
    encoder_path = payload.get("selected_path") if isinstance(payload, dict) else None
tasks = diagnostics.get("task_count") if isinstance(diagnostics, dict) else None
try:
    tasks = int(tasks) if tasks is not None else None
except Exception:
    tasks = None

try:
    molecules = int(diagnostics.get("num_molecules"))
except Exception:
    molecules = None

try:
    elapsed_val = float(elapsed)
except Exception:
    elapsed_val = None

summary_line = "[ci][info] tox21 summary: model={model} tasks={tasks} molecules={molecules} val_batches={val_batches} test_batches={test_batches} wall={wall}".format(
    model=encoder_path or "<unknown>",
    tasks=tasks if tasks is not None else "<unset>",
    molecules=molecules if molecules is not None else "<unset>",
    val_batches=_resolve_batches("val") or 0,
    test_batches=_resolve_batches("test") or 0,
    wall=(f"{elapsed_val:.1f}s" if isinstance(elapsed_val, float) else "<unknown>"),
)
print(summary_line)
PY

if [[ "${SOURCE}" == "pretrain_frozen" && -f "$env_file" && -n "${PRETRAIN_EXP_ID:-}" ]]; then
  if grep -Eq '^MET_BENCHMARK_BASELINE=true$' "$env_file"; then
    freeze_marker="${EXPERIMENTS_ROOT%/}/${PRETRAIN_EXP_ID}/bench/encoder_frozen.ok"
    mkdir -p "$(dirname "$freeze_marker")"
    if touch "$freeze_marker"; then
      echo "[ci] encoder lineage frozen at ${PRETRAIN_EXP_ID}" >&2
    else
      echo "[ci][warn] unable to write freeze marker at ${freeze_marker}" >&2
    fi
  fi
fi

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages
