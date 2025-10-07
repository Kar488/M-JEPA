#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config
export MJEPACI_STAGE="tox21"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

echo "[ci] EXP_ID=${EXP_ID}" >&2
echo "[ci] EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT}" >&2
echo "[ci] DATA_ROOT=${DATA_ROOT:-<unset>}" >&2
echo "[ci] EXPERIMENT_DIR=${EXPERIMENT_DIR}" >&2
echo "[ci] ARTIFACTS_DIR=${ARTIFACTS_DIR}" >&2
echo "[ci] PRETRAIN_ARTIFACTS_DIR=${PRETRAIN_ARTIFACTS_DIR}" >&2

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
  echo "[tox21] required manifest missing: $MANIFEST_PATH" >&2
  exit 1
fi

python_cmd=()
resolve_ci_python python_cmd

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
    echo "[tox21] required file missing: $path" >&2
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
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
elif [[ "$SOURCE" == "fine_tuned" ]]; then
  TOX21_ENCODER_CHECKPOINT="${FINETUNE_DIR}/seed_0/ft_best.pt"
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
else
  TOX21_ENCODER_CHECKPOINT="${TOX21_ENCODER_CHECKPOINT:-$MANIFEST_PATH}"
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
fi

ensure_dir "$TOX21_ENCODER_CHECKPOINT"

env_file="$MET_ENV_FILE"
echo "[tox21] writing summary env to ${env_file}" >&2

export TOX21_ENCODER_SOURCE="$SOURCE"
export TOX21_ENCODER_CHECKPOINT

# ensure the parm matches train_jepa_ci.yml
run_stage tox21

stage_file="${TOX21_DIR}/stage-outputs/tox21_${SOURCE}.json"
"${python_cmd[@]}" - <<'PY' "$stage_file" "$SOURCE" "$env_file"
import json, os, sys
stage_path, source, env_path = sys.argv[1:4]
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
PY

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages
