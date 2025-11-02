#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config
export MJEPACI_STAGE="tox21"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

if [[ ! -f "$APP_DIR/scripts/train_jepa.py" ]]; then
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
  if [[ -f "$repo_root/scripts/train_jepa.py" ]]; then
    APP_DIR="$repo_root"
    export APP_DIR
    export PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}"
  fi
fi

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

SOURCE="${TOX21_EVALUATION_MODE:-${TOX21_ENCODER_SOURCE:-pretrain_frozen}}"
# When the orchestrator finishes a non-frozen fine-tune it leaves
# ``FROZEN=0`` in the environment and drops the exported checkpoints into
# ``$FINETUNE_DIR``.  Historically the Tox21 stage still defaulted to the
# ``pretrain_frozen`` evaluation path in that situation, forcing the
# downstream case study to retrain a probe over a frozen backbone.  This
# produced the random-baseline metrics observed in recent runs.  Detect the
# fine-tuned lineage and automatically upgrade the mode to ``end_to_end``
# so the evaluation reuses the freshly tuned encoder unless the user has
# explicitly requested a different mode.
auto_full_finetune="0"
if [[ "$SOURCE" == "pretrain_frozen" ]]; then
  if [[ -z "${TOX21_EVALUATION_MODE:-}" && -z "${TOX21_ENCODER_SOURCE:-}" ]]; then
    if [[ "${FROZEN:-1}" != "1" ]]; then
      if [[ -n "${FINETUNE_DIR:-}" ]]; then
        if [[ -f "${FINETUNE_DIR}/encoder_ft.pt" || -f "${FINETUNE_DIR}/seed_0/ft_best.pt" ]]; then
          echo "[tox21] auto-detected fine-tuned encoder; switching evaluation mode to end_to_end" >&2
          SOURCE="end_to_end"
          auto_full_finetune="1"
        fi
      fi
    fi
  fi
fi
MANIFEST_DEFAULT="${PRETRAIN_MANIFEST:-${PRETRAIN_ARTIFACTS_DIR}/encoder_manifest.json}"
MANIFEST_PATH="${TOX21_ENCODER_MANIFEST:-$MANIFEST_DEFAULT}"

if [[ "$SOURCE" == "fine_tuned" || "$SOURCE" == "end_to_end" ]]; then
  if [[ -z "${FINETUNE_EPOCHS:-}" ]]; then
    FINETUNE_EPOCHS=18
  fi
  if [[ -z "${TOX21_FINETUNE_PATIENCE:-}" ]]; then
    TOX21_FINETUNE_PATIENCE=10
  fi
  export FINETUNE_EPOCHS
  export TOX21_FINETUNE_PATIENCE
fi

if [[ "$SOURCE" == "fine_tuned" || "$SOURCE" == "end_to_end" ]]; then
  full_flag="${TOX21_FULL_FINETUNE:-}"
  if [[ -z "$full_flag" ]]; then
    TOX21_FULL_FINETUNE="true"
    export TOX21_FULL_FINETUNE
    if [[ "$auto_full_finetune" == "1" ]]; then
      echo "[tox21] auto-enabling full fine-tuning for end_to_end evaluation" >&2
    fi
  else
    export TOX21_FULL_FINETUNE
  fi
elif [[ -n "${TOX21_FULL_FINETUNE:-}" ]]; then
  export TOX21_FULL_FINETUNE
fi

echo "[tox21] using pretrain experiment id=${PRETRAIN_EXP_ID}" >&2
echo "[tox21] tox21 env path=${GITHUB_ENV}" >&2
echo "[tox21] encoder checkpoint source=${SOURCE}" >&2
echo "[tox21] manifest path=${MANIFEST_PATH}" >&2
echo "[tox21] state path=${PRETRAIN_STATE_FILE} (canonical=${PRETRAIN_STATE_FILE_CANONICAL:-n/a})" >&2
echo "[tox21] expected tox21 env seed=${PRETRAIN_TOX21_ENV}" >&2

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[diag] about to exit: missing manifest (MANIFEST_PATH=${MANIFEST_PATH})" >&2
  echo "[ci] error: expected ${MANIFEST_PATH} but it was not found. Set PRETRAIN_EXP_ID=<id> to reuse an existing run or rerun pretrain." >&2
  exit 1
fi

print_python_cmd() {
  local target="${1:-python_cmd}"
  local -n arr="$target"
  local joined=""
  if (( ${#arr[@]} )); then
    printf -v joined '%q ' "${arr[@]}"
    joined=${joined% }
  fi
  echo "[diag] ${target}=${joined}"
}

repair_micromamba_env() {
  if [[ ! -x "${APP_DIR}/scripts/ci/prepare_env.sh" ]]; then
    echo "[tox21] warn: scripts/ci/prepare_env.sh missing; cannot repair micromamba env" >&2
    return 1
  fi
  if APP_DIR="$APP_DIR" MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}" \
    bash "${APP_DIR}/scripts/ci/prepare_env.sh"; then
    return 0
  fi
  local status=$?
  echo "[tox21] warn: prepare_env.sh failed to repair micromamba env (status ${status})" >&2
  return "$status"
}

python_interp_cmd=()
python_cmd=()

unset MJEPACI_FORCE_SYSTEM_PYTHON MJEPACI_SYSTEM_PYTHON_BIN

python_mamba_ready=0
if ensure_micromamba; then
  python_interp_cmd=("$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 python -u)
  python_mamba_ready=1
  print_python_cmd python_interp_cmd
  if ! "${python_interp_cmd[@]}" - <<'PY'
import sys
sys.exit(0)
PY
  then
    echo "[tox21] warn: micromamba env 'mjepa' missing python; attempting repair" >&2
    if repair_micromamba_env; then
      python_interp_cmd=("$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 python -u)
      print_python_cmd python_interp_cmd
      if "${python_interp_cmd[@]}" - <<'PY'
import sys
sys.exit(0)
PY
      then
        python_mamba_ready=1
      else
        python_mamba_ready=0
        echo "[tox21] warn: micromamba python still unavailable after repair" >&2
        python_interp_cmd=()
      fi
    else
      python_mamba_ready=0
      python_interp_cmd=()
    fi
  else
    python_mamba_ready=1
  fi
else
  echo "[tox21] warn: micromamba bootstrap unavailable; falling back to system python" >&2
fi

if (( python_mamba_ready )) && [[ -z "${MJEPACI_STAGE_SHIM:-}" ]]; then
  if ! "${python_interp_cmd[@]}" - <<'PY'
try:
    import yaml  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    echo "[tox21] warn: micromamba env missing PyYAML; attempting repair" >&2
    if repair_micromamba_env; then
      python_interp_cmd=("$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 python -u)
      print_python_cmd python_interp_cmd
      if "${python_interp_cmd[@]}" - <<'PY'
try:
    import yaml  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
      then
        :
      else
        echo "[tox21] warn: micromamba python still missing PyYAML after repair" >&2
        python_interp_cmd=()
      fi
    else
      python_interp_cmd=()
    fi
  fi
fi

if (( ${#python_interp_cmd[@]} == 0 )); then
  if py=$(python_bin 2>/dev/null); then
    py_path="$(command -v "$py" 2>/dev/null || true)"
    if [[ -z "$py_path" ]]; then
      py_path="$py"
    fi
    python_interp_cmd=(env PYTHONUNBUFFERED=1 "$py_path" -u)
    MJEPACI_FORCE_SYSTEM_PYTHON=1
    MJEPACI_SYSTEM_PYTHON_BIN="$py_path"
    export MJEPACI_FORCE_SYSTEM_PYTHON MJEPACI_SYSTEM_PYTHON_BIN
    echo "[tox21] warn: using system python interpreter (${py_path})" >&2
  elif [[ -n "${MJEPACI_STAGE_SHIM:-}" ]]; then
    resolve_ci_python python_cmd
    if (( ${#python_cmd[@]} )); then
      python_interp_cmd=("${python_cmd[@]}")
      python_cmd=()
    fi
  fi
fi

if (( ${#python_interp_cmd[@]} == 0 )); then
  echo "[diag] about to exit: python interpreter unavailable (python_interp_cmd=<empty>)" >&2
  echo "[tox21] error: unable to resolve a python interpreter" >&2
  exit 1
fi

echo "[diag] PY=$(command -v python || true)"

orig_encoder_override="${TOX21_ENCODER_CHECKPOINT:-}"
encoder_decision_source="unknown"

extract_finetune_export() {
  local json_path="$1"
  [[ -f "$json_path" ]] || return 1
  local result
  local status=0
  set +e
  print_python_cmd python_interp_cmd
  result=$("${python_interp_cmd[@]}" - "$json_path" <<'PY'
import json
import os
import sys

try:
    payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception:
    sys.exit(1)

entry = payload.get("encoder_finetuned") if isinstance(payload, dict) else None
if isinstance(entry, dict):
    path = entry.get("checkpoint")
    if path:
        print(os.path.abspath(path))
PY
  )
  status=$?
  set -e
  if (( status != 0 )); then
    return 1
  fi
  result=${result:-}
  [[ -n "$result" ]] || return 1
  printf '%s\n' "$result"
  return 0
}

select_encoder_candidate() {
  local -n out_path=$1
  local -n out_label=$2
  shift 2
  local last_path="" last_label=""
  out_path=""
  out_label=""
  while (( $# >= 2 )); do
    local label="$1"; shift
    local path="$1"; shift
    [[ -n "$path" ]] || continue
    last_path="$path"
    last_label="$label"
    if [[ -f "$path" ]]; then
      out_path="$path"
      out_label="$label"
      return 0
    fi
    echo "[tox21] candidate missing (${label}): $path" >&2
  done
  out_path="$last_path"
  out_label="${last_label:-unknown}"
  return 1
}

MET_ENV_FILE="${PRETRAIN_EXPERIMENT_ROOT}/met_benchmark.env"
mkdir -p "$(dirname "$MET_ENV_FILE")"
if [[ "$SOURCE" == "fine_tuned" || "$SOURCE" == "end_to_end" ]] && [[ -f "$MET_ENV_FILE" ]]; then
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
    echo "[diag] about to exit: required path missing (path=${path})" >&2
    echo "[ci] error: required file missing: $path" >&2
    exit 1
  fi
}

if [[ "$SOURCE" == "pretrain_frozen" ]]; then
  ensure_dir "$MANIFEST_PATH"
  manifest_encoder=""
  print_python_cmd python_interp_cmd
  manifest_encoder=$("${python_interp_cmd[@]}" - "$MANIFEST_PATH" <<'PY'
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
  encoder_candidates=()
  encoder_candidates+=("manifest" "${manifest_encoder:-}")
  if [[ -n "${PRETRAIN_DIR:-}" ]]; then
    encoder_candidates+=("pretrain_dir" "${PRETRAIN_DIR%/}/encoder.pt")
  fi
  if [[ -n "${PRETRAIN_ARTIFACTS_DIR:-}" ]]; then
    encoder_candidates+=("artifacts" "${PRETRAIN_ARTIFACTS_DIR%/}/encoder.pt")
  fi

  resolved_path=""
  resolved_label=""
  if ! select_encoder_candidate resolved_path resolved_label "${encoder_candidates[@]}"; then
    select_status=$?
  else
    select_status=0
  fi

  TOX21_ENCODER_CHECKPOINT="$resolved_path"
  if [[ -z "${resolved_label:-}" ]]; then
    resolved_label="manifest"
  fi
  encoder_decision_source="$resolved_label"

  if (( select_status )); then
    if [[ -n "$TOX21_ENCODER_CHECKPOINT" ]]; then
      echo "[tox21] falling back to ${encoder_decision_source}: ${TOX21_ENCODER_CHECKPOINT}" >&2
    else
      echo "[tox21] warning: encoder candidates unavailable from manifest ${MANIFEST_PATH}" >&2
    fi
  fi

  encoder_hint_parts=()
  i=0
  while (( i < ${#encoder_candidates[@]} )); do
    label="${encoder_candidates[i]}"
    path="${encoder_candidates[i+1]}"
    encoder_hint_parts+=("${label}=${path:-<unset>}")
    ((i+=2))
  done

  if [[ -z "$TOX21_ENCODER_CHECKPOINT" || ! -e "$TOX21_ENCODER_CHECKPOINT" ]]; then
    echo "[diag] about to exit: encoder checkpoint missing (SOURCE=${SOURCE} resolved=${TOX21_ENCODER_CHECKPOINT:-<unset>} candidates=${encoder_hint_parts[*]})" >&2
    echo "[tox21] error: encoder checkpoint not found. Checked candidates: ${encoder_hint_parts[*]}" >&2
    exit 1
  fi

  export TOX21_ENCODER_CHECKPOINT
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
elif [[ "$SOURCE" == "frozen_finetuned" ]]; then
  stage_json="${FINETUNE_DIR}/stage-outputs/finetune.json"
  ft_export_path=""
  if ! ft_export_path=$(extract_finetune_export "$stage_json"); then
    if [[ -f "$stage_json" ]]; then
      echo "[tox21] warning: encoder_finetuned checkpoint not recorded in ${stage_json}" >&2
    else
      echo "[tox21] warning: expected ${stage_json} for frozen_finetuned mode" >&2
    fi
    ft_export_path=""
  fi
  resolved_path=""
  resolved_label=""
  if ! select_encoder_candidate resolved_path resolved_label \
    finetune_export "$ft_export_path" \
    explicit_override "$orig_encoder_override" \
    encoder_ft "${FINETUNE_DIR}/encoder_ft.pt" \
    seed_best "${FINETUNE_DIR}/seed_0/ft_best.pt"; then
    select_status=$?
  else
    select_status=0
  fi
  TOX21_ENCODER_CHECKPOINT="$resolved_path"
  encoder_decision_source="$resolved_label"
  if (( select_status )); then
    echo "[tox21] warning: falling back to ${encoder_decision_source}: ${TOX21_ENCODER_CHECKPOINT}" >&2
  fi
  export TOX21_ENCODER_CHECKPOINT
  # Preserve the original pretrain manifest so downstream evaluation can
  # reconstruct the encoder configuration when using frozen fine-tuned
  # checkpoints. The finetune bookkeeping file lacks the required
  # hyperparameters section.
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
elif [[ "$SOURCE" == "fine_tuned" || "$SOURCE" == "end_to_end" ]]; then
  stage_json="${FINETUNE_DIR}/stage-outputs/finetune.json"
  ft_export_path=""
  if ! ft_export_path=$(extract_finetune_export "$stage_json"); then
    if [[ -f "$stage_json" ]]; then
      echo "[tox21] warning: encoder_finetuned checkpoint not recorded in ${stage_json}" >&2
    else
      echo "[tox21] warning: expected ${stage_json} for ${SOURCE} mode" >&2
    fi
    ft_export_path=""
  fi
  resolved_path=""
  resolved_label=""
  if ! select_encoder_candidate resolved_path resolved_label \
    finetune_export "$ft_export_path" \
    explicit_override "$orig_encoder_override" \
    seed_best "${FINETUNE_DIR}/seed_0/ft_best.pt" \
    encoder_ft "${FINETUNE_DIR}/encoder_ft.pt"; then
    select_status=$?
  else
    select_status=0
  fi
  TOX21_ENCODER_CHECKPOINT="$resolved_path"
  encoder_decision_source="$resolved_label"
  if (( select_status )); then
    echo "[tox21] warning: falling back to ${encoder_decision_source}: ${TOX21_ENCODER_CHECKPOINT}" >&2
  fi
  export TOX21_ENCODER_CHECKPOINT
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
else
  TOX21_ENCODER_CHECKPOINT="${TOX21_ENCODER_CHECKPOINT:-$MANIFEST_PATH}"
  export TOX21_ENCODER_CHECKPOINT
  export TOX21_ENCODER_MANIFEST="$MANIFEST_PATH"
  if [[ -n "$orig_encoder_override" ]]; then
    encoder_decision_source="explicit_override"
  else
    encoder_decision_source="manifest"
  fi
fi

if [[ -z "${encoder_decision_source:-}" ]]; then
  encoder_decision_source="manifest"
fi

if [[ -z "${TOX21_ENCODER_CHECKPOINT:-}" ]]; then
  echo "[diag] about to exit: encoder checkpoint empty (SOURCE=${SOURCE} encoder_decision_source=${encoder_decision_source:-<unset>})" >&2
  echo "[tox21] error: encoder checkpoint path resolved to empty string" >&2
  exit 1
fi

echo "[tox21] eval: using encoder_checkpoint=${TOX21_ENCODER_CHECKPOINT} (mode=${SOURCE} origin=${encoder_decision_source})" >&2

ensure_dir "$TOX21_ENCODER_CHECKPOINT"

: "${CI_DIAG:=1}"
export CI_DIAG

tox21_stage_args=()
if [[ -n "${MJEPACI_STAGE_SHIM:-}" ]]; then
  echo "[tox21] warning: stage shim active; skipping tox21 argument construction" >&2
  python_cmd=("${python_interp_cmd[@]}" -m scripts.commands.tox21 --stage-shim)
else
  if build_stage_args "tox21"; then
    tox21_stage_args=("${STAGE_ARGS[@]}")
    if (( ${#tox21_stage_args[@]} == 0 )); then
      echo "[diag] about to exit: tox21 argument vector empty after build_stage_args" >&2
      echo "[tox21] error: tox21 argument vector resolved to empty list" >&2
      exit 1
    fi
    python_cmd=("${python_interp_cmd[@]}" -m scripts.commands.tox21 "${tox21_stage_args[@]}")
  else
    build_status=$?
    echo "[diag] about to exit: build_stage_args failed for tox21 (status=${build_status})" >&2
    echo "[tox21] error: unable to construct tox21 argument vector" >&2
    exit "$build_status"
  fi
fi

if (( ${#python_cmd[@]} == 0 )); then
  echo "[diag] about to exit: python_cmd missing after tox21 arg resolution" >&2
  echo "[tox21] error: tox21 python command unavailable" >&2
  exit 1
fi

print_python_cmd python_cmd

env_file="$MET_ENV_FILE"
echo "[tox21] writing summary env to ${env_file}" >&2

export TOX21_ENCODER_SOURCE="$SOURCE"
export TOX21_EVALUATION_MODE="$SOURCE"
export TOX21_ENCODER_CHECKPOINT

# ensure the parm matches train_jepa_ci.yml
SECONDS=0
"$STAGE_BIN" tox21
elapsed="${SECONDS}" 

stage_file="${TOX21_DIR}/stage-outputs/tox21_${SOURCE}.json"
print_python_cmd python_interp_cmd
"${python_interp_cmd[@]}" - <<'PY' "$stage_file" "$SOURCE" "$env_file" "$elapsed"
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

tasks_block = {}
if isinstance(payload, dict):
    tasks_block = payload.get("tasks") or {}

def _first_task_value(key):
    if isinstance(tasks_block, dict):
        for info in tasks_block.values():
            if isinstance(info, dict) and key in info:
                return info.get(key)
    return None

met_flag = payload.get("met_benchmark")
if met_flag is None:
    met_flag = payload.get("tox21_gate_passed")
selected_path = _first_task_value("selected_path")

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
    encoder_path = _first_task_value("stage_path")
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
