#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

unset BESTCFG_NO_EPOCHS
export MJEPACI_STAGE="finetune"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

# Some CI tests only provide EXPERIMENTS_ROOT and PRETRAIN_EXP_ID. Derive the
# lineage directories early so resolve_encoder_checkpoint sees the same defaults
# as downstream CI jobs and so tests that stub encoder.pt files behave like the
# real pipeline.
if [[ -n "${PRETRAIN_EXP_ID:-}" && -n "${EXPERIMENTS_ROOT:-}" ]]; then
  : "${PRETRAIN_EXPERIMENT_ROOT:=${EXPERIMENTS_ROOT%/}/${PRETRAIN_EXP_ID}}"
  : "${PRETRAIN_DIR:=${PRETRAIN_EXPERIMENT_ROOT%/}/pretrain}"
  : "${PRETRAIN_ARTIFACTS_DIR:=${PRETRAIN_EXPERIMENT_ROOT%/}/artifacts}"
  : "${PRETRAIN_MANIFEST:=${PRETRAIN_ARTIFACTS_DIR%/}/encoder_manifest.json}"
  : "${PRETRAIN_ENCODER_PATH:=${PRETRAIN_DIR%/}/encoder.pt}"
fi

ci_print_env_diag

: "${LINEAR_HEAD_SOFT_TIMEOUT_EXIT:=86}"

if [[ -n "${BESTCFG_SKIP:-}" ]]; then
  BESTCFG_SKIP="${BESTCFG_SKIP} max_pretrain_batches max_finetune_batches task_type metric"
else
  BESTCFG_SKIP="max_pretrain_batches max_finetune_batches task_type metric"
fi
export BESTCFG_SKIP

export WANDB_NAME="finetune"
export WANDB_JOB_TYPE="finetune"

# Skip fine-tuning if baseline evaluation already met benchmark
MET_ENV_FILE=""
local_met_env=""
if [[ -n "${EXP_ROOT:-}" ]]; then
  local_met_env="${EXP_ROOT%/}/met_benchmark.env"
elif [[ -n "${EXPERIMENT_DIR:-}" ]]; then
  local_met_env="${EXPERIMENT_DIR%/}/met_benchmark.env"
elif [[ -n "${FINETUNE_DIR:-}" ]]; then
  local_met_env="${FINETUNE_DIR%/}/met_benchmark.env"
fi

pretrain_met_env=""
if [[ -n "${PRETRAIN_EXPERIMENT_ROOT:-}" ]]; then
  pretrain_met_env="${PRETRAIN_EXPERIMENT_ROOT%/}/met_benchmark.env"
fi

pretrain_dir_gate=""
if [[ -n "${PRETRAIN_DIR:-}" ]]; then
  pretrain_dir_gate="$(dirname "${PRETRAIN_DIR%/}")/met_benchmark.env"
fi

pretrain_artifacts_gate=""
if [[ -n "${PRETRAIN_ARTIFACTS_DIR:-}" ]]; then
  pretrain_artifacts_gate="$(dirname "${PRETRAIN_ARTIFACTS_DIR%/}")/met_benchmark.env"
fi

# Avoid duplicate logging when derived paths collapse onto the lineage root.
if [[ -n "$pretrain_met_env" && -n "$pretrain_dir_gate" && "$pretrain_met_env" == "$pretrain_dir_gate" ]]; then
  pretrain_dir_gate=""
fi
if [[ -n "$pretrain_met_env" && -n "$pretrain_artifacts_gate" && "$pretrain_met_env" == "$pretrain_artifacts_gate" ]]; then
  pretrain_artifacts_gate=""
fi
if [[ -n "$pretrain_dir_gate" && -n "$pretrain_artifacts_gate" && "$pretrain_dir_gate" == "$pretrain_artifacts_gate" ]]; then
  pretrain_artifacts_gate=""
fi

echo "[finetune][gate] env roots: EXP_ROOT=${EXP_ROOT:-<unset>} EXPERIMENT_DIR=${EXPERIMENT_DIR:-<unset>} FINETUNE_DIR=${FINETUNE_DIR:-<unset>} PRETRAIN_EXPERIMENT_ROOT=${PRETRAIN_EXPERIMENT_ROOT:-<unset>}" >&2

if [[ -n "$local_met_env" ]]; then
  if [[ -f "$local_met_env" ]]; then
    echo "[finetune][gate] candidate local gate path=$local_met_env (present)" >&2
  else
    echo "[finetune][gate] candidate local gate path=$local_met_env (missing)" >&2
  fi
else
  echo "[finetune][gate] candidate local gate path=<unset>" >&2
fi

log_pretrain_candidate() {
  local label="$1" path="$2"
  if [[ -z "$path" ]]; then
    echo "[finetune][gate] candidate ${label} gate path=<unset>" >&2
    return
  fi
  if [[ -f "$path" ]]; then
    echo "[finetune][gate] candidate ${label} gate path=$path (present)" >&2
  else
    echo "[finetune][gate] candidate ${label} gate path=$path (missing)" >&2
  fi
}

log_pretrain_candidate "pretrain" "$pretrain_met_env"
log_pretrain_candidate "pretrain (derived from PRETRAIN_DIR)" "$pretrain_dir_gate"
log_pretrain_candidate "pretrain (derived from PRETRAIN_ARTIFACTS_DIR)" "$pretrain_artifacts_gate"

declare -a gate_candidates=()
if [[ -n "$local_met_env" ]]; then
  gate_candidates+=("$local_met_env")
fi
for candidate in "$pretrain_met_env" "$pretrain_dir_gate" "$pretrain_artifacts_gate"; do
  if [[ -n "$candidate" ]]; then
    gate_candidates+=("$candidate")
  fi
done

for candidate in "${gate_candidates[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    MET_ENV_FILE="$candidate"
    break
  fi
done

if [[ -z "$MET_ENV_FILE" ]]; then
  for candidate in "${gate_candidates[@]}"; do
    if [[ -n "$candidate" ]]; then
      MET_ENV_FILE="$candidate"
      break
    fi
  done
fi

if [[ -n "$MET_ENV_FILE" ]]; then
  if [[ -f "$MET_ENV_FILE" ]]; then
    echo "[finetune][gate] selected gate file=$MET_ENV_FILE (readable)" >&2
  else
    echo "[finetune][gate] selected gate file=$MET_ENV_FILE (not found)" >&2
  fi
else
  echo "[finetune][gate] selected gate file=<unset>" >&2
fi

if [[ -n "$MET_ENV_FILE" && -f "$MET_ENV_FILE" ]]; then
  raw_line=""
  while IFS= read -r raw_line || [[ -n "${raw_line:-}" ]]; do
    line="${raw_line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    [[ "$line" == '#'* ]] && continue
    case "${line,,}" in
      export*)
        line="${line#export}"
        line="${line#"${line%%[![:space:]]*}"}"
        ;;
    esac
    if [[ "$line" != *'='* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    [[ -z "$key" ]] && continue
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "[finetune] ignoring invalid gate entry: ${raw_line}" >&2
      continue
    fi
    export "$key"="$value"
  done <"$MET_ENV_FILE"
fi

# When the gate result is unavailable the reroute logic should no-op.
# Ensure the flag carries an explicit "unknown" marker instead of
# inheriting the "false" default from parameter expansion.
baseline_status="unknown"
if [[ -z "${MET_BENCHMARK_BASELINE+x}" ]]; then
  baseline_status="unknown"
else
  # Treat blank or whitespace-only values as unknown to avoid false negatives.
  baseline_trimmed="${MET_BENCHMARK_BASELINE//[[:space:]]/}"
  if [[ -z "$baseline_trimmed" ]]; then
    baseline_status="unknown"
  else
    baseline_lower="${baseline_trimmed,,}"
    case "$baseline_lower" in
      true|false)
        baseline_status="$baseline_lower"
        ;;
      *)
        baseline_status="unknown"
        ;;
    esac
  fi
fi
export MET_BENCHMARK_BASELINE="$baseline_status"

echo "[finetune][gate] normalized MET_BENCHMARK_BASELINE=${baseline_status}" >&2

if [[ "$baseline_status" == "true" ]]; then
  echo "[finetune] Baseline met benchmark; skipping fine-tune stage."
  exit 0
fi

manifest_path="${PRETRAIN_MANIFEST}"
encoder_ckpt="$(resolve_encoder_checkpoint)"

echo "[finetune] using pretrain experiment id=${PRETRAIN_EXP_ID} checkpoint=${encoder_ckpt}" >&2
echo "[finetune] encoder manifest=${manifest_path}" >&2

if [[ -z "$encoder_ckpt" ]]; then
  echo "[ci] error: missing pretrain checkpoint for finetune (PRETRAIN_ENCODER_PATH=${PRETRAIN_ENCODER_PATH:-<unset>}). Set PRETRAIN_EXP_ID=<id> to reuse an existing run or rerun pretrain." >&2
  exit 1
fi

if [[ -n "${PRETRAIN_ENCODER_PATH:-}" && "$encoder_ckpt" != "${PRETRAIN_ENCODER_PATH:-}" ]]; then
  echo "[finetune] PRETRAIN_ENCODER_PATH pointed to ${PRETRAIN_ENCODER_PATH}; using manifest-derived path ${encoder_ckpt}" >&2
fi

if [[ ! -f "$encoder_ckpt" ]]; then
  echo "[ci] error: expected ${encoder_ckpt} but it was not found. Set PRETRAIN_EXP_ID=<id> to reuse an existing run or rerun pretrain." >&2
  exit 1
fi

if [[ ! -f "$manifest_path" ]]; then
  echo "[ci] error: expected ${manifest_path} but it was not found. Set PRETRAIN_EXP_ID=<id> to reuse an existing run or rerun pretrain." >&2
  exit 1
fi

if [[ "$baseline_status" == "false" ]]; then
  : "${FINETUNE_LABELED_CSV:=${APP_DIR}/data/tox21/data.csv}"
  if [[ -z "${FINETUNE_LABELED_DIR:-}" ]]; then
    FINETUNE_LABELED_DIR="$(dirname "${FINETUNE_LABELED_CSV}")"
  elif [[ -f "${FINETUNE_LABELED_DIR}" ]]; then
    FINETUNE_LABELED_CSV="${FINETUNE_LABELED_CSV:-${FINETUNE_LABELED_DIR}}"
    FINETUNE_LABELED_DIR="$(dirname "${FINETUNE_LABELED_DIR}")"
  fi
  default_tasks=(
    "NR-AR" "NR-AR-LBD" "NR-AhR" "NR-Aromatase"
    "NR-ER" "NR-ER-LBD" "NR-PPAR-gamma" "SR-ARE"
    "SR-ATAD5" "SR-HSE" "SR-MMP" "SR-p53"
  )
  if [[ -n "${TOX21_FINE_TUNE_TASK:-}" ]]; then
    default_tasks=("${TOX21_FINE_TUNE_TASK}")
  fi
  IFS=',' read -r -a explicit_list <<<"${FINETUNE_LABEL_COLS:-}"
  if [[ ${#explicit_list[@]} -gt 0 ]]; then
    tasks=("${explicit_list[@]}")
  else
    tasks=("${default_tasks[@]}")
  fi
  if [[ -n "${FINETUNE_LABEL_COL:-}" ]]; then
    tasks=("${FINETUNE_LABEL_COL}")
  fi
  if [[ ${#tasks[@]} -eq 0 ]]; then
    tasks=("NR-AR")
  fi
  FINETUNE_LABEL_COLS="$(printf '%s,' "${tasks[@]}")"
  FINETUNE_LABEL_COLS="${FINETUNE_LABEL_COLS%,}"
  export FINETUNE_LABEL_COLS
  FINETUNE_LABEL_COL="${tasks[0]}"
  : "${FINETUNE_TASK_TYPE:=classification}"
  : "${FINETUNE_METRIC:=val_auc}"
  : "${FINETUNE_USE_SCAFFOLD:=true}"
  : "${FINETUNE_SEED_0:=0}"
  if [[ -z ${FINETUNE_SEED_1+x} ]]; then
    FINETUNE_SEED_1=1
  fi
  if [[ -z ${FINETUNE_SEED_2+x} ]]; then
    FINETUNE_SEED_2=2
  fi
  : "${FINETUNE_HIDDEN_DIM:=384}"
  : "${FINETUNE_NUM_LAYERS:=4}"
  : "${FINETUNE_DROPOUT:=0.15}"
  : "${FINETUNE_DATASET_OVERRIDE_REASON:=tox21_gate_failure}"

  export FINETUNE_LABELED_CSV
  export FINETUNE_LABELED_DIR
  export FINETUNE_LABEL_COL
  export FINETUNE_TASK_TYPE
  export FINETUNE_METRIC
  export FINETUNE_USE_SCAFFOLD

  export FINETUNE_SEED_0
  if [[ -n "${FINETUNE_SEED_1}" ]]; then
    export FINETUNE_SEED_1
  else
    unset FINETUNE_SEED_1
  fi
  if [[ -n "${FINETUNE_SEED_2}" ]]; then
    export FINETUNE_SEED_2
  else
    unset FINETUNE_SEED_2
  fi

  export FINETUNE_HIDDEN_DIM
  export FINETUNE_NUM_LAYERS
  export FINETUNE_DROPOUT

  export FINETUNE_DATASET_OVERRIDE_REASON

  echo "[finetune] Baseline gate unmet; redirecting fine-tune to Tox21 tasks: ${FINETUNE_LABEL_COLS} (task_type=${FINETUNE_TASK_TYPE} metric=${FINETUNE_METRIC})" >&2
fi

# fix: ensure fine-tune emits stage outputs for downstream evaluation
export STAGE_OUTPUTS_DIR="${FINETUNE_DIR}/stage-outputs"
mkdir -p "$STAGE_OUTPUTS_DIR"

#ensure the parm matches train_jepa_ci.yml
if ! run_stage finetune; then
  rc=$?
  if [[ $rc -eq ${LINEAR_HEAD_SOFT_TIMEOUT_EXIT:-86} ]]; then
    echo "::warning::[finetune] linear-head training exited early after exhausting wall-clock headroom (rc=${rc})." >&2
    echo "[finetune] Marking stage as retryable. Rerun scripts/ci/run-finetune.sh once additional time budget is available." >&2
  fi
  exit "$rc"
fi

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages
