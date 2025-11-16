#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config
export MJEPACI_STAGE="finetune"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

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

if [[ -n "$local_met_env" && -f "$local_met_env" ]]; then
  MET_ENV_FILE="$local_met_env"
elif [[ -n "$pretrain_met_env" && -f "$pretrain_met_env" ]]; then
  MET_ENV_FILE="$pretrain_met_env"
elif [[ -n "$local_met_env" ]]; then
  MET_ENV_FILE="$local_met_env"
else
  MET_ENV_FILE="$pretrain_met_env"
fi

if [[ -f "$MET_ENV_FILE" ]]; then
  while IFS='=' read -r key value; do
    [[ -z "$key" ]] && continue
    export "$key"="$value"
  done <"$MET_ENV_FILE"
  if [[ "${MET_BENCHMARK_BASELINE:-false}" == "true" ]]; then
    echo "[finetune] Baseline met benchmark; skipping fine-tune stage."
    exit 0
  fi
fi

# When the gate result is unavailable the reroute logic should no-op.
# Ensure the flag carries an explicit "unknown" marker instead of
# inheriting the "false" default from parameter expansion.
if [[ -z "${MET_BENCHMARK_BASELINE+x}" ]]; then
  export MET_BENCHMARK_BASELINE="unknown"
else
  # Treat blank or whitespace-only values as unknown to avoid false negatives.
  baseline_trimmed="${MET_BENCHMARK_BASELINE//[[:space:]]/}"
  if [[ -z "$baseline_trimmed" ]]; then
    export MET_BENCHMARK_BASELINE="unknown"
  else
    baseline_lower="${baseline_trimmed,,}"
    case "$baseline_lower" in
      true|false)
        export MET_BENCHMARK_BASELINE="$baseline_lower"
        ;;
      *)
        export MET_BENCHMARK_BASELINE="unknown"
        ;;
    esac
  fi
fi

baseline_flag="${MET_BENCHMARK_BASELINE:-false}"
baseline_flag_lc="${baseline_flag,,}"
if [[ "$baseline_flag_lc" == "false" ]]; then
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
