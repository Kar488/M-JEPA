#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config
export MJEPACI_STAGE="finetune"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

ci_print_env_diag

if [[ -n "${BESTCFG_SKIP:-}" ]]; then
  BESTCFG_SKIP="${BESTCFG_SKIP} max_pretrain_batches max_finetune_batches"
else
  BESTCFG_SKIP="max_pretrain_batches max_finetune_batches"
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
MET_ENV_FILE="${EXP_ROOT}/met_benchmark.env"
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

baseline_flag="${MET_BENCHMARK_BASELINE:-false}"
baseline_flag_lc="${baseline_flag,,}"
if [[ "$baseline_flag_lc" == "false" ]]; then
  : "${FINETUNE_LABELED_DIR:=${APP_DIR}/data/tox21/data.csv}"
  if [[ -z "${FINETUNE_LABEL_COL:-}" ]]; then
    if [[ -n "${TOX21_FINE_TUNE_TASK:-}" ]]; then
      FINETUNE_LABEL_COL="$TOX21_FINE_TUNE_TASK"
    else
      FINETUNE_LABEL_COL="NR-AR"
    fi
  fi
  : "${FINETUNE_TASK_TYPE:=classification}"
  : "${FINETUNE_METRIC:=val_auc}"
  : "${FINETUNE_USE_SCAFFOLD:=true}"
  : "${FINETUNE_SEED_0:=11}"
  : "${FINETUNE_SEED_1:=29}"
  : "${FINETUNE_SEED_2:=37}"
  : "${FINETUNE_DATASET_OVERRIDE_REASON:=tox21_gate_failure}"

  export FINETUNE_LABELED_DIR
  export FINETUNE_LABEL_COL
  export FINETUNE_TASK_TYPE
  export FINETUNE_METRIC
  export FINETUNE_USE_SCAFFOLD
  export FINETUNE_SEED_0 FINETUNE_SEED_1 FINETUNE_SEED_2
  export FINETUNE_DATASET_OVERRIDE_REASON

  echo "[finetune] Baseline gate unmet; redirecting fine-tune to Tox21 task '${FINETUNE_LABEL_COL}'" >&2
fi

# fix: ensure fine-tune emits stage outputs for downstream evaluation
export STAGE_OUTPUTS_DIR="${FINETUNE_DIR}/stage-outputs"
mkdir -p "$STAGE_OUTPUTS_DIR"

#ensure the parm matches train_jepa_ci.yml
run_stage finetune

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages
