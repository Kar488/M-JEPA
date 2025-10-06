#!/usr/bin/env bash
set -euo pipefail

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config
export MJEPACI_STAGE="finetune"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="finetune"
export WANDB_JOB_TYPE="finetune"

encoder_ckpt="${PRETRAIN_DIR}/encoder.pt"
manifest_path="${PRETRAIN_MANIFEST}"

echo "[finetune] using pretrain experiment id=${PRETRAIN_EXP_ID} checkpoint=${encoder_ckpt}" >&2
echo "[finetune] encoder manifest=${manifest_path}" >&2

if [[ ! -f "$encoder_ckpt" ]]; then
  echo "[finetune] required encoder checkpoint missing: $encoder_ckpt" >&2
  exit 1
fi

if [[ ! -f "$manifest_path" ]]; then
  echo "[finetune] required encoder manifest missing: $manifest_path" >&2
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

#ensure the parm matches train_jepa_ci.yml
run_stage finetune

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages
