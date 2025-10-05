#!/usr/bin/env bash
set -euo pipefail

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="finetune"
export WANDB_JOB_TYPE="finetune"

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
