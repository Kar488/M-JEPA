#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="pretrain"
export WANDB_JOB_TYPE="pretrain"

#ensure the parm matches train_jepa_ci.yml
BEST_ARGS_FILE="${GRID_DIR}/phase2_best_args.txt"
[[ -s "$BEST_ARGS_FILE" ]] || { echo "[pretrain][fatal] missing $BEST_ARGS_FILE"; exit 2; }

EXTRA_ARGS="$(cat "$BEST_ARGS_FILE")"
echo "[pretrain] using EXTRA_ARGS=${EXTRA_ARGS}"

run_stage pretrain $EXTRA_ARGS