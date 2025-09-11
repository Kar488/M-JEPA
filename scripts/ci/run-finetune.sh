#!/usr/bin/env bash
set -euo pipefail

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="finetune"
export WANDB_JOB_TYPE="finetune"

#ensure the parm matches train_jepa_ci.yml
run_stage finetune 

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages
