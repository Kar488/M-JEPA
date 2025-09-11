#!/usr/bin/env bash
set -euo pipefail

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="tox21"
export WANDB_JOB_TYPE="tox21"

#ensure the parm matches train_jepa_ci.yml
run_stage tox21 

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages