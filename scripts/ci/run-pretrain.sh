#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="pretrain"
export WANDB_JOB_TYPE="pretrain"

#ensure the parm matches train_jepa_ci.yml
run_stage pretrain