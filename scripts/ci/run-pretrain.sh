#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/lib/stage.sh"

export WANDB_NAME="pretrain"
export WANDB_JOB_TYPE="pretrain"

run_stage pretrain