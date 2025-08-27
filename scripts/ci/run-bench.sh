#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="benchmark"
export WANDB_JOB_TYPE="benchmark"

run_stage benchmark 
