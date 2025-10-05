#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="${WANDB_NAME:-report}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-report}"

run_stage report
