#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="tox21"
export WANDB_JOB_TYPE="tox21"

run_stage tox21 