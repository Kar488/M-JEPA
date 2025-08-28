#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"
 
export WANDB_NAME="grid"
export WANDB_JOB_TYPE="grid"

# Decide which mode to use
: "${GRID_MODE:=custom}"   # default is custom grid-search
# allowed values: custom | wandb

if [ "$GRID_MODE" = "wandb" ]; then
    echo "[grid] running wandb sweep agent"
    # Replace with your actual sweep ID
    SWEEP_ID="karthik-iyer-la-trobe-university/mjepa/ufcnx6kq"
    # Run N configs on this machine
    wandb agent --count 200 "$SWEEP_ID"

else
    echo "[grid] running custom grid-search"
    #ensure the parm matches train_jepa_ci.yml
    run_stage grid_search
fi