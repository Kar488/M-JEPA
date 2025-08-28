#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"
 
export WANDB_NAME="grid"
export WANDB_JOB_TYPE="grid"

# Decide which mode to use
DEBUG: GRID_MODE='custom' -> CLEAN='custom'
: "${GRID_MODE:?GRID_MODE must be set to 'wandb' or 'custom'}"
# allowed values: custom | wandb

# Normalize GRID_MODE (strip spaces and quotes)
# Strip spaces
GRID_MODE_CLEAN="${GRID_MODE//[[:space:]]/}"
# Strip quotes
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\"/}"
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\'/}"

echo "DEBUG: GRID_MODE='$GRID_MODE' -> CLEAN='$GRID_MODE_CLEAN'"

if [[ "$GRID_MODE_CLEAN" == "wandb" ]]; then
    echo "[grid] running wandb sweep agent"
    SWEEP_ID="$WANDB_ENTITY/$WANDB_PROJECT/$WANDB_SWEEP_ID1"
    echo "DEBUG: Using sweep ID: $SWEEP_ID"
    wandb agent --count ${WANDB_COUNT:-50} "$SWEEP_ID"
else
    echo "[grid] running custom grid-search"
    #ensure the parm matches train_jepa_ci.yml
    run_stage grid_search
fi