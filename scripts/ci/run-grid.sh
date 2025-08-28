#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"
 
export WANDB_NAME="grid"
export WANDB_JOB_TYPE="grid"

# Decide which mode to use
: "${GRID_MODE:?GRID_MODE must be set to 'wandb' or 'custom'}"  # error if unset
# allowed values: custom | wandb

# Normalize GRID_MODE (strip spaces and quotes)
# Strip spaces
GRID_MODE_CLEAN="${GRID_MODE//[[:space:]]/}"
# Strip quotes
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\"/}"
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\'/}"

echo "DEBUG: GRID_MODE='$GRID_MODE' -> CLEAN='$GRID_MODE_CLEAN'"

: "${HARD_WALL_MINS:=240}"
: "${KILL_AFTER_SECS:=60}"

SOFT=$((HARD_WALL_MINS*60))
GRACE="${KILL_AFTER_SECS:-60}"

echo "[stage] wall budget=${HARD_WALL_MINS}m (${SOFT}s), grace=${GRACE}s"

  ensure_micromamba
  : "${WANDB_NAME:=$s}"; export WANDB_NAME
  : "${WANDB_JOB_TYPE:=$s}"; export WANDB_JOB_TYPE
  export WANDB_RUN_GROUP="$GITHUB_RUN_ID"
  
if [[ "$GRID_MODE_CLEAN" == "wandb" ]]; then
    echo "[grid] running wandb sweep agent"
    SWEEP_ID="$WANDB_ENTITY/$WANDB_PROJECT/$WANDB_SWEEP_ID1"
    echo "DEBUG: Using sweep ID: $SWEEP_ID"
    timeout --signal=SIGINT --kill-after="$GRACE" "$SOFT" \
        python -m wandb agent --count ${WANDB_COUNT:-50} "$SWEEP_ID" \
            2>&1 | tee "$LOG_DIR/wandb_agent.log"
        
else
    echo "[grid] running custom grid-search"
    #ensure the parm matches train_jepa_ci.yml
    run_stage grid_search
fi