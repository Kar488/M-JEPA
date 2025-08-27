#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="tox21"
if needs_stage "$TOX21_DIR" \
      "$APP_DIR/scripts/train_jepa.py" \
      "$APP_DIR/scripts/ci/train_jepa_ci.yml" \
      "$APP_DIR/scripts/ci/run-tox21.sh" \
      "$APP_DIR/scripts/ci/common.sh" \
      "$GRID_DIR/best_grid_config.json" \
      "$APP_DIR/scripts/ci/data/tox21/data.csv"; then

  echo "[tox21] starting tox21 evaluation"
  build_argv_from_yaml tox21
  expand_array_vars ARGV
    
  export WANDB_NAME="tox21"
  export WANDB_JOB_TYPE="tox21"

  mapfile -t BEST < <(best_config_args tox21)
  expand_array_vars BEST
  
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" tox21 \
      "${ARGV[@]}" 2>&1 | tee "$LOG_DIR/tox21.log"
  mark_stage_done "$TOX21_DIR"
  echo "[tox21] completed"
else
  echo "[tox21] cache hit - skipping"
fi
