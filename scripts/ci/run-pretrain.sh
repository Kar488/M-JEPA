#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="pretrain"
if needs_stage "$PRETRAIN_DIR" \
      "$APP_DIR/scripts/train_jepa.py" \
      "$APP_DIR/scripts/ci/train_jepa_ci.yml" \
      "$APP_DIR/scripts/ci/run-pretrain.sh" \
      "$APP_DIR/scripts/ci/common.sh" \
      "$GRID_DIR/best_grid_config.json"; then

  echo "[pretrain] starting from grid outputs at $GRID_DIR"

  export CKPT_DIR="$PRETRAIN_DIR"
  build_argv_from_yaml pretrain # YAML → ARGV (stage config)
  expand_array_vars ARGV

  export WANDB_NAME="pretrain"
  export WANDB_JOB_TYPE="pretrain"

  mapfile -t BEST < <(best_config_args pretrain) # grid → tuned args
  expand_array_vars BEST
  
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" pretrain \
    "${ARGV[@]}" 2>&1 | tee "$LOG_DIR/pretrain.log"
  mark_stage_done "$PRETRAIN_DIR"
  echo "[pretrain] completed"
else
  echo "[pretrain] cache hit - skipping"
fi
