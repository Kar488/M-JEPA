#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="finetune"
if needs_stage "$FINETUNE_DIR" \
      "$APP_DIR/scripts/train_jepa.py" \
      "$APP_DIR/scripts/ci/train_jepa_ci.yml" \
      "$APP_DIR/scripts/ci/run-finetune.sh" \
      "$APP_DIR/scripts/ci/common.sh" \
      "$GRID_DIR/best_grid_config.json" \
      "$PRETRAIN_DIR/encoder.pt"; then
      
  echo "[finetune] starting from grid=$GRID_DIR pretrain=$PRETRAIN_DIR"
  export CKPT_DIR="$FINETUNE_DIR"
  build_argv_from_yaml finetune
  expand_array_vars ARGV

  export WANDB_NAME="finetune"
  export WANDB_JOB_TYPE="finetune"

  mapfile -t BEST < <(best_config_args finetune))
  expand_array_vars BEST
  
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" finetune \
    "${ARGV[@]}" 2>&1 | tee "$LOG_DIR/finetune.log"
  mark_stage_done "$FINETUNE_DIR"
  echo "[finetune] completed"
else
  echo "[finetune] cache hit - skipping"
fi
