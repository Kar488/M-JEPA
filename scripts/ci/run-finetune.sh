#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="finetune"
if needs_stage "$FINETUNE_DIR" "$GRID_DIR" "$PRETRAIN_DIR"; then
  echo "[finetune] starting from grid=$GRID_DIR pretrain=$PRETRAIN_DIR"
  export CKPT_DIR="$FINETUNE_DIR"
  build_argv_from_yaml finetune
  mapfile -t EXTRA < <(best_config_args finetune)
  ARGV+=("${EXTRA[@]}")
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" finetune "${ARGV[@]}" \
    2>&1 | tee "$LOG_DIR/finetune.log"
  mark_stage_done "$FINETUNE_DIR"
  echo "[finetune] completed"
else
  echo "[finetune] cache hit - skipping"
fi
