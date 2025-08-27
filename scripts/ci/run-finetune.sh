#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="finetune"
if needs_stage "$FINETUNE_DIR" "$GRID_DIR" "$PRETRAIN_DIR"; then
  echo "[finetune] starting from grid=$GRID_DIR pretrain=$PRETRAIN_DIR"
  export CKPT_DIR="$FINETUNE_DIR"
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" finetune $(yaml_args finetune) \
    2>&1 | tee "$LOG_DIR/finetune.log"
  mark_stage_done "$FINETUNE_DIR"
  echo "[finetune] completed"
else
  echo "[finetune] cache hit - skipping"
fi
