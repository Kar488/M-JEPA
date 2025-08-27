#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="pretrain"
if needs_stage "$PRETRAIN_DIR" "$GRID_DIR"; then
  echo "[pretrain] starting from grid outputs at $GRID_DIR"
  export CKPT_DIR="$PRETRAIN_DIR"
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" pretrain $(yaml_args pretrain) \
    2>&1 | tee "$LOG_DIR/pretrain.log"
  mark_stage_done "$PRETRAIN_DIR"
  echo "[pretrain] completed"
else
  echo "[pretrain] cache hit - skipping"
fi
