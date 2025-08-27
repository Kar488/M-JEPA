#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="grid"
if needs_stage "$GRID_DIR" "$APP_DIR/scripts/train_jepa.py"; then
  echo "[grid] starting hyper-parameter search"
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" grid-search "${ARGV[@]}" \
    2>&1 | tee "$LOG_DIR/grid.log"
  mark_stage_done "$GRID_DIR"
  echo "[grid] completed"
else
  echo "[grid] cache hit - skipping"
fi
