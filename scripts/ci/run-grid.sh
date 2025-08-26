#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="grid"
if needs_stage "$GRID_DIR" "$APP_DIR/scripts/train_jepa.py"; then
  echo "[grid] starting hyper-parameter search"
  simulate_progress
  # Placeholder for actual grid search invocation
  mark_stage_done "$GRID_DIR"
  echo "[grid] completed"
else
  echo "[grid] cache hit - skipping"
fi
