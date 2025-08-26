#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="pretrain"
if needs_stage "$PRETRAIN_DIR" "$GRID_DIR"; then
  echo "[pretrain] starting from grid outputs at $GRID_DIR"
  simulate_progress
  # Placeholder for actual pretraining invocation
  mark_stage_done "$PRETRAIN_DIR"
  echo "[pretrain] completed"
else
  echo "[pretrain] cache hit - skipping"
fi
