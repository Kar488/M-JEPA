#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="finetune"
if needs_stage "$FINETUNE_DIR" "$GRID_DIR" "$PRETRAIN_DIR"; then
  echo "[finetune] starting from grid=$GRID_DIR pretrain=$PRETRAIN_DIR"
  simulate_progress
  # Placeholder for actual finetuning invocation
  mark_stage_done "$FINETUNE_DIR"
  echo "[finetune] completed"
else
  echo "[finetune] cache hit - skipping"
fi
