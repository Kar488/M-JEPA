#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="bench"
if needs_stage "$BENCH_DIR" "$GRID_DIR" "$PRETRAIN_DIR" "$FINETUNE_DIR"; then
  echo "[bench] starting benchmarks"
  simulate_progress
  # Placeholder for actual benchmarking invocation
  mark_stage_done "$BENCH_DIR"
  echo "[bench] completed"
else
  echo "[bench] cache hit - skipping"
fi
