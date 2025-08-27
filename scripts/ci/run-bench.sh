#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="bench"
if needs_stage "$BENCH_DIR" "$GRID_DIR" "$PRETRAIN_DIR" "$FINETUNE_DIR"; then
  echo "[bench] starting benchmarks"
  build_argv_from_yaml bench
  mapfile -t EXTRA < <(best_config_args bench)
  ARGV+=("${EXTRA[@]}")
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" benchmark "${ARGV[@]}" \
    2>&1 | tee "$LOG_DIR/bench.log"
  mark_stage_done "$BENCH_DIR"
  echo "[bench] completed"
else
  echo "[bench] cache hit - skipping"
fi
