#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="bench"
if needs_stage "$BENCH_DIR" \
      "$APP_DIR/scripts/train_jepa.py" \
      "$APP_DIR/scripts/ci/train_jepa_ci.yml" \
      "$APP_DIR/scripts/ci/run-bench.sh" \
      "$APP_DIR/scripts/ci/common.sh" \
      "$GRID_DIR/best_grid_config.json" \
      "$FINETUNE_DIR/seed_0/ft_best.pt"; then
  echo "[bench] starting benchmarks"
  build_argv_from_yaml bench
  expand_array_vars ARGV
  mapfile -t BEST < <(best_config_args bench)
  expand_array_vars BEST
  
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" benchmark \
      "${ARGV[@]}" 2>&1 | tee "$LOG_DIR/bench.log"
  mark_stage_done "$BENCH_DIR"
  echo "[bench] completed"
else
  echo "[bench] cache hit - skipping"
fi
