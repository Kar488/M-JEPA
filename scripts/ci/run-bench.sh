#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"


# --- 1) VAL precheck (eval-only, no training) ---
export BENCH_DATA_DIR="${BENCH_VAL_DIR}"             # becomes --test-dir
export REPORT_STEM="val_${GITHUB_RUN_ID:-${RUN_ID:-bench}}"
export WANDB_NAME="evaluate-val"
export WANDB_JOB_TYPE="evaluate"
run_stage bench

# --- 2) TEST benchmark (eval-only, no training) ---
export BENCH_DATA_DIR="${BENCH_TEST_DIR}"            # becomes --test-dir
export REPORT_STEM="test_${GITHUB_RUN_ID:-${RUN_ID:-bench}}"
export WANDB_NAME="benchmark-test"
export WANDB_JOB_TYPE="benchmark"
run_stage bench

