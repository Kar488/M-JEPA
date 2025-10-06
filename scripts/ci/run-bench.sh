#!/usr/bin/env bash
set -euo pipefail
export MJEPACI_STAGE="bench"
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

encoder_ckpt="${PRETRAIN_DIR}/encoder.pt"
echo "[bench] using pretrain experiment id=${PRETRAIN_EXP_ID} checkpoint=${encoder_ckpt}" >&2
if [[ ! -f "$encoder_ckpt" ]]; then
  echo "[bench] required encoder checkpoint missing: $encoder_ckpt" >&2
  exit 1
fi


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

