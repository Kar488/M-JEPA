#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR
export MJEPACI_STAGE="bench"
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

# Default labelled splits when the workflow does not provide overrides.  This
# keeps the benchmark stage non-empty even in smoke tests that only materialise
# the repository sample datasets.
: "${BENCH_VAL_DIR:=${APP_DIR}/data/katielinkmoleculenet_benchmark/val}"
: "${BENCH_TEST_DIR:=${APP_DIR}/data/katielinkmoleculenet_benchmark/test}"
: "${BENCH_RUN_VAL:=0}"
: "${BENCH_RUN_TEST:=1}"

should_run_bench() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

ci_print_env_diag

encoder_ckpt="$(resolve_encoder_checkpoint)"
echo "[bench] using pretrain experiment id=${PRETRAIN_EXP_ID} checkpoint=${encoder_ckpt}" >&2

if [[ -z "$encoder_ckpt" ]]; then
  echo "[ci] error: missing pretrain checkpoint for bench (PRETRAIN_ENCODER_PATH=${PRETRAIN_ENCODER_PATH:-<unset>}). Set PRETRAIN_EXP_ID=<id> to reuse an existing run or rerun pretrain." >&2
  exit 1
fi

if [[ -n "${PRETRAIN_ENCODER_PATH:-}" && "$encoder_ckpt" != "${PRETRAIN_ENCODER_PATH:-}" ]]; then
  echo "[bench] PRETRAIN_ENCODER_PATH pointed to ${PRETRAIN_ENCODER_PATH}; using manifest-derived path ${encoder_ckpt}" >&2
fi

if [[ ! -f "$encoder_ckpt" ]]; then
  echo "[ci] error: expected ${encoder_ckpt} but it was not found. Set PRETRAIN_EXP_ID=<id> to reuse an existing run or rerun pretrain." >&2
  exit 1
fi


# --- 1) VAL precheck (eval-only, no training) ---
if should_run_bench "${BENCH_RUN_VAL}"; then
  export BENCH_DATA_DIR="${BENCH_VAL_DIR}"             # becomes --labeled-dir
  export REPORT_STEM="val_${GITHUB_RUN_ID:-${RUN_ID:-bench}}"
  export WANDB_NAME="evaluate-val"
  export WANDB_JOB_TYPE="evaluate"
  run_stage bench
fi

# --- 2) TEST benchmark (eval-only, no training) ---
if should_run_bench "${BENCH_RUN_TEST}"; then
  export BENCH_DATA_DIR="${BENCH_TEST_DIR}"            # becomes --labeled-dir
  export REPORT_STEM="test_${GITHUB_RUN_ID:-${RUN_ID:-bench}}"
  export WANDB_NAME="benchmark-test"
  export WANDB_JOB_TYPE="benchmark"
  run_stage bench
fi

if ! should_run_bench "${BENCH_RUN_VAL}" && ! should_run_bench "${BENCH_RUN_TEST}"; then
  echo "[bench] BENCH_RUN_VAL=0 and BENCH_RUN_TEST=0; nothing to run." >&2
  exit 1
fi
