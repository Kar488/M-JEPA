#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

export MJEPACI_STAGE="phase2"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

if declare -F ci_setup_vast_ssh_key >/dev/null 2>&1; then
  ci_setup_vast_ssh_key || true
fi

if (( ! FROZEN )) && [[ -n "${EXP_ID:-}" ]]; then
  if [[ -z "${GRID_EXP_ID:-}" || "${GRID_EXP_ID}" == "${ORIGINAL_PRETRAIN_EXP_ID:-}" ]]; then
    GRID_EXP_ID="$EXP_ID"
    export GRID_EXP_ID
  fi
  if [[ -z "${PRETRAIN_EXP_ID:-}" || "${PRETRAIN_EXP_ID}" == "${ORIGINAL_PRETRAIN_EXP_ID:-}" ]]; then
    PRETRAIN_EXP_ID="$EXP_ID"
    export PRETRAIN_EXP_ID
  fi
  if [[ -n "${GRID_EXP_ID:-}" ]]; then
    GRID_SOURCE_DIR="${EXPERIMENTS_ROOT%/}/${GRID_EXP_ID}/grid"
    export GRID_SOURCE_DIR
  fi
fi

if [[ -z "${STAGE_BIN:-}" ]]; then
  STAGE_BIN="run_stage"
fi
export STAGE_BIN

ci_print_env_diag "$STAGE_BIN"

steps=(phase2_sweep phase2_recheck phase2_export)
for step in "${steps[@]}"; do
  "$STAGE_BIN" "$step"
done
