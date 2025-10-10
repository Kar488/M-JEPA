#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

if [[ "${GRID_EXP_ID+x}" == x ]]; then
  CI_PHASE2_INCOMING_GRID_EXP_ID_SET=1
  CI_PHASE2_INCOMING_GRID_EXP_ID="$GRID_EXP_ID"
else
  CI_PHASE2_INCOMING_GRID_EXP_ID_SET=0
  CI_PHASE2_INCOMING_GRID_EXP_ID=""
fi

if [[ "${PRETRAIN_EXP_ID+x}" == x ]]; then
  CI_PHASE2_INCOMING_PRETRAIN_EXP_ID_SET=1
  CI_PHASE2_INCOMING_PRETRAIN_EXP_ID="$PRETRAIN_EXP_ID"
else
  CI_PHASE2_INCOMING_PRETRAIN_EXP_ID_SET=0
  CI_PHASE2_INCOMING_PRETRAIN_EXP_ID=""
fi

export MJEPACI_STAGE="phase2"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

if declare -F ci_setup_vast_ssh_key >/dev/null 2>&1; then
  ci_setup_vast_ssh_key || true
fi

common_grid_exp_id="${GRID_EXP_ID:-}"
common_pretrain_exp_id="${PRETRAIN_EXP_ID:-}"

ci_phase2_should_bind_ids=0
if [[ -n "${EXP_ID:-}" ]]; then
  if (( ! FROZEN )) || [[ "${FORCE_UNFREEZE_GRID:-}" == "1" ]]; then
    ci_phase2_should_bind_ids=1
  fi
fi

new_grid_exp_id=""
new_pretrain_exp_id=""

if (( ci_phase2_should_bind_ids )); then
  if [[ "${GRID_EXP_ID:-}" != "${EXP_ID:-}" ]]; then
    if (( ! CI_PHASE2_INCOMING_GRID_EXP_ID_SET )) || [[ -z "${CI_PHASE2_INCOMING_GRID_EXP_ID}" ]] || \
       [[ -z "${ORIGINAL_PRETRAIN_EXP_ID:-}" ]] || \
       [[ "${GRID_EXP_ID:-}" == "${ORIGINAL_PRETRAIN_EXP_ID:-}" ]] || \
       ([[ -n "${PRETRAIN_STATE_ID:-}" ]] && [[ "${GRID_EXP_ID:-}" == "${PRETRAIN_STATE_ID:-}" ]]); then
      new_grid_exp_id="${EXP_ID}"
    fi
  fi

  if [[ "${PRETRAIN_EXP_ID:-}" != "${EXP_ID:-}" ]]; then
    if (( ! CI_PHASE2_INCOMING_PRETRAIN_EXP_ID_SET )) || [[ -z "${CI_PHASE2_INCOMING_PRETRAIN_EXP_ID}" ]] || \
       [[ -z "${ORIGINAL_PRETRAIN_EXP_ID:-}" ]] || \
       [[ "${PRETRAIN_EXP_ID:-}" == "${ORIGINAL_PRETRAIN_EXP_ID:-}" ]] || \
       ([[ -n "${PRETRAIN_STATE_ID:-}" ]] && [[ "${PRETRAIN_EXP_ID:-}" == "${PRETRAIN_STATE_ID:-}" ]]); then
      new_pretrain_exp_id="${EXP_ID}"
    fi
  fi
fi

if [[ -n "$new_grid_exp_id" ]]; then
  GRID_EXP_ID="$new_grid_exp_id"
  export GRID_EXP_ID
fi

if [[ -n "$new_pretrain_exp_id" ]]; then
  PRETRAIN_EXP_ID="$new_pretrain_exp_id"
  export PRETRAIN_EXP_ID
fi

if [[ -n "$new_grid_exp_id" ]] || [[ -n "$new_pretrain_exp_id" ]]; then
  echo "[phase2] binding EXP_ID=${EXP_ID:-<unset>} -> GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>} (previous grid=${common_grid_exp_id:-<unset>} pretrain=${common_pretrain_exp_id:-<unset>})" >&2
  ci_phase2_refresh_lineage_bindings \
    "$new_pretrain_exp_id" \
    "$new_grid_exp_id" \
    "$common_pretrain_exp_id" \
    "$common_grid_exp_id"
elif [[ -z "${GRID_SOURCE_DIR:-}" && -n "${GRID_EXP_ID:-}" ]]; then
  GRID_SOURCE_DIR="${EXPERIMENTS_ROOT%/}/${GRID_EXP_ID}/grid"
  export GRID_SOURCE_DIR
fi

unset CI_PHASE2_INCOMING_GRID_EXP_ID_SET CI_PHASE2_INCOMING_GRID_EXP_ID
unset CI_PHASE2_INCOMING_PRETRAIN_EXP_ID_SET CI_PHASE2_INCOMING_PRETRAIN_EXP_ID

if [[ -z "${STAGE_BIN:-}" ]]; then
  STAGE_BIN="run_stage"
fi
export STAGE_BIN

ci_print_env_diag "$STAGE_BIN"

steps=(phase2_sweep phase2_recheck phase2_export)
for step in "${steps[@]}"; do
  "$STAGE_BIN" "$step"
done
