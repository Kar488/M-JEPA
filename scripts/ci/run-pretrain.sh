#!/usr/bin/env bash
set -euo pipefail

export BESTCFG_NO_EPOCHS=1              # drop both epochs from best_config

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"

export WANDB_NAME="pretrain"
export WANDB_JOB_TYPE="pretrain"

export STAGE_OUTPUTS_DIR="${PRETRAIN_DIR}/stage-outputs"
mkdir -p "$STAGE_OUTPUTS_DIR"

#ensure the parm matches train_jepa_ci.yml

run_stage pretrain

# Preserve key artifacts alongside checkpoints so downstream jobs can fetch them easily.
if [[ -f "$PRETRAIN_DIR/encoder.pt" ]]; then
  ln -sf "$PRETRAIN_DIR/encoder.pt" "$STAGE_OUTPUTS_DIR/encoder.pt"
fi

if [[ -f "$PRETRAIN_MANIFEST" ]]; then
  mkdir -p "${PRETRAIN_DIR}/artifacts"
  cp "$PRETRAIN_MANIFEST" "${PRETRAIN_DIR}/artifacts/encoder_manifest.json"
  ln -sf "$PRETRAIN_MANIFEST" "$STAGE_OUTPUTS_DIR/encoder_manifest.json"
fi

unset BESTCFG_NO_EPOCHS                     # avoid leaking to other stages
