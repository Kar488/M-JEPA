#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="tox21"
if needs_stage "$TOX21_DIR" "$GRID_DIR"; then
  echo "[tox21] starting tox21 evaluation"
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" tox21 $(yaml_args tox21) \
    2>&1 | tee "$LOG_DIR/tox21.log"
  mark_stage_done "$TOX21_DIR"
  echo "[tox21] completed"
else
  echo "[tox21] cache hit - skipping"
fi
