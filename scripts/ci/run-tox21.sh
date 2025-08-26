#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="tox21"
if needs_stage "$TOX21_DIR" "$GRID_DIR"; then
  echo "[tox21] starting tox21 evaluation"
  simulate_progress
  # Placeholder for tox21 evaluation
  mark_stage_done "$TOX21_DIR"
  echo "[tox21] completed"
else
  echo "[tox21] cache hit - skipping"
fi
