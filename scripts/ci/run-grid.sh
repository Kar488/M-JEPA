#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="grid"

if needs_stage "$GRID_DIR" \
      "$APP_DIR/scripts/train_jepa.py" \
      "$APP_DIR/scripts/ci/train_jepa_ci.yml" \
      "$APP_DIR/scripts/ci/run-grid.sh" \
      "$APP_DIR/scripts/ci/common.sh"; then

  export TRAIN_JEPA_CI="$APP_DIR/scripts/ci/train_jepa_ci.yml"
  build_argv_from_yaml grid_search
  expand_array_vars ARGV

  export WANDB_NAME="grid"
  export WANDB_JOB_TYPE="grid"
  
  # Build ARGV array from YAML and run grid-search with proper quoting
  # --- Dynamic discovery of supported flags from the tool itself ---
  # This keeps the shell thin and avoids duplicating CLI knowledge here.
  mapfile -t ALLOWED < <(
    $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" grid-search --help \
      | sed -n 's/.*\(--[a-z0-9-]\+\).*/\1/p' | sort -u
  )
  is_allowed() {
    local f="$1"
    local a
    for a in "${ALLOWED[@]}"; do
      [[ "$a" == "$f" ]] && return 0
    done
    return 1
  }
  FILTERED=()
  i=0
  while [[ $i -lt ${#ARGV[@]} ]]; do
    flag="${ARGV[$i]}"
    if [[ "$flag" == --* ]] && is_allowed "$flag"; then
      FILTERED+=("$flag")
      # attach ALL subsequent non-flag tokens (supports nargs='+')
      j=$((i+1))
      while [[ $j -lt ${#ARGV[@]} && "${ARGV[$j]}" != --* ]]; do
        FILTERED+=("${ARGV[$j]}")
        ((j++))
      done
      i=$j; continue
    fi
    ((i+=1))
  done


  # derive budget from CLI if present; else fall back to env or 270
  get_val() { local i=0; while (( i < ${#FILTERED[@]} )); do [[ "${FILTERED[$i]}" == "$1" ]] && { echo "${FILTERED[$((i+1))]}"; return 0; }; ((i++)); done; return 1; }
  BUDGET_MINS="$(get_val --time-budget-mins || true)"
  [[ -z "$BUDGET_MINS" || ! "$BUDGET_MINS" =~ ^[0-9]+$ || "$BUDGET_MINS" -eq 0 ]] && BUDGET_MINS="${HARD_WALL_MINS:-270}"
  SOFT="${BUDGET_MINS}m"
  GRACE="${KILL_AFTER_SECS:-60}s"


  # timeout must wrap the whole micromamba run; use python -u instead of stdbuf
  timeout --signal=SIGINT --kill-after="$GRACE" "$SOFT" \
    # Run the grid search inside the micromamba env
    $MMBIN run -n mjepa env PYTHONUNBUFFERED=1 \
    python "$APP_DIR/scripts/train_jepa.py" grid-search "${FILTERED[@]}" \
    2>&1 | tee "$LOG_DIR/grid.log"

  mark_stage_done "$GRID_DIR"
  echo "[grid] completed"
else
   echo "[grid] cache hit - skipping"
fi
