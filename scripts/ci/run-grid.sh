#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="grid"

  export TRAIN_JEPA_CI="$APP_DIR/scripts/ci/train_jepa_ci.yml"
  build_argv_from_yaml grid_search
  expand_array_vars ARGV
  echo "[debug] APP_DIR=$APP_DIR"
    echo "[debug] PYTHONPATH=$PYTHONPATH"

    # Optional one-off debug (set DEBUG_IMPORTS=1 in the job to enable)
  if [[ "${DEBUG_IMPORTS:-0}" == "1" ]]; then
    echo "[debug] APP_DIR=$APP_DIR"
    echo "[debug] PYTHONPATH=$PYTHONPATH"
    $MMBIN run -n mjepa python - <<'PY' || true
import os, sys, importlib, traceback
print("APP_DIR:", os.environ.get("APP_DIR"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("sys.path[:5]:", sys.path[:5])
for mod in ("experiments.grid_search","experiments","mjepa.experiments"):
    try:
        m = importlib.import_module(mod)
        print(f"OK import {mod} @ {getattr(m,'__file__',m)}")
    except Exception as e:
        print(f"FAIL import {mod}: {e}"); traceback.print_exc()
PY
  fi


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

  # Run the grid search inside the micromamba env
  $MMBIN run -n mjepa \
    python "$APP_DIR/scripts/train_jepa.py" grid-search "${FILTERED[@]}" \
    2>&1 | tee "$LOG_DIR/grid.log"

  mark_stage_done "$GRID_DIR"
  echo "[grid] completed"
else
   echo "[grid] cache hit - skipping"
fi
