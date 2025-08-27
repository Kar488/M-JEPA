#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_micromamba

STAGE="grid"
if needs_stage "$GRID_DIR" "$APP_DIR/scripts/train_jepa.py"; then
  echo "[grid] starting hyper-parameter search"
  export TRAIN_JEPA_CI="$APP_DIR/scripts/ci/train_jepa_ci.yml"
  build_argv_from_yaml grid_search
  # Build ARGV array from YAML and run grid-search with proper quoting
  $MMBIN run -n mjepa python "$APP_DIR/scripts/train_jepa.py" grid-search "${FILTERED[@]}" \
  # Whitelist flags that the "grid-search" subcommand actually accepts
  WHITELIST=(
    --mask-ratios --hidden-dims --num-layers-list --gnn-types --ema-decays --add-3d-options
    --pretrain-batch-sizes --finetune-batch-sizes
    --pretrain-epochs-options --finetune-epochs-options
    --learning-rates --sample-unlabeled --sample-labeled --n-rows-per-file
    --max-pretrain-batches --max-finetune-batches
    --time-budget-mins --force-tqdm --out-csv --best-config-out --use-wandb
  )
  # Build FILTERED from ARGV by keeping only whitelisted flags (and their values, if any)
  FILTERED=()
  i=0
  while [[ $i -lt ${#ARGV[@]} ]]; do
    flag="${ARGV[$i]}"
    next="${ARGV[$((i+1))]:-}"
    if [[ " ${WHITELIST[*]} " == *" $flag "* ]]; then
      FILTERED+=("$flag")
      # Heuristic: treat the next token as a value if it does not start with "--"
      if [[ -n "$next" && "$next" != --* ]]; then
        FILTERED+=("$next")
        ((i+=2))
        continue
      fi
    fi
    ((i+=1))
  done
  
  python "$APP_DIR/scripts/train_jepa.py" grid-search "${FILTERED[@]}" \
    2>&1 | tee "$LOG_DIR/grid.log"
  mark_stage_done "$GRID_DIR"
  echo "[grid] completed"
else
  echo "[grid] cache hit - skipping"
fi
