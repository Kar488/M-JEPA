#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"
 
export WANDB_NAME="grid"
export WANDB_JOB_TYPE="grid"

# Decide which mode to use
: "${GRID_MODE:?GRID_MODE must be set to 'wandb' or 'custom'}"  # error if unset
# allowed values: custom | wandb

# Normalize GRID_MODE (strip spaces and quotes)
# Strip spaces
GRID_MODE_CLEAN="${GRID_MODE//[[:space:]]/}"
# Strip quotes
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\"/}"
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\'/}"

echo "DEBUG: GRID_MODE='$GRID_MODE' -> CLEAN='$GRID_MODE_CLEAN'"

if [[ "$GRID_MODE_CLEAN" == "wandb" ]]; then
    echo "[grid] running wandb sweep agent"

    # keep one umbrella group; never force a run id for agent trials
    unset WANDB_NAME WANDB_RUN_ID
    : "${WANDB_RUN_GROUP:=${GITHUB_RUN_ID:-pipeline-$(date -u +%Y%m%dT%H%M%SZ)}}"
    export WANDB_RUN_GROUP WANDB_RESUME=allow

    # -------- Phase 1 orchestration (two sweeps) --------
    echo "[phase1] creating sweeps…"
    JEPA_ID=$(wandb sweep -q "${JEPA_SWEEP_SPEC:-$APP_DIR/sweep/phase1_jepa.yaml}")
    echo "$JEPA_ID" > "${GRID_DIR:-$APP_DIR/grid}/sweep_jepa.id"

    CONTRAST_ID=$(wandb sweep -q "${CONTRAST_SWEEP_SPEC:-$APP_DIR/sweep/phase1_contrastive.yaml}")
    echo "$CONTRAST_ID" > "${GRID_DIR:-$APP_DIR/grid}/sweep_contrast.id"
    echo "[phase1] JEPA=$JEPA_ID  CONTRAST=$CONTRAST_ID"

    cd "$APP_DIR"

    # run JEPA agents (serial on 1 GPU; change CUDA_VISIBLE_DEVICES to fan out)
    SWEEP_ID="$JEPA_ID"
    echo "[phase1] JEPA → $SWEEP_ID"
    export WANDB_COUNT=${WANDB_COUNT:-30}
    export SWEEP_ID
    run_with_timeout wandb_agent

    # run Contrastive agents
    SWEEP_ID="$CONTRAST_ID"
    echo "[phase1] Contrastive → $SWEEP_ID"
    export WANDB_COUNT=${WANDB_COUNT:-30}
    export SWEEP_ID
    run_with_timeout wandb_agent
    
    # paired-effect report
    echo "[phase1] paired-effect"
    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
    python -u "$APP_DIR/scripts/paired_effect_from_wandb.py" \
        --project "${WANDB_PROJECT}" --group "${WANDB_RUN_GROUP}" \
    2>&1 | tee "${LOG_DIR:-$APP_DIR/logs}/paired_effect.log"

    # === pick winner/task from paired-effect output ===
    PE_JSON="${GRID_DIR:-$APP_DIR/grid}/paired_effect.json"
    if [[ -f "$PE_JSON" ]]; then
        METHOD_WINNER="$(python - <<'PY'
import json, os
p = os.environ.get("PE_JSON")
with open(p, "r") as f:
    j = json.load(f)
print(j.get("winner","jepa"))
PY
)"
  TASK_FROM_PE="$(python - <<'PY'
import json, os
p = os.environ.get("PE_JSON")
with open(p, "r") as f:
    j = json.load(f)
print(j.get("task","regression"))
PY
)"
        export METHOD_WINNER TASK_FROM_PE
        echo "[phase1] paired-effect decided winner=${METHOD_WINNER} task=${TASK_FROM_PE}"
    else
        echo "[phase1][warn] ${PE_JSON} not found; falling back to defaults (winner=jepa, task=regression)"
        : "${METHOD_WINNER:=jepa}"
        : "${TASK_FROM_PE:=regression}"
    fi

    # export best + write Phase-2 (Bayes) sweep YAML (explicit args; no ARGV dependency)
    WINNER="${METHOD_WINNER:-jepa}"
    BEST_SWEEP="$JEPA_ID"; [[ "$WINNER" == "contrastive" ]] && BEST_SWEEP="$CONTRAST_ID"
    echo "[phase1] export best (winner=$WINNER) from sweep=$BEST_SWEEP"

    OUT_PATH="${EXPORT_OUT_PATH:-${GRID_DIR:-$APP_DIR/grid}/best_grid_config.json}"
    PHASE2_PATH="${EXPORT_PHASE2_PATH:-$APP_DIR/sweeps/grid_sweep_phase2.yaml}"

    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -u "$APP_DIR/scripts/export_best_from_wandb.py" \
        --sweep-id "$BEST_SWEEP" \
        --task "$TASK_FROM_PE" \
        --phase2-method bayes \
        --emit-bounds \
        --out "$OUT_PATH" \
        --phase2-yaml "$PHASE2_PATH" \
        --phase2-unlabeled-dir "${PHASE2_UNLABELED_DIR:-$APP_DIR/data/ZINC-canonicalized}" \
        --phase2-labeled-dir   "${PHASE2_LABELED_DIR:-$APP_DIR/data/katielinkmoleculenet_benchmark/train}" \
      2>&1 | tee "${LOG_DIR:-$APP_DIR/logs}/export_best_phase1.log"

    # create the Phase-2 sweep and persist its ID for the next job
    SWEEP_ID2="$(wandb sweep -q "$PHASE2_PATH")"
    echo -n "$SWEEP_ID2" > "${GRID_DIR:-$APP_DIR/grid}/phase2_sweep_id.txt"
    echo "[phase2] created sweep: $SWEEP_ID2  (saved to $GRID_DIR/phase2_sweep_id.txt)"
        
else
    echo "[grid] running custom grid-search"
    #ensure the parm matches train_jepa_ci.yml
    run_stage grid_search
fi