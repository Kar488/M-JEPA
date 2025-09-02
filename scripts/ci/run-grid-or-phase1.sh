#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"
source "$(dirname "$0")/wandb_utils.sh"
 
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

# -------- resolve JEPA/Contrastive sweep spec paths cleanly ----------
resolve_spec() {
  local hint="$1"
  shift
  # candidates to try, in order
  local candidates=(
    "$APP_DIR/sweeps/sweep_phase1_${hint}.yaml"
    "$APP_DIR/sweep/sweep_phase1_${hint}.yaml"
    "$APP_DIR/sweep_phase1_${hint}.yaml"
  )
  for p in "${candidates[@]}"; do
    [[ -f "$p" ]] && { echo "$p"; return 0; }
  done
  return 1
}

if [[ "$GRID_MODE_CLEAN" == "wandb" ]]; then
    echo "[grid] running wandb sweep agent"

    # verify external tooling we rely on
    require_cmd perl
    require_cmd sed
    require_cmd dos2unix

    # keep one umbrella group; never force a run id for agent trials
    unset WANDB_NAME WANDB_RUN_ID
    : "${WANDB_RUN_GROUP:=${GITHUB_RUN_ID:-pipeline-$(date -u +%Y%m%dT%H%M%SZ)}}"
    export WANDB_RUN_GROUP WANDB_RESUME=allow

    # -------- Phase 1 orchestration (two sweeps) --------

    # Create sweeps via the Python module inside the env that has wandb installed
    # (pass project/entity explicitly so it doesn't rely on local config)
    echo "[phase1] creating sweeps…jepa"
    JEPA_SPEC="${JEPA_SWEEP_SPEC:-}"
    if [[ -z "$JEPA_SPEC" ]]; then
        JEPA_SPEC="$(resolve_spec jepa)" || {
            echo "[fatal] missing sweep spec for JEPA."
            echo "  looked in:"
            echo "    - $APP_DIR/sweeps/sweep_phase1_jepa.yaml"
            echo "    - $APP_DIR/sweep/sweep_phase1_jepa.yaml"
            echo "    - $APP_DIR/sweep_phase1_jepa.yaml"
            echo "  git status:"; git -C "$APP_DIR" status --porcelain
            echo "  tree:"; ls -lah "$APP_DIR/sweeps" 2>/dev/null || true
            exit 1
        }
    fi
    JEPA_ID="$(wandb_sweep_create "$JEPA_SPEC")"

    echo "[phase1] creating sweeps…contrastive"
    CONTRAST_SPEC="${CONTRAST_SWEEP_SPEC:-}" 
    if [[ -z "$CONTRAST_SPEC" ]]; then
        CONTRAST_SPEC="$(resolve_spec contrastive)" || {
            echo "[fatal] missing sweep spec for Contrastive (same candidate paths)."; exit 1;
        }
    fi
    CONTRAST_ID="$(wandb_sweep_create "$CONTRAST_SPEC")"

    # hard-validate: must be exactly 8 lowercase letters/digits
    if [[ ! "$JEPA_ID" =~ ^[a-z0-9]{8}$ ]] || [[ ! "$CONTRAST_ID" =~ ^[a-z0-9]{8}$ ]]; then
        echo "[phase1][fatal] bad sweep ids: JEPA_ID='$JEPA_ID' CONTRAST_ID='$CONTRAST_ID'" >&2
        exit 1
    fi
    echo "[phase1] JEPA=$JEPA_ID  CONTRAST=$CONTRAST_ID"

    cd "$APP_DIR"

    # run JEPA agents (serial on 1 GPU; change CUDA_VISIBLE_DEVICES to fan out)
    export SWEEP_ID="$(qualify_sweep_id "$JEPA_ID")"
    echo "[phase1] JEPA → $SWEEP_ID"
    echo "Count → $WANDB_COUNT"
    : "${WANDB_COUNT:=30}"
    run_with_timeout wandb_agent || exit 1

    # run Contrastive agents
    export SWEEP_ID="$(qualify_sweep_id "$CONTRAST_ID")"
    echo "[phase1] Contrastive → $SWEEP_ID"
    echo "Count → $WANDB_COUNT"
    : "${WANDB_COUNT:=30}"
    run_with_timeout wandb_agent || exit 1
    
    # paired-effect report
    echo "[phase1] paired-effect"
    # choose metric by the task the runs log (AUC for classification, RMSE for regression)
    PE_METRIC="val_rmse"
    [[ "${TASK_FROM_PE:-regression}" == "classification" ]] && PE_METRIC="val_auc"
    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -u "$APP_DIR/scripts/ci/paired_effect_from_wandb.py" \
        --project "${WANDB_PROJECT}" \
        --group   "${WANDB_RUN_GROUP}" \
        --metric  "${PE_METRIC}" \
        --aggregate pair-seed \
        --seed "${CI_SEED:-42}" \
        --strict \
      2>&1 | tee "${LOG_DIR:-$APP_DIR/logs}/paired_effect.log"


    # === pick winner/task from paired-effect output ===
    PE_JSON="${GRID_DIR:-$APP_DIR/grid}/paired_effect.json"
    if [[ -f "$PE_JSON" ]]; then
        METHOD_WINNER="$(env PE_JSON="$PE_JSON" python - <<'PY'
import json, os
p = os.environ.get("PE_JSON")
with open(p, "r") as f:
    j = json.load(f)
print(j.get("winner","jepa"))
PY
)"
  TASK_FROM_PE="$(env PE_JSON="$PE_JSON" python - <<'PY'
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
    BEST_ID="$JEPA_ID"; [[ "$WINNER" == "contrastive" ]] && BEST_ID="$CONTRAST_ID"
    echo "[phase1] export best (winner=$WINNER) from sweep=$BEST_ID"
    BEST_SWEEP="${WANDB_ENTITY}/${WANDB_PROJECT}/${BEST_ID}"

    OUT_PATH="${EXPORT_OUT_PATH:-${GRID_DIR:-$APP_DIR/grid}/best_grid_config.json}"
    PHASE2_PATH="${EXPORT_PHASE2_PATH:-$APP_DIR/sweeps/grid_sweep_phase2.yaml}"

    PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -u "$APP_DIR/scripts/ci/export_best_from_wandb.py" \
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
    SWEEP_ID2="$(wandb_sweep_create "$PHASE2_PATH")"
    [[ "$SWEEP_ID2" =~ ^[a-z0-9]{8}$ ]] || { echo "[phase2][fatal] bad sweep id: '$SWEEP_ID2'"; exit 1; }
    echo -n "$SWEEP_ID2" > "${GRID_DIR:-$APP_DIR/grid}/phase2_sweep_id.txt"
    echo "[phase2] created sweep: $SWEEP_ID2  (saved to $GRID_DIR/phase2_sweep_id.txt)"
        
else
    echo "[grid] running custom grid-search"
    #ensure the parm matches train_jepa_ci.yml
    run_stage grid_search
fi