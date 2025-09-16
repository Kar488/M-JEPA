#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"
source "$(dirname "$0")/wandb_utils.sh"

export WANDB_NAME="grid"
export WANDB_JOB_TYPE="grid"

# Decide which mode to use
: "${GRID_MODE:?GRID_MODE must be set to 'wandb' or 'custom'}"
# allowed values: custom | wandb

# Normalize GRID_MODE (strip spaces and quotes)
GRID_MODE_CLEAN="${GRID_MODE//[[:space:]]/}"
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\"/}"
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\'/}"

# --- enforce pairing-friendly sweeps (identical shared knobs) ---
check_shared_equal() {
  local jepa="$1" ctr="$2"; shift 2
  local keys=(gnn_type hidden_dim num_layers contiguity)
  for k in "${keys[@]}"; do
    local a b
    a="$(yq ".parameters.${k}" "$jepa")"
    b="$(yq ".parameters.${k}" "$ctr")"
    if [[ "$a" != "$b" ]]; then
      echo "[fatal] sweep mismatch for '${k}':"
      echo "  JEPA:        $a"
      echo "  Contrastive: $b" >&2
      exit 1
    fi
  done

}

if [[ "$GRID_MODE_CLEAN" == "wandb" ]]; then
  echo "[grid] running wandb sweep agent"

  require_cmd perl
  require_cmd sed
  require_cmd dos2unix
  require_cmd yq

  unset WANDB_NAME WANDB_RUN_ID

  # Create a single sweep for JEPA and another for contrastive; reuse IDs
  : "${WANDB_COUNT:=30}"
  export WANDB_RUN_GROUP="${GITHUB_RUN_ID:-pipeline-$(date -u +%Y%m%dT%H%M%SZ)}"
  export WANDB_RESUME=allow
  echo "[phase1] WANDB_COUNT=$WANDB_COUNT group=$WANDB_RUN_GROUP"

  JEPA_SPEC="$APP_DIR/sweeps/sweep_phase1_jepa.yaml"
  CONTRAST_SPEC="$APP_DIR/sweeps/sweep_phase1_contrastive.yaml"
  [[ -f "$JEPA_SPEC" ]] || { echo "[fatal] missing sweep spec $JEPA_SPEC" >&2; exit 1; }
  [[ -f "$CONTRAST_SPEC" ]] || { echo "[fatal] missing sweep spec $CONTRAST_SPEC" >&2; exit 1; }


  TMP_JEPA="$(mktemp)";      yq ".method = \"random\"" "$JEPA_SPEC" > "$TMP_JEPA"
  TMP_CONTRAST="$(mktemp)";  yq ".method = \"random\"" "$CONTRAST_SPEC" > "$TMP_CONTRAST"

  check_shared_equal "$TMP_JEPA" "$TMP_CONTRAST"

  JEPA_ID="$(wandb_sweep_create "$TMP_JEPA")"
  CONTRAST_ID="$(wandb_sweep_create "$TMP_CONTRAST")"
  if [[ ! "$JEPA_ID" =~ ^[a-z0-9]{8}$ ]] || [[ ! "$CONTRAST_ID" =~ ^[a-z0-9]{8}$ ]]; then
    echo "[phase1][fatal] bad sweep ids: JEPA_ID='$JEPA_ID' CONTRAST_ID='$CONTRAST_ID'" >&2
    exit 1
  fi
  echo "[phase1] JEPA sweep id=$JEPA_ID  contrastive sweep id=$CONTRAST_ID"

  cd "$APP_DIR"
  export SWEEP_ID="$(qualify_sweep_id "$JEPA_ID")"
  echo "[phase1] launching JEPA agent for sweep $SWEEP_ID"
  run_with_timeout wandb_agent || exit 1

  export SWEEP_ID="$(qualify_sweep_id "$CONTRAST_ID")"
  echo "[phase1] launching contrastive agent for sweep $SWEEP_ID"
  run_with_timeout wandb_agent || exit 1

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

  PE_JSON="${GRID_DIR:-$APP_DIR/grid}/paired_effect.json"
  if [[ -f "$PE_JSON" ]]; then
    if command -v jq >/dev/null 2>&1; then
      METHOD_WINNER="$(jq -r '.winner // "jepa"' "$PE_JSON")"
      TASK_FROM_PE="$(jq -r '.task   // "regression"' "$PE_JSON")"
    else
      METHOD_WINNER="$(grep -o '"winner"\s*:\s*"[^"]*"' "$PE_JSON" | head -1 | sed 's/.*"winner"\s*:\s*"\([^"]*\)".*/\1/')"
      TASK_FROM_PE="$(grep -o '"task"\s*:\s*"[^"]*"' "$PE_JSON" | head -1 | sed 's/.*"task"\s*:\s*"\([^"]*\)".*/\1/')"
      : "${METHOD_WINNER:=jepa}"
      : "${TASK_FROM_PE:=regression}"
    fi
  else
    echo "[phase1][warn] ${PE_JSON} not found; falling back to defaults (winner=jepa, task=regression)"
    METHOD_WINNER="jepa"
    TASK_FROM_PE="regression"
  fi
  export METHOD_WINNER TASK_FROM_PE
  echo "[phase1] paired-effect decided winner=${METHOD_WINNER} task=${TASK_FROM_PE}"

  WINNER="$METHOD_WINNER"
  BEST_ID="$JEPA_ID"; [[ "$WINNER" == "contrastive" ]] && BEST_ID="$CONTRAST_ID"
  BEST_SWEEP="${WANDB_ENTITY}/${WANDB_PROJECT}/${BEST_ID}"

  OUT_PATH="${GRID_DIR:-$APP_DIR/grid}/best.json"
  
  TMP_BEST="$(mktemp)"
  TMP_PHASE2="$(mktemp)"
  LOG_TMP="$(mktemp)"
  trap 'rm -f "$TMP_BEST" "$TMP_PHASE2"' EXIT

  LOG_TMP="$(mktemp)"
  PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -u "$APP_DIR/scripts/ci/export_best_from_wandb.py" \
        --sweep-id "$BEST_SWEEP" \
        --task "$TASK_FROM_PE" \
        --phase2-method bayes \
        --emit-bounds \
        --out "$TMP_BEST" \
        --phase2-yaml "$TMP_PHASE2" \
        --phase2-unlabeled-dir "${PHASE2_UNLABELED_DIR:-$APP_DIR/data/ZINC-canonicalized}" \
        --phase2-labeled-dir   "${PHASE2_LABELED_DIR:-$APP_DIR/data/katielinkmoleculenet_benchmark/train}" \
      2>&1 | tee "$LOG_TMP"

  mkdir -p "$(dirname "$OUT_PATH")"
  cp "$TMP_BEST" "$OUT_PATH"
  echo "[phase1] staged best config to $OUT_PATH"

  FINAL_CFG="${EXPORT_OUT_PATH:-${GRID_DIR:-$APP_DIR/grid}/best_grid_config.json}"
  if [[ "$FINAL_CFG" != "$OUT_PATH" ]]; then
    mkdir -p "$(dirname "$FINAL_CFG")"
    cp "$TMP_BEST" "$FINAL_CFG"
    echo "[phase1] exported best config to $FINAL_CFG"
  fi

  CANONICAL_P2="$APP_DIR/sweeps/grid_sweep_phase2.yaml"
  mkdir -p "$(dirname "$CANONICAL_P2")"
  cp "$TMP_PHASE2" "$CANONICAL_P2"
  echo "[phase1] staged Phase-2 sweep YAML to $CANONICAL_P2"

  FINAL_P2="${EXPORT_PHASE2_PATH:-$CANONICAL_P2}"
  if [[ "$FINAL_P2" != "$CANONICAL_P2" ]]; then
    mkdir -p "$(dirname "$FINAL_P2")"
    cp "$TMP_PHASE2" "$FINAL_P2"
    

  SWEEP_ID2="$(wandb_sweep_create "$FINAL_P2")"
  [[ "$SWEEP_ID2" =~ ^[a-z0-9]{8}$ ]] || { echo "[phase2][fatal] bad sweep id: '$SWEEP_ID2'" >&2; exit 1; }
  echo -n "$SWEEP_ID2" > "${GRID_DIR:-$APP_DIR/grid}/phase2_sweep_id.txt"
  echo "[phase2] created sweep: $SWEEP_ID2  (saved to $GRID_DIR/phase2_sweep_id.txt)"

else
  echo "[grid] running custom grid-search"
  run_stage grid_search
fi
