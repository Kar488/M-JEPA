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
  if [[ -n "${SWEEP_CACHE_DIR:-}" ]]; then
    echo "[phase1] dataset cache root: $SWEEP_CACHE_DIR"
  fi

  JEPA_SPEC="$APP_DIR/sweeps/sweep_phase1_jepa.yaml"
  CONTRAST_SPEC="$APP_DIR/sweeps/sweep_phase1_contrastive.yaml"
  [[ -f "$JEPA_SPEC" ]] || { echo "[fatal] missing sweep spec $JEPA_SPEC" >&2; exit 1; }
  [[ -f "$CONTRAST_SPEC" ]] || { echo "[fatal] missing sweep spec $CONTRAST_SPEC" >&2; exit 1; }


  TMP_JEPA="$(mktemp)";      yq ".method = \"random\"" "$JEPA_SPEC" > "$TMP_JEPA"
  TMP_CONTRAST="$(mktemp)";  yq ".method = \"random\"" "$CONTRAST_SPEC" > "$TMP_CONTRAST"

  if [[ -n "${PHASE1_BACKBONES:-}" ]]; then
    export PHASE1_BACKBONES
    for spec in "$TMP_JEPA" "$TMP_CONTRAST"; do
      yq -i '.parameters.gnn_type.values = (strenv(PHASE1_BACKBONES)
        | split(",")
        | map(gsub("^\\s+|\\s+$"; ""))
        | map(select(length > 0)))' "$spec"
    done
  fi

  if [[ -n "${PHASE1_SEEDS:-}" ]]; then
    export PHASE1_SEEDS
    for spec in "$TMP_JEPA" "$TMP_CONTRAST"; do
      yq -i '.parameters.seed.values = (strenv(PHASE1_SEEDS)
        | split(",")
        | map(gsub("^\\s+|\\s+$"; ""))
        | map(select(length > 0))
        | map(tonumber))' "$spec"
    done
  fi

  check_shared_equal "$TMP_JEPA" "$TMP_CONTRAST"

  JEPA_ID="$(wandb_sweep_create "$TMP_JEPA")"
  CONTRAST_ID="$(wandb_sweep_create "$TMP_CONTRAST")"
  if [[ ! "$JEPA_ID" =~ ^[a-z0-9]{8}$ ]] || [[ ! "$CONTRAST_ID" =~ ^[a-z0-9]{8}$ ]]; then
    echo "[phase1][fatal] bad sweep ids: JEPA_ID='$JEPA_ID' CONTRAST_ID='$CONTRAST_ID'" >&2
    exit 1
  fi
  echo "[phase1] JEPA sweep id=$JEPA_ID  contrastive sweep id=$CONTRAST_ID"

  cd "$APP_DIR"
  BASE_LOG_DIR="${LOG_DIR:-$APP_DIR/logs}"
  mapfile -t GRID_VISIBLE_GPUS < <(visible_gpu_ids)
  PHASE1_GPU_COUNT="${#GRID_VISIBLE_GPUS[@]}"
  if (( PHASE1_GPU_COUNT >= 2 )); then
    echo "[phase1] detected $PHASE1_GPU_COUNT GPUs; launching agents in parallel"
    declare -a PHASE1_GPU_SPLITS
    split_gpu_ids PHASE1_GPU_SPLITS 2 "${GRID_VISIBLE_GPUS[@]}"

    (
      export LOG_DIR="${BASE_LOG_DIR}/phase1_jepa"
      mkdir -p "$LOG_DIR"
      if [[ -n "${PHASE1_GPU_SPLITS[0]:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${PHASE1_GPU_SPLITS[0]}"
        echo "[phase1] JEPA agent using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      fi
      export SWEEP_ID="$(qualify_sweep_id "$JEPA_ID")"
      echo "[phase1] launching JEPA agent for sweep $SWEEP_ID"
      run_with_timeout wandb_agent
    ) &
    PHASE1_JEPA_PID=$!

    (
      export LOG_DIR="${BASE_LOG_DIR}/phase1_contrastive"
      mkdir -p "$LOG_DIR"
      if [[ -n "${PHASE1_GPU_SPLITS[1]:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${PHASE1_GPU_SPLITS[1]}"
        echo "[phase1] contrastive agent using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      fi
      export SWEEP_ID="$(qualify_sweep_id "$CONTRAST_ID")"
      echo "[phase1] launching contrastive agent for sweep $SWEEP_ID"
      run_with_timeout wandb_agent
    ) &
    PHASE1_CONTRAST_PID=$!

    set +e
    wait "$PHASE1_JEPA_PID"
    PHASE1_JEPA_RC=$?
    wait "$PHASE1_CONTRAST_PID"
    PHASE1_CONTRAST_RC=$?
    set -e

    if (( PHASE1_JEPA_RC != 0 || PHASE1_CONTRAST_RC != 0 )); then
      echo "[phase1][fatal] sweep agents failed: JEPA rc=$PHASE1_JEPA_RC contrastive rc=$PHASE1_CONTRAST_RC" >&2
      exit 1
    fi
  else
    export SWEEP_ID="$(qualify_sweep_id "$JEPA_ID")"
    echo "[phase1] launching JEPA agent for sweep $SWEEP_ID"
    run_with_timeout wandb_agent || exit 1

    export SWEEP_ID="$(qualify_sweep_id "$CONTRAST_ID")"
    echo "[phase1] launching contrastive agent for sweep $SWEEP_ID"
    run_with_timeout wandb_agent || exit 1
  fi

  PE_METRIC="val_rmse"
  [[ "${TASK_FROM_PE:-regression}" == "classification" ]] && PE_METRIC="val_auc"

  # Require that paired-effect analysis only considers runs that have reached
  # the minimum training budgets that Phase-1 sweeps schedule.  Allow
  # overrides via environment variables so CI callers can tighten or loosen the
  # thresholds without editing this script.
  if [[ -z "${PE_MIN_PRETRAIN_EPOCHS+x}" ]]; then
    PE_MIN_PRETRAIN_EPOCHS=5
  fi
  if [[ -z "${PE_MIN_FINETUNE_EPOCHS+x}" ]]; then
    PE_MIN_FINETUNE_EPOCHS=1
  fi
  if [[ -z "${PE_MIN_PRETRAIN_BATCHES+x}" ]]; then
    PE_MIN_PRETRAIN_BATCHES=200
  fi

  PE_FILTER_FLAGS=()
  if [[ -n "${PE_MIN_PRETRAIN_EPOCHS}" ]]; then
    PE_FILTER_FLAGS+=("--min_pretrain_epochs" "${PE_MIN_PRETRAIN_EPOCHS}")
  fi
  if [[ -n "${PE_MIN_FINETUNE_EPOCHS}" ]]; then
    PE_FILTER_FLAGS+=("--min_finetune_epochs" "${PE_MIN_FINETUNE_EPOCHS}")
  fi
  if [[ -n "${PE_MIN_PRETRAIN_BATCHES}" ]]; then
    PE_FILTER_FLAGS+=("--min_pretrain_batches" "${PE_MIN_PRETRAIN_BATCHES}")
  fi

  "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
    python -u "$APP_DIR/scripts/ci/paired_effect_from_wandb.py" \
      --project "${WANDB_PROJECT}" \
      --group   "${WANDB_RUN_GROUP}" \
      --metric  "${PE_METRIC}" \
      --aggregate pair-seed \
      --seed "${CI_SEED:-42}" \
      --strict \
      "${PE_FILTER_FLAGS[@]}" \
    2>&1 | tee "${LOG_DIR:-$APP_DIR/logs}/paired_effect.log"

  PE_JSON="${GRID_DIR:-$APP_DIR/grid}/paired_effect.json"
  if [[ -f "$PE_JSON" ]]; then
    if ! py=$(python_bin 2>/dev/null); then
      echo "[phase1][fatal] python interpreter not found for paired-effect resolution" >&2
      exit 1
    fi

    if read -r METHOD_WINNER TASK_FROM_PE PE_DECISION_STATUS < <(
      "$py" -m scripts.ci.phase1_decision "$PE_JSON"
    ); then
      if [[ "$PE_DECISION_STATUS" == "tie" ]]; then
        echo "[phase1][warn] paired-effect reported a tie (winner field: ${METHOD_WINNER})"
        if [[ -n "${PHASE1_TIE_BREAKER:-}" ]]; then
          case "${PHASE1_TIE_BREAKER}" in
            jepa|contrastive)
              METHOD_WINNER="${PHASE1_TIE_BREAKER}"
              echo "[phase1] resolved tie via PHASE1_TIE_BREAKER=${METHOD_WINNER}"
              ;;
            *)
              echo "[phase1][fatal] invalid PHASE1_TIE_BREAKER='${PHASE1_TIE_BREAKER}' (expected 'jepa' or 'contrastive')" >&2
              exit 1
              ;;
          esac
        else
          echo "[phase1][fatal] tie detected; set PHASE1_TIE_BREAKER to 'jepa' or 'contrastive' to continue" >&2
          exit 1
        fi
      fi
    else
      echo "[phase1][fatal] failed to interpret paired-effect output" >&2
      exit 1
    fi
  else
    if [[ -n "${PHASE1_FALLBACK_WINNER:-}" ]]; then
      case "${PHASE1_FALLBACK_WINNER}" in
        jepa|contrastive)
          METHOD_WINNER="${PHASE1_FALLBACK_WINNER}"
          TASK_FROM_PE="${PHASE1_FALLBACK_TASK:-regression}"
          echo "[phase1][warn] ${PE_JSON} not found; using PHASE1_FALLBACK_WINNER=${METHOD_WINNER}"
          ;;
        *)
          echo "[phase1][fatal] invalid PHASE1_FALLBACK_WINNER='${PHASE1_FALLBACK_WINNER}' (expected 'jepa' or 'contrastive')" >&2
          exit 1
          ;;
      esac
    else
      echo "[phase1][fatal] ${PE_JSON} not found and no PHASE1_FALLBACK_WINNER provided" >&2
      exit 1
    fi
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

  CANONICAL_P2="${GRID_DIR:-$APP_DIR/grid}/grid_sweep_phase2.yaml"
  mkdir -p "$(dirname "$CANONICAL_P2")"
  cp "$TMP_PHASE2" "$CANONICAL_P2"
  echo "[phase1] staged Phase-2 sweep YAML to $CANONICAL_P2"

  FINAL_P2="${EXPORT_PHASE2_PATH:-$CANONICAL_P2}"
  if [[ "$FINAL_P2" != "$CANONICAL_P2" ]]; then
    mkdir -p "$(dirname "$FINAL_P2")"
    cp "$TMP_PHASE2" "$FINAL_P2"
  fi

  SWEEP_ID2="$(wandb_sweep_create "$FINAL_P2")"
  [[ "$SWEEP_ID2" =~ ^[a-z0-9]{8}$ ]] || { echo "[phase2][fatal] bad sweep id: '$SWEEP_ID2'" >&2; exit 1; }
  echo -n "$SWEEP_ID2" > "${GRID_DIR:-$APP_DIR/grid}/phase2_sweep_id.txt"
  echo "[phase2] created sweep: $SWEEP_ID2  (saved to $GRID_DIR/phase2_sweep_id.txt)"

else
  echo "[grid] running custom grid-search"
  run_stage grid_search
fi
