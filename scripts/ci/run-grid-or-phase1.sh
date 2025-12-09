#!/usr/bin/env bash
set -euo pipefail

# ``common.sh`` derives canonical directories from ``RUN_ID``/``EXP_ID``/
# ``GRID_EXP_ID`` at import time.  When grid mode is ``custom`` we always want
# a fresh slot instead of inheriting identifiers from the caller, so normalise
# the values before sourcing any helpers.  The early sanitised mode is reused
# later when deciding which code path to execute.
GRID_MODE_EARLY_RAW="${GRID_MODE-}"
GRID_MODE_EARLY_CLEAN=""
if [[ -n "${GRID_MODE_EARLY_RAW}" ]]; then
  GRID_MODE_EARLY_CLEAN="${GRID_MODE_EARLY_RAW//[[:space:]]/}"
  GRID_MODE_EARLY_CLEAN="${GRID_MODE_EARLY_CLEAN//\"/}"
  GRID_MODE_EARLY_CLEAN="${GRID_MODE_EARLY_CLEAN//\'/}"
fi

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="$(date +%s)"
fi

if [[ -z "${EXP_ID:-}" ]]; then
  EXP_ID="$RUN_ID"
fi

if [[ -z "${GRID_EXP_ID:-}" ]]; then
  GRID_EXP_ID="${EXP_ID}"
fi

if [[ "${GRID_MODE_EARLY_CLEAN}" == "custom" ]]; then
  if [[ "${EXP_ID}" != "${RUN_ID}" ]]; then
    EXP_ID="$RUN_ID"
  fi
  if [[ "${GRID_EXP_ID}" != "${RUN_ID}" ]]; then
    GRID_EXP_ID="$RUN_ID"
  fi
fi

export RUN_ID EXP_ID GRID_EXP_ID

export MJEPACI_STAGE="phase1"

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
  : "${WANDB_COUNT:=2}"
  : "${PHASE1_JEPA_COUNT:=${WANDB_COUNT}}"
  : "${PHASE1_CONTRAST_COUNT:=${WANDB_COUNT}}"

  export WANDB_RUN_GROUP="${GITHUB_RUN_ID:-pipeline-$(date -u +%Y%m%dT%H%M%SZ)}"
  export WANDB_RESUME=allow
  export WANDB_COUNT="$PHASE1_JEPA_COUNT"
  echo "[phase1] JEPA runs=${PHASE1_JEPA_COUNT} contrastive runs=${PHASE1_CONTRAST_COUNT} group=$WANDB_RUN_GROUP"
  if [[ -n "${SWEEP_CACHE_DIR:-}" ]]; then
    echo "[phase1] dataset cache root: $SWEEP_CACHE_DIR"
  fi

  JEPA_SPEC="$APP_DIR/sweeps/sweep_phase1_jepa.yaml"
  CONTRAST_SPEC="$APP_DIR/sweeps/sweep_phase1_contrastive.yaml"
  [[ -f "$JEPA_SPEC" ]] || { echo "[fatal] missing sweep spec $JEPA_SPEC" >&2; exit 1; }
  [[ -f "$CONTRAST_SPEC" ]] || { echo "[fatal] missing sweep spec $CONTRAST_SPEC" >&2; exit 1; }


  TMP_JEPA="$(mktemp)";      yq ".method = \"random\"" "$JEPA_SPEC" > "$TMP_JEPA"
  TMP_CONTRAST="$(mktemp)";  yq ".method = \"random\"" "$CONTRAST_SPEC" > "$TMP_CONTRAST"
  # WandB sweeps ignore a top-level `seed`; pairing is instead controlled via
  # PHASE1_BACKBONES / PHASE1_SEEDS so both specs enumerate identical combos.

  if [[ -n "${PHASE1_BACKBONES:-}" ]]; then
    export PHASE1_BACKBONES
    for spec in "$TMP_JEPA" "$TMP_CONTRAST"; do
      # Use -y to specify YAML output when editing in place
      yq -y -i --arg backbones "$PHASE1_BACKBONES" '.parameters.gnn_type.values = ($backbones
        | split(",")
        | map(gsub("^\\s+|\\s+$"; ""))
        | map(select(length > 0)))' "$spec"
    done
  fi

  if [[ -n "${PHASE1_SEEDS:-}" ]]; then
    export PHASE1_SEEDS
    for spec in "$TMP_JEPA" "$TMP_CONTRAST"; do
      # Use -y to specify YAML output when editing in place
      yq -y -i --arg seeds "$PHASE1_SEEDS" '.parameters.seed.values = ($seeds
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

  JEPA_SWEEP_ID="$(qualify_sweep_id "$JEPA_ID")"
  CONTRAST_SWEEP_ID="$(qualify_sweep_id "$CONTRAST_ID")"

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
      export SWEEP_ID="$JEPA_SWEEP_ID"
      export WANDB_COUNT="$PHASE1_JEPA_COUNT"
      echo "[phase1] launching JEPA agent for sweep $SWEEP_ID (count=$WANDB_COUNT)"
      if ! run_with_timeout wandb_agent; then
        rc=$?
        if [[ $rc -eq 2 ]]; then
          echo "[phase1][warn] JEPA agent returned rc=2; treating as sweep exhaustion"
        else
          exit "$rc"
        fi
      fi
    ) &
    PHASE1_JEPA_PID=$!

    (
      export LOG_DIR="${BASE_LOG_DIR}/phase1_contrastive"
      mkdir -p "$LOG_DIR"
      if [[ -n "${PHASE1_GPU_SPLITS[1]:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${PHASE1_GPU_SPLITS[1]}"
        echo "[phase1] contrastive agent using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      fi
      export SWEEP_ID="$CONTRAST_SWEEP_ID"
      export WANDB_COUNT="$PHASE1_CONTRAST_COUNT"
      echo "[phase1] launching contrastive agent for sweep $SWEEP_ID (count=$WANDB_COUNT)"
      if ! run_with_timeout wandb_agent; then
        rc=$?
        if [[ $rc -eq 2 ]]; then
          echo "[phase1][warn] contrastive agent returned rc=2; treating as sweep exhaustion"
        else
          exit "$rc"
        fi
      fi
    ) &
    PHASE1_CONTRAST_PID=$!

    set +e
    wait "$PHASE1_JEPA_PID"
    PHASE1_JEPA_RC=$?
    wait "$PHASE1_CONTRAST_PID"
    PHASE1_CONTRAST_RC=$?
    if [[ $PHASE1_JEPA_RC -eq 2 ]]; then
      echo "[phase1][warn] normalising JEPA agent rc=2 to success (sweep exhaustion)"
      PHASE1_JEPA_RC=0
    fi
    if [[ $PHASE1_CONTRAST_RC -eq 2 ]]; then
      echo "[phase1][warn] normalising contrastive agent rc=2 to success (sweep exhaustion)"
      PHASE1_CONTRAST_RC=0
    fi
    set -e

    if (( PHASE1_JEPA_RC != 0 || PHASE1_CONTRAST_RC != 0 )); then
      echo "[phase1][fatal] sweep agents failed: JEPA rc=$PHASE1_JEPA_RC contrastive rc=$PHASE1_CONTRAST_RC" >&2
      exit 1
    fi
  else
    export SWEEP_ID="$JEPA_SWEEP_ID"
    export WANDB_COUNT="$PHASE1_JEPA_COUNT"
    echo "[phase1] launching JEPA agent for sweep $SWEEP_ID (count=$WANDB_COUNT)"
    (
      if ! run_with_timeout wandb_agent; then
        rc=$?
        if [[ $rc -eq 2 ]]; then
          echo "[phase1][warn] JEPA agent returned rc=2; treating as sweep exhaustion"
        else
          exit "$rc"
        fi
      fi
    )

    export SWEEP_ID="$CONTRAST_SWEEP_ID"
    export WANDB_COUNT="$PHASE1_CONTRAST_COUNT"
    echo "[phase1] launching contrastive agent for sweep $SWEEP_ID (count=$WANDB_COUNT)"
    (
      if ! run_with_timeout wandb_agent; then
        rc=$?
        if [[ $rc -eq 2 ]]; then
          echo "[phase1][warn] contrastive agent returned rc=2; treating as sweep exhaustion"
        else
          exit "$rc"
        fi
      fi
    )
  fi

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

  if ! ensure_micromamba; then
    echo "[phase1][fatal] micromamba unavailable; set MMBIN or install micromamba" >&2
    exit 1
  fi

  "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
    python -u "$APP_DIR/scripts/ci/paired_effect_from_wandb.py" \
      --project "${WANDB_PROJECT}" \
      --group   "${WANDB_RUN_GROUP}" \
      --aggregate pair-seed \
      --seed "${CI_SEED:-42}" \
      --strict \
      --sweep "$JEPA_ID" \
      --sweep "$CONTRAST_ID" \
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
      elif [[ "$PE_DECISION_STATUS" == "tied-primary" ]]; then
        echo "[phase1][warn] paired-effect primary metric within tie tolerance; defaulting to ${METHOD_WINNER}"
        if tie_note=$("$py" - "$PE_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
info = payload.get("tiebreaker_metric") or {}
label = info.get("label") or info.get("canonical") or "tie-breaker metric"
counts = info.get("counts") or {}
reasons = info.get("unavailable_reasons") or []
extra = []
for method in ("jepa", "contrastive"):
    method_counts = counts.get(method) or {}
    raw = method_counts.get("raw") or 0
    valid = method_counts.get("valid") or 0
    if raw and valid and raw != valid:
        extra.append(f"{method} discarded {raw - valid} non-finite value(s)")
    elif raw and not valid:
        extra.append(f"{method} recorded only NaNs ({raw} value(s))")
    elif not raw:
        extra.append(f"{method} missing {label}")
notes = reasons if reasons else extra
if notes:
    print(f"{label}: " + "; ".join(notes))
PY
        ); then
          if [[ -n "$tie_note" ]]; then
            echo "[phase1][info] tie-breaker diagnostics: $tie_note"
          fi
        else
          echo "[phase1][warn] failed to summarise tie-breaker diagnostics" >&2
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

  phase1_force_reuse="$(normalize_bool "${CI_PHASE1_FORCE_REUSE_GRID:-0}" 0)"
  if [[ -n "${EXP_ID:-}" && "${GRID_EXP_ID:-}" != "${EXP_ID}" ]]; then
    local reuse_old=0
    if (( FROZEN )) && [[ "${CI_FORCE_UNFREEZE_GRID}" != "1" ]]; then
      reuse_old=1
    elif (( phase1_force_reuse )); then
      reuse_old=1
    fi
    if (( reuse_old )); then
      echo "[phase1] reusing existing grid lineage ${GRID_EXP_ID} (EXP_ID=${EXP_ID})" >&2
    else
      local old_grid_id="${GRID_EXP_ID:-}"
      GRID_EXP_ID="$EXP_ID"
      export GRID_EXP_ID
      if declare -F ci_phase2_refresh_lineage_bindings >/dev/null 2>&1; then
        ci_phase2_refresh_lineage_bindings "" "$GRID_EXP_ID" "" "$old_grid_id"
      fi
      if [[ -n "${EXPERIMENTS_ROOT:-}" ]]; then
        local new_grid_root="${EXPERIMENTS_ROOT%/}/${GRID_EXP_ID}/grid"
        GRID_DIR="$new_grid_root"
        export GRID_DIR
        GRID_SOURCE_DIR="$new_grid_root"
        export GRID_SOURCE_DIR
      fi
    fi
  fi

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
        --metrics-csv "${GRID_DIR:-$APP_DIR/grid}/phase1_runs.csv" \
        --winner-csv "${GRID_DIR:-$APP_DIR/grid}/phase2_winner_config.csv" \
        --include-sweep "${WANDB_ENTITY}/${WANDB_PROJECT}/${JEPA_ID}" \
        --include-sweep "${WANDB_ENTITY}/${WANDB_PROJECT}/${CONTRAST_ID}" \
        --out "$TMP_BEST" \
        --phase2-yaml "$TMP_PHASE2" \
        --phase2-unlabeled-dir "${PHASE2_UNLABELED_DIR:-${CACHE_DIR:-$APP_DIR/cache/graphs_10m}}" \
        --phase2-labeled-dir   "${PHASE2_LABELED_DIR:-$APP_DIR/data/katielinkmoleculenet_benchmark/train}" \
      2>&1 | tee "$LOG_TMP"

  mkdir -p "$(dirname "$OUT_PATH")"
  cp "$TMP_BEST" "$OUT_PATH"
  echo "[phase1] staged best config to $OUT_PATH"

  FINAL_CFG_DEFAULT="${GRID_DIR:-$APP_DIR/grid}/best_grid_config.json"
  FINAL_CFG="${EXPORT_OUT_PATH:-$FINAL_CFG_DEFAULT}"
  if [[ "$FINAL_CFG" != "$OUT_PATH" ]]; then
    if mkdir -p "$(dirname "$FINAL_CFG")" 2>/dev/null && cp "$TMP_BEST" "$FINAL_CFG" 2>/dev/null; then
      echo "[phase1] exported best config to $FINAL_CFG"
    else
      echo "[phase1][warn] unable to export best config to $FINAL_CFG; falling back to $FINAL_CFG_DEFAULT" >&2
      mkdir -p "$(dirname "$FINAL_CFG_DEFAULT")"
      cp "$TMP_BEST" "$FINAL_CFG_DEFAULT"
      if [[ "$FINAL_CFG_DEFAULT" != "$OUT_PATH" ]]; then
        echo "[phase1] exported best config to $FINAL_CFG_DEFAULT"
      fi
    fi
  fi

  CANONICAL_P2="${GRID_DIR:-$APP_DIR/grid}/grid_sweep_phase2.yaml"
  mkdir -p "$(dirname "$CANONICAL_P2")"
  cp "$TMP_PHASE2" "$CANONICAL_P2"
  echo "[phase1] staged Phase-2 sweep YAML to $CANONICAL_P2"

  FINAL_P2="${EXPORT_PHASE2_PATH:-$CANONICAL_P2}"
  if [[ "$FINAL_P2" != "$CANONICAL_P2" ]]; then
    if mkdir -p "$(dirname "$FINAL_P2")" 2>/dev/null && cp "$TMP_PHASE2" "$FINAL_P2" 2>/dev/null; then
      echo "[phase1] exported Phase-2 sweep YAML to $FINAL_P2"
    else
      echo "[phase1][warn] unable to export Phase-2 sweep YAML to $FINAL_P2; falling back to $CANONICAL_P2" >&2
      FINAL_P2="$CANONICAL_P2"
    fi
  fi

  SWEEP_ID2="$(wandb_sweep_create "$FINAL_P2")"
  [[ "$SWEEP_ID2" =~ ^[a-z0-9]{8}$ ]] || { echo "[phase2][fatal] bad sweep id: '$SWEEP_ID2'" >&2; exit 1; }
  echo -n "$SWEEP_ID2" > "${GRID_DIR:-$APP_DIR/grid}/phase2_sweep_id.txt"
  echo "[phase2] created sweep: $SWEEP_ID2  (saved to $GRID_DIR/phase2_sweep_id.txt)"

else
  echo "[grid] running custom grid-search"

  # In custom mode we always provision a fresh experiment/grid slot that is
  # anchored to the current RUN_ID.  Callers may still export PRETRAIN_EXP_ID so
  # dependent stages can reuse the frozen encoder lineage, but Phase-1 should
  # never inherit the previous grid identifiers in this scenario.  Normalise all
  # identifiers to the fresh run slot before invoking any helpers so directory
  # resolution stays consistent across both the stage shim and the default Vast
  # execution path.
  fresh_id="${RUN_ID:-}"
  if [[ -z "$fresh_id" ]]; then
    fresh_id="$(date +%s)"
    RUN_ID="$fresh_id"
    EXP_ID="$fresh_id"
    GRID_EXP_ID="$fresh_id"
    export RUN_ID EXP_ID GRID_EXP_ID
  fi

  if [[ -n "${MJEPACI_STAGE_SHIM:-}" && -x "${MJEPACI_STAGE_SHIM}" ]]; then
    # When a shim is available we want to force the execution through it rather
    # than attempting to launch remote agents.  ``run_stage`` already promotes
    # the shim, but recording an explicit log line here makes the behaviour
    # obvious when debugging CI output.
    echo "[grid] invoking stage shim for grid_search (OUT_DIR will be canonical)"
  fi

  run_stage grid_search

  # When running against a shim the stage may emit bookkeeping files, but in
  # real custom runs we still want to persist the identifiers that were
  # allocated for the invocation.  Record them explicitly so test doubles and
  # downstream tooling can inspect which experiment slot was provisioned.
  # Prefer the stage directory used by the shim (OUT_DIR) and fall back to the
  # canonical location derived from the run id if the environment variables are
  # unset.  This keeps custom runs predictable even when EXP_ID/GRID_EXP_ID
  # haven't been exported by the caller yet (the shim is still able to populate
  # them before we persist the bookkeeping files).
  grid_stage_dir="${OUT_DIR:-}"
  if [[ -z "${grid_stage_dir}" ]]; then
    grid_stage_dir="$(stage_dir grid_search)"
  fi
  exp_bookkeeping="${EXP_ID:-${RUN_ID:-}}"
  canonical_stage_dir="${EXPERIMENTS_ROOT%/}/${exp_bookkeeping}/grid_search"

  if [[ -z "${grid_stage_dir}" || ! -d "${grid_stage_dir}" ]]; then
    # Ensure we still create the expected hierarchy under the experiments root
    # so downstream tests can discover the allocated slot.  Always create the
    # canonical path in case the shim used a different OUT_DIR.
    grid_stage_dir="${canonical_stage_dir}"
  fi

  grid_stage_outputs="${grid_stage_dir}/stage-outputs"
  mkdir -p "${grid_stage_outputs}"

  grid_exp_bookkeeping="${GRID_EXP_ID:-${exp_bookkeeping}}"

  printf '%s' "${exp_bookkeeping}" > "${grid_stage_outputs}/exp_id.txt"
  printf '%s' "${grid_exp_bookkeeping}" > "${grid_stage_outputs}/grid_exp_id.txt"

  if [[ "${grid_stage_dir}" != "${canonical_stage_dir}" ]]; then
    mkdir -p "${canonical_stage_dir}/stage-outputs"
    cp "${grid_stage_outputs}/exp_id.txt" "${canonical_stage_dir}/stage-outputs/exp_id.txt"
    cp "${grid_stage_outputs}/grid_exp_id.txt" "${canonical_stage_dir}/stage-outputs/grid_exp_id.txt"
  fi
fi
