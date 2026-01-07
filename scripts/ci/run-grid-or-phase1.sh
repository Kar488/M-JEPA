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

# Enforce paired seeds/backbones for Phase-1 so paired-effect analysis sees
# enough overlapping (pair_id, seed) combinations without changing core
# training hyperparameters. When PHASE1_BACKBONES is unset, default to the
# values already encoded in the sweep specs so the resulting sweeps visibly
# enumerate every backbone.
PHASE1_MIN_SEED_PAIRS=${PHASE1_MIN_SEED_PAIRS:-2}
PHASE1_BACKBONES=${PHASE1_BACKBONES-}

# When callers forget to set CUDA_VISIBLE_DEVICES, expose all visible GPUs so
# the splitter can divide them between JEPA and contrastive agents.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  mapfile -t _phase1_all_gpus < <(visible_gpu_ids)
  if (( ${#_phase1_all_gpus[@]} > 0 )); then
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${_phase1_all_gpus[*]}")"
    echo "[phase1] exposing all visible GPUs via CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
  unset _phase1_all_gpus
fi

# Decide which mode to use
: "${GRID_MODE:?GRID_MODE must be set to 'wandb' or 'custom'}"
# allowed values: custom | wandb

# Normalize GRID_MODE (strip spaces and quotes)
GRID_MODE_CLEAN="${GRID_MODE//[[:space:]]/}"
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\"/}"
GRID_MODE_CLEAN="${GRID_MODE_CLEAN//\'/}"

phase1_force_reuse="$(normalize_bool "${CI_PHASE1_FORCE_REUSE_GRID:-0}" 0)"

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

ci_phase1_locate_existing_grid() {
  local out_id_var="$1" out_dir_var="$2"
  local root="${EXPERIMENTS_ROOT%/}"
  local -a candidates=()
  candidates+=("${GRID_SOURCE_DIR:-}" "${GRID_DIR:-}")
  if [[ -n "$root" ]]; then
    if [[ -n "${GRID_EXP_ID:-}" ]]; then
      candidates+=("${root}/${GRID_EXP_ID}/grid")
    fi
    if [[ -n "${EXP_ID:-}" ]]; then
      candidates+=("${root}/${EXP_ID}/grid")
    fi
  fi

  local candidate=""
  local best_dir="" best_id=""
  declare -A seen_candidates=()
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    candidate="${candidate%/}"
    if [[ -n "${seen_candidates[$candidate]:-}" ]]; then
      continue
    fi
    seen_candidates[$candidate]=1
    if [[ -f "${candidate}/grid_sweep_phase2.yaml" || -f "${candidate}/phase2_sweep_id.txt" || -d "${candidate}/phase1_export" ]]; then
      best_dir="$candidate"
      best_id="$(basename "$(dirname "$candidate")")"
      break
    fi
  done

  printf -v "$out_id_var" '%s' "$best_id"
  printf -v "$out_dir_var" '%s' "$best_dir"
  [[ -n "$best_dir" ]]
}

phase1_reuse_requested=0
phase1_reuse_reason=""
if (( FROZEN )) && [[ "${CI_FORCE_UNFREEZE_GRID}" != "1" ]]; then
  phase1_reuse_requested=1
  phase1_reuse_reason="frozen lineage"
fi
if (( phase1_force_reuse )); then
  phase1_reuse_requested=1
  if [[ -n "$phase1_reuse_reason" ]]; then
    phase1_reuse_reason+=", CI_PHASE1_FORCE_REUSE_GRID=1"
  else
    phase1_reuse_reason="CI_PHASE1_FORCE_REUSE_GRID=1"
  fi
fi

if (( phase1_reuse_requested )); then
  reuse_grid_id=""
  reuse_grid_dir=""
  if ci_phase1_locate_existing_grid reuse_grid_id reuse_grid_dir; then
    GRID_DIR="$reuse_grid_dir"
    GRID_SOURCE_DIR="$reuse_grid_dir"
    export GRID_DIR GRID_SOURCE_DIR
    if [[ -n "$reuse_grid_id" ]]; then
      GRID_EXP_ID="$reuse_grid_id"
      export GRID_EXP_ID
    fi
    echo "[phase1] skipping Phase-1 sweep (${phase1_reuse_reason}) using existing grid at ${reuse_grid_dir}" >&2
    exit 0
  fi

  if (( FROZEN )) && [[ "${CI_FORCE_UNFREEZE_GRID}" != "1" ]]; then
    echo "[phase1][fatal] ${phase1_reuse_reason:-reuse requested} but no completed Phase-1 grid found under ${GRID_SOURCE_DIR:-<unset>} or ${EXPERIMENTS_ROOT:-<unset>}" >&2
    exit 1
  fi

  echo "[phase1][warn] ${phase1_reuse_reason:-reuse requested} but no prior Phase-1 grid detected; proceeding with fresh sweeps" >&2
fi

if [[ "$GRID_MODE_CLEAN" == "wandb" ]]; then
  echo "[grid] running wandb sweep agent"

  require_cmd perl
  require_cmd sed
  require_cmd dos2unix
  require_cmd yq
  require_cmd jq

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


  TMP_BASE_JEPA="$(mktemp)";      yq ".method = \"random\"" "$JEPA_SPEC" > "$TMP_BASE_JEPA"
  TMP_BASE_CONTRAST="$(mktemp)";  yq ".method = \"random\"" "$CONTRAST_SPEC" > "$TMP_BASE_CONTRAST"

  BACKBONES=()
  BACKBONE_SOURCE="spec"
  if [[ -n "${PHASE1_BACKBONES}" ]]; then
    BACKBONE_SOURCE="env"
    IFS="," read -ra RAW_BACKBONES <<< "$PHASE1_BACKBONES"
    for raw in "${RAW_BACKBONES[@]}"; do
      candidate="${raw//[[:space:]]/}"
      if [[ -n "$candidate" ]]; then
        BACKBONES+=("$candidate")
      fi
    done
  else
    mapfile -t BACKBONES < <(yq '.parameters.gnn_type.values[]' "$TMP_BASE_JEPA")
  fi

  if (( ${#BACKBONES[@]} == 0 )); then
    echo "[phase1][fatal] no backbones resolved (PHASE1_BACKBONES='${PHASE1_BACKBONES:-<unset>}' spec=${JEPA_SPEC})" >&2
    exit 1
  fi

  # Harmonise seeds across methods and backbones, padding to ensure paired-effect
  # sees enough (pair_id, seed) intersections.
  PHASE1_SEED_LIST=()
  if [[ -n "${PHASE1_SEEDS:-}" ]]; then
    IFS="," read -ra RAW_SEEDS <<< "$PHASE1_SEEDS"
    for seed in "${RAW_SEEDS[@]}"; do
      token="${seed//[[:space:]]/}"
      if [[ -n "$token" ]]; then
        PHASE1_SEED_LIST+=("$token")
      fi
    done
  else
    mapfile -t PHASE1_SEED_LIST < <(yq '.parameters.seed.values[]' "$TMP_BASE_JEPA")
  fi

  if (( ${#PHASE1_SEED_LIST[@]} < PHASE1_MIN_SEED_PAIRS )); then
    echo "[phase1][warn] extending PHASE1_SEEDS to guarantee at least ${PHASE1_MIN_SEED_PAIRS} paired seeds" >&2
    next_seed=0
    while (( ${#PHASE1_SEED_LIST[@]} < PHASE1_MIN_SEED_PAIRS )); do
      candidate="$next_seed"
      duplicate=0
      for existing in "${PHASE1_SEED_LIST[@]}"; do
        if [[ "$existing" == "$candidate" ]]; then
          duplicate=1
          break
        fi
      done
      if (( ! duplicate )); then
        PHASE1_SEED_LIST+=("$candidate")
      fi
      ((next_seed++))
    done
  fi

  SEEDS_CSV="$(IFS=","; echo "${PHASE1_SEED_LIST[*]}")"

  TMP_JEPA_SPECS=()
  TMP_CONTRAST_SPECS=()
  SWEEP_BACKBONES=()

  echo "[phase1] using backbones from ${BACKBONE_SOURCE}: ${BACKBONES[*]}"

  TMP_SEEDED_JEPA="$(mktemp)"; cp "$TMP_BASE_JEPA" "$TMP_SEEDED_JEPA"
  TMP_SEEDED_CONTRAST="$(mktemp)"; cp "$TMP_BASE_CONTRAST" "$TMP_SEEDED_CONTRAST"

  for spec in "$TMP_SEEDED_JEPA" "$TMP_SEEDED_CONTRAST"; do
    yq -y -i --arg seeds "$SEEDS_CSV" '.parameters.seed.values = ($seeds
      | split(",")
      | map(gsub("^\\s+|\\s+$"; ""))
      | map(select(length > 0))
      | map(tonumber))' "$spec"
  done

  # When callers explicitly pin PHASE1_BACKBONES to a single backbone, split
  # sweeps so each agent sticks to that architecture. If multiple backbones are
  # provided, keep them in a shared sweep so W&B can sample all candidates
  # instead of silently clamping to the first entry.
  if [[ "$BACKBONE_SOURCE" == "env" && ${#BACKBONES[@]} -gt 1 ]]; then
    for spec in "$TMP_SEEDED_JEPA" "$TMP_SEEDED_CONTRAST"; do
      yq -y -i --argjson backbones "$(printf '%s\n' "${BACKBONES[@]}" | jq -R . | jq -s .)" \
        '.parameters.gnn_type.values = $backbones' "$spec"
    done
  fi

  if [[ "$BACKBONE_SOURCE" == "env" && ${#BACKBONES[@]} -eq 1 ]]; then
    for backbone in "${BACKBONES[@]}"; do
      TMP_JEPA="$(mktemp)"; cp "$TMP_SEEDED_JEPA" "$TMP_JEPA"
      TMP_CONTRAST="$(mktemp)"; cp "$TMP_SEEDED_CONTRAST" "$TMP_CONTRAST"

      for spec in "$TMP_JEPA" "$TMP_CONTRAST"; do
        yq -y -i --arg backbone "$backbone" '.parameters.gnn_type.values = [$backbone]' "$spec"
      done

      check_shared_equal "$TMP_JEPA" "$TMP_CONTRAST"

      TMP_JEPA_SPECS+=("$TMP_JEPA")
      TMP_CONTRAST_SPECS+=("$TMP_CONTRAST")
      SWEEP_BACKBONES+=("$backbone")
    done
  else
    # Preserve the full backbone sweep defined in the YAML spec (or the env
    # filtered list above).
    check_shared_equal "$TMP_SEEDED_JEPA" "$TMP_SEEDED_CONTRAST"

    TMP_JEPA_SPECS+=("$TMP_SEEDED_JEPA")
    TMP_CONTRAST_SPECS+=("$TMP_SEEDED_CONTRAST")
    SWEEP_BACKBONES+=("multi")
  fi

  if [[ "${PHASE1_DRYRUN_SPEC_ONLY:-0}" == "1" ]]; then
    DRYRUN_OUT_ROOT="${PHASE1_DRYRUN_OUTPUT_DIR:-${APP_DIR}/logs/phase1_spec_dryrun}"
    mkdir -p "$DRYRUN_OUT_ROOT"

    for idx in "${!TMP_JEPA_SPECS[@]}"; do
      backbone="${SWEEP_BACKBONES[$idx]}"
      backbone_slug="${backbone//[^a-zA-Z0-9_-]/}"
      if [[ -z "$backbone_slug" ]]; then
        backbone_slug="b${idx}"
      fi

      cp "${TMP_JEPA_SPECS[$idx]}" "${DRYRUN_OUT_ROOT}/jepa_${backbone_slug}.yaml"
      cp "${TMP_CONTRAST_SPECS[$idx]}" "${DRYRUN_OUT_ROOT}/contrastive_${backbone_slug}.yaml"
      echo "[phase1][dryrun] captured specs for backbone=${backbone} into ${DRYRUN_OUT_ROOT}" >&2
    done

    exit 0
  fi

  JEPA_IDS=()
  CONTRAST_IDS=()

  for idx in "${!TMP_JEPA_SPECS[@]}"; do
    JEPA_ID="$(wandb_sweep_create "${TMP_JEPA_SPECS[$idx]}")"
    CONTRAST_ID="$(wandb_sweep_create "${TMP_CONTRAST_SPECS[$idx]}")"
    if [[ ! "$JEPA_ID" =~ ^[a-z0-9]{8}$ ]] || [[ ! "$CONTRAST_ID" =~ ^[a-z0-9]{8}$ ]]; then
      echo "[phase1][fatal] bad sweep ids for backbones ${SWEEP_BACKBONES[*]}: JEPA_ID='$JEPA_ID' CONTRAST_ID='$CONTRAST_ID'" >&2
      exit 1
    fi
    echo "[phase1] (${SWEEP_BACKBONES[$idx]}) JEPA sweep id=$JEPA_ID  contrastive sweep id=$CONTRAST_ID"

    JEPA_IDS+=("$JEPA_ID")
    CONTRAST_IDS+=("$CONTRAST_ID")
  done

  cd "$APP_DIR"
  BASE_LOG_DIR="${LOG_DIR:-$APP_DIR/logs}"
  mapfile -t GRID_VISIBLE_GPUS < <(visible_gpu_ids)
  PHASE1_GPU_COUNT="${#GRID_VISIBLE_GPUS[@]}"
  PHASE1_GRACEFUL_STOP=0

  phase1_check_graceful_stop() {
    local dir="${1:-}"
    if [[ -n "$dir" ]] && was_graceful_stop "wandb_agent" "$dir"; then
      PHASE1_GRACEFUL_STOP=1
      return 0
    fi
    return 1
  }

  run_backbone_agents() {
    local jepa_id="$1" contrast_id="$2" backbone="$3"

    local jepa_sweep contrast_sweep
    jepa_sweep="$(qualify_sweep_id "$jepa_id")"
    contrast_sweep="$(qualify_sweep_id "$contrast_id")"

    local backbone_slug
    backbone_slug="${backbone//[^a-zA-Z0-9_-]/}"
    if [[ -z "$backbone_slug" ]]; then
      backbone_slug="$backbone"
    fi

    local jepa_log_dir="${BASE_LOG_DIR}/phase1_jepa_${backbone_slug}"
    local contrast_log_dir="${BASE_LOG_DIR}/phase1_contrastive_${backbone_slug}"

    if (( PHASE1_GPU_COUNT >= 2 )); then
      declare -a PHASE1_GPU_SPLITS
      split_gpu_ids PHASE1_GPU_SPLITS 2 "${GRID_VISIBLE_GPUS[@]}"

      (
        export LOG_DIR="$jepa_log_dir"
        mkdir -p "$LOG_DIR"
        if [[ -n "${PHASE1_GPU_SPLITS[0]:-}" ]]; then
          export CUDA_VISIBLE_DEVICES="${PHASE1_GPU_SPLITS[0]}"
          echo "[phase1] (${backbone}) JEPA agent using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        fi
        export SWEEP_ID="$jepa_sweep"
        export WANDB_COUNT="$PHASE1_JEPA_COUNT"
        echo "[phase1] (${backbone}) launching JEPA agent for sweep $SWEEP_ID (count=$WANDB_COUNT)"
        if ! run_with_timeout wandb_agent; then
          rc=$?
          if [[ $rc -eq 2 ]]; then
            echo "[phase1][warn] (${backbone}) JEPA agent returned rc=2; treating as sweep exhaustion"
          else
            exit "$rc"
          fi
        fi
      ) &
      PHASE1_JEPA_PID=$!

      (
        export LOG_DIR="$contrast_log_dir"
        mkdir -p "$LOG_DIR"
        if [[ -n "${PHASE1_GPU_SPLITS[1]:-}" ]]; then
          export CUDA_VISIBLE_DEVICES="${PHASE1_GPU_SPLITS[1]}"
          echo "[phase1] (${backbone}) contrastive agent using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        fi
        export SWEEP_ID="$contrast_sweep"
        export WANDB_COUNT="$PHASE1_CONTRAST_COUNT"
        echo "[phase1] (${backbone}) launching contrastive agent for sweep $SWEEP_ID (count=$WANDB_COUNT)"
        if ! run_with_timeout wandb_agent; then
          rc=$?
          if [[ $rc -eq 2 ]]; then
            echo "[phase1][warn] (${backbone}) contrastive agent returned rc=2; treating as sweep exhaustion"
          else
            exit "$rc"
          fi
        fi
      ) &
      PHASE1_CONTRAST_PID=$!

      set +e
      wait "$PHASE1_JEPA_PID"; PHASE1_JEPA_RC=$?
      wait "$PHASE1_CONTRAST_PID"; PHASE1_CONTRAST_RC=$?
      if [[ $PHASE1_JEPA_RC -eq 2 ]]; then
        echo "[phase1][warn] (${backbone}) normalising JEPA agent rc=2 to success (sweep exhaustion)"
        PHASE1_JEPA_RC=0
      fi
      if [[ $PHASE1_CONTRAST_RC -eq 2 ]]; then
        echo "[phase1][warn] (${backbone}) normalising contrastive agent rc=2 to success (sweep exhaustion)"
        PHASE1_CONTRAST_RC=0
      fi
      set -e

      if (( PHASE1_JEPA_RC != 0 || PHASE1_CONTRAST_RC != 0 )); then
        echo "[phase1][fatal] (${backbone}) sweep agents failed: JEPA rc=$PHASE1_JEPA_RC contrastive rc=$PHASE1_CONTRAST_RC" >&2
        exit 1
      fi

      phase1_check_graceful_stop "$jepa_log_dir" || true
      phase1_check_graceful_stop "$contrast_log_dir" || true
      if (( PHASE1_GRACEFUL_STOP )); then
        return 0
      fi
    else
      export LOG_DIR="$jepa_log_dir"
      mkdir -p "$LOG_DIR"
      export SWEEP_ID="$jepa_sweep"
      export WANDB_COUNT="$PHASE1_JEPA_COUNT"
      echo "[phase1] (${backbone}) launching JEPA agent for sweep $SWEEP_ID (count=$WANDB_COUNT)"
      (
        if ! run_with_timeout wandb_agent; then
          rc=$?
          if [[ $rc -eq 2 ]]; then
            echo "[phase1][warn] (${backbone}) JEPA agent returned rc=2; treating as sweep exhaustion"
          else
            exit "$rc"
          fi
        fi
      )
      phase1_check_graceful_stop "$jepa_log_dir" || true
      if (( PHASE1_GRACEFUL_STOP )); then
        return 0
      fi

      export LOG_DIR="$contrast_log_dir"
      mkdir -p "$LOG_DIR"
      export SWEEP_ID="$contrast_sweep"
      export WANDB_COUNT="$PHASE1_CONTRAST_COUNT"
      echo "[phase1] (${backbone}) launching contrastive agent for sweep $SWEEP_ID (count=$WANDB_COUNT)"
      (
        if ! run_with_timeout wandb_agent; then
          rc=$?
          if [[ $rc -eq 2 ]]; then
            echo "[phase1][warn] (${backbone}) contrastive agent returned rc=2; treating as sweep exhaustion"
          else
            exit "$rc"
          fi
        fi
      )
      phase1_check_graceful_stop "$contrast_log_dir" || true
      if (( PHASE1_GRACEFUL_STOP )); then
        return 0
      fi
    fi
  }

  for idx in "${!JEPA_IDS[@]}"; do
    run_backbone_agents "${JEPA_IDS[$idx]}" "${CONTRAST_IDS[$idx]}" "${SWEEP_BACKBONES[$idx]}"
    if (( PHASE1_GRACEFUL_STOP )); then
      break
    fi
  done

  if (( PHASE1_GRACEFUL_STOP )); then
    echo "[phase1] info: phase1 sweep cancelled; skipping artifact exports." >&2
    exit 0
  fi

  # Require that paired-effect analysis only considers runs that have reached
  # the minimum training budgets that Phase-1 sweeps schedule.  Allow
  # overrides via environment variables so CI callers can tighten or loosen the
  # thresholds without editing this script, and align pretrain thresholds with
  # the sweep's actual batch budgets to avoid early-terminated runs skewing
  # comparisons.
  if [[ -z "${PE_MIN_PRETRAIN_EPOCHS+x}" ]]; then
    PE_MIN_PRETRAIN_EPOCHS=5
  fi
  if [[ -z "${PE_MIN_FINETUNE_EPOCHS+x}" ]]; then
    PE_MIN_FINETUNE_EPOCHS=1
  fi

  : "${PE_MIN_PRETRAIN_BATCHES_FRAC:=0.8}"
  if [[ -z "${PE_MIN_PRETRAIN_BATCHES+x}" && -n "${PHASE1_MAX_PRETRAIN_BATCHES:-}" ]]; then
    PE_MIN_PRETRAIN_BATCHES=$(PHASE1_MAX_PRETRAIN_BATCHES="$PHASE1_MAX_PRETRAIN_BATCHES" \
      PE_MIN_PRETRAIN_BATCHES_FRAC="$PE_MIN_PRETRAIN_BATCHES_FRAC" python - <<'PY'
import math
import os
frac = float(os.environ.get("PE_MIN_PRETRAIN_BATCHES_FRAC", "0.8"))
budget = float(os.environ.get("PHASE1_MAX_PRETRAIN_BATCHES", "0"))
print(int(math.floor(budget * frac)))
PY
    )
  elif [[ -z "${PE_MIN_PRETRAIN_BATCHES+x}" ]]; then
    max_pretrain_batches=$(yq '.parameters.max_pretrain_batches.value // (.parameters.max_pretrain_batches.values[0] // 0)' "${TMP_JEPA_SPECS[0]}")
    computed_min_pretrain_batches=$(MAX_PRETRAIN_BATCHES="$max_pretrain_batches" PE_MIN_PRETRAIN_BATCHES_FRAC="$PE_MIN_PRETRAIN_BATCHES_FRAC" python - <<'PY'
import math
import os
max_batches = int(os.environ.get("MAX_PRETRAIN_BATCHES", "0"))
frac = float(os.environ.get("PE_MIN_PRETRAIN_BATCHES_FRAC", "0.8"))
print(int(math.floor(max_batches * frac)))
PY
    )
    PE_MIN_PRETRAIN_BATCHES="$computed_min_pretrain_batches"
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

  PE_SWEEP_FLAGS=()
  for sweep_id in "${JEPA_IDS[@]}" "${CONTRAST_IDS[@]}"; do
    PE_SWEEP_FLAGS+=("--sweep" "$sweep_id")
  done

  "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
    python -u "$APP_DIR/scripts/ci/paired_effect_from_wandb.py" \
      --project "${WANDB_PROJECT}" \
      --group   "${WANDB_RUN_GROUP}" \
      --aggregate pair-seed \
      --seed "${CI_SEED:-42}" \
      --strict \
      "${PE_SWEEP_FLAGS[@]}" \
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
  if [[ "$WINNER" == "contrastive" ]]; then
    WINNER_SWEEPS=("${CONTRAST_IDS[@]}")
  else
    WINNER_SWEEPS=("${JEPA_IDS[@]}")
  fi
  BEST_ID="${WINNER_SWEEPS[0]}"
  BEST_SWEEP="${WANDB_ENTITY}/${WANDB_PROJECT}/${BEST_ID}"

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
  phase1_stage_outputs="${GRID_DIR:-$APP_DIR/grid}/phase1_export/stage-outputs"
  mkdir -p "$phase1_stage_outputs"
  EXPORT_INCLUDE_SWEEPS=()
  for sweep_id in "${JEPA_IDS[@]}" "${CONTRAST_IDS[@]}"; do
    EXPORT_INCLUDE_SWEEPS+=("--include-sweep" "${WANDB_ENTITY}/${WANDB_PROJECT}/${sweep_id}")
  done

  PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -u "$APP_DIR/scripts/ci/export_best_from_wandb.py" \
        --sweep-id "$BEST_SWEEP" \
        --task "$TASK_FROM_PE" \
        --phase2-method bayes \
        --emit-bounds \
        --metrics-csv "${phase1_stage_outputs}/phase1_runs.csv" \
        --winner-csv "${phase1_stage_outputs}/phase2_winner_config.csv" \
        "${EXPORT_INCLUDE_SWEEPS[@]}" \
        --out "$TMP_BEST" \
        --phase2-yaml "$TMP_PHASE2" \
        --phase2-unlabeled-dir "${PHASE2_UNLABELED_DIR:-${DATA_ROOT:-$APP_DIR}/data/ZINC-canonicalized}" \
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
