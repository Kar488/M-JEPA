#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

# Use the same helpers layer you already use elsewhere
# (stage.sh gives you the timeout + logging wrappers)
source "$(dirname "$0")/stage.sh"   # provides run_with_timeout/wandb_agent branch:contentReference[oaicite:0]{index=0}
: "${WANDB_RUN_GROUP:=${GITHUB_RUN_ID:-phase2-$(date -u +%Y%m%dT%H%M%SZ)}}"
export WANDB_RUN_GROUP

SWEEP_ID_FILE="${GRID_DIR:-$APP_DIR/grid}/phase2_sweep_id.txt"
if [[ ! -f "$SWEEP_ID_FILE" ]]; then
  echo "[phase2][fatal] sweep id file not found: $SWEEP_ID_FILE" >&2
  exit 1
fi

# 1) Run the Phase-2 (Bayes) sweep agent
export WANDB_SWEEP_ID2="$(cat "$SWEEP_ID_FILE")"
export SWEEP_ID="${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_SWEEP_ID2}"  # stage.sh reads SWEEP_ID
: "${WANDB_COUNT:=100}"                                                # default trials
PHASE2_TOTAL_COUNT="$WANDB_COUNT"

: "${PHASE2_LABELED_DIR:?[phase2] PHASE2_LABELED_DIR not set}"
: "${PHASE2_UNLABELED_DIR:?[phase2] PHASE2_UNLABELED_DIR not set}"
[[ -d "$PHASE2_LABELED_DIR"   ]] || { echo "[phase2][fatal] not a dir: $PHASE2_LABELED_DIR"; exit 2; }
[[ -d "$PHASE2_UNLABELED_DIR" ]] || { echo "[phase2][fatal] not a dir: $PHASE2_UNLABELED_DIR"; exit 2; }
echo "[phase2] labeled=$PHASE2_LABELED_DIR  unlabeled=$PHASE2_UNLABELED_DIR"


echo "[phase2] using sweep: $SWEEP_ID"

BASE_LOG_DIR="${LOG_DIR:-$APP_DIR/logs}"
mapfile -t GRID_VISIBLE_GPUS < <(visible_gpu_ids)
PHASE2_GPU_COUNT="${#GRID_VISIBLE_GPUS[@]}"

PHASE2_AGENT_WORKERS=1
if [[ -n "${PHASE2_AGENT_COUNT:-}" ]]; then
  if [[ "${PHASE2_AGENT_COUNT}" =~ ^[0-9]+$ ]]; then
    PHASE2_AGENT_WORKERS="${PHASE2_AGENT_COUNT}"
  else
    echo "[phase2][warn] ignoring non-numeric PHASE2_AGENT_COUNT='${PHASE2_AGENT_COUNT}'"
    PHASE2_AGENT_WORKERS=1
  fi
elif (( PHASE2_GPU_COUNT > 1 )); then
  PHASE2_AGENT_WORKERS="$PHASE2_GPU_COUNT"
fi

if (( PHASE2_GPU_COUNT > 0 && PHASE2_AGENT_WORKERS > PHASE2_GPU_COUNT )); then
  PHASE2_AGENT_WORKERS="$PHASE2_GPU_COUNT"
fi

if (( PHASE2_AGENT_WORKERS < 1 )); then
  PHASE2_AGENT_WORKERS=1
fi

if (( PHASE2_AGENT_WORKERS == 1 || PHASE2_GPU_COUNT <= 1 )); then
  export WANDB_COUNT="$PHASE2_TOTAL_COUNT"
  run_with_timeout wandb_agent || exit 1    # uses SWEEP_ID & WANDB_COUNT internally
else
  declare -a PHASE2_GPU_SPLITS
  split_gpu_ids PHASE2_GPU_SPLITS "$PHASE2_AGENT_WORKERS" "${GRID_VISIBLE_GPUS[@]}"

  declare -a PHASE2_AGENT_COUNTS=()
  base=$(( PHASE2_TOTAL_COUNT / PHASE2_AGENT_WORKERS ))
  remainder=$(( PHASE2_TOTAL_COUNT % PHASE2_AGENT_WORKERS ))
  for ((i=0; i<PHASE2_AGENT_WORKERS; ++i)); do
    count=$base
    if (( i < remainder )); then
      count=$((count + 1))
    fi
    PHASE2_AGENT_COUNTS+=("$count")
  done

  declare -a PHASE2_PIDS=()
  declare -a PHASE2_AGENT_LABELS=()
  launched=0

  echo "[phase2] launching $PHASE2_AGENT_WORKERS parallel agents (target total count=$PHASE2_TOTAL_COUNT)"
  for ((i=0; i<PHASE2_AGENT_WORKERS; ++i)); do
    count="${PHASE2_AGENT_COUNTS[$i]}"
    if (( count <= 0 )); then
      continue
    fi
    (
      export LOG_DIR="${BASE_LOG_DIR}/phase2_agent_${i}"
      mkdir -p "$LOG_DIR"
      if [[ -n "${PHASE2_GPU_SPLITS[$i]:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${PHASE2_GPU_SPLITS[$i]}"
        echo "[phase2] agent#$i using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      else
        unset CUDA_VISIBLE_DEVICES
      fi
      export WANDB_COUNT="$count"
      echo "[phase2] launching agent#$i with count=$count"
      run_with_timeout wandb_agent
    ) &
    PHASE2_PIDS+=($!)
    PHASE2_AGENT_LABELS+=("agent#$i")
    ((launched++))
  done

  if (( launched == 0 )); then
    echo "[phase2][warn] parallel scheduling resulted in zero active agents; falling back to sequential"
    export WANDB_COUNT="$PHASE2_TOTAL_COUNT"
    run_with_timeout wandb_agent || exit 1
  else
    set +e
    PHASE2_FAIL_RC=0
    for idx in "${!PHASE2_PIDS[@]}"; do
      pid="${PHASE2_PIDS[$idx]}"
      if wait "$pid"; then
        :
      else
        rc=$?
        echo "[phase2][error] ${PHASE2_AGENT_LABELS[$idx]} failed (rc=$rc)" >&2
        PHASE2_FAIL_RC=$rc
      fi
    done
    set -e
    if (( PHASE2_FAIL_RC != 0 )); then
      echo "[phase2][fatal] one or more sweep agents failed" >&2
      exit "$PHASE2_FAIL_RC"
    fi
  fi
fi

# 2) Recheck top-k on extra seeds (with the same timeout pattern)
: "${TOPK_RECHECK:=5}"
: "${EXTRA_SEEDS:=3}"
: "${PHASE2_METRIC:=val_rmse}"
: "${PHASE2_DIRECTION:=min}"
: "${PHASE2_UNLABELED_DIR:=$APP_DIR/data/ZINC-canonicalized}"
: "${PHASE2_LABELED_DIR:=$APP_DIR/data/katielinkmoleculenet_benchmark/train}"

# wall/timeouts – separate knob so you don't couple to the agent's budget
SOFT=$(( (${HARD_WALL_MINS:-60}) * 60 ))   # seconds
GRACE="${KILL_AFTER_SECS:-60}"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/recheck_topk.log"
echo "[recheck_topk] wall budget=${SOFT}s, grace=${GRACE}s"


timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
  "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
  python -u "$APP_DIR/scripts/ci/recheck_topk_from_wandb.py" \
    --sweep "${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_SWEEP_ID2}" \
    --metric "$PHASE2_METRIC" \
    --direction "$PHASE2_DIRECTION" \
    --topk "$TOPK_RECHECK" \
    --extra_seeds "$EXTRA_SEEDS" \
    --program "$APP_DIR/scripts/train_jepa.py" \
    --subcmd "sweep-run" \
    --unlabeled-dir "$PHASE2_UNLABELED_DIR" \
    --labeled-dir   "$PHASE2_LABELED_DIR" \
    --out "${GRID_DIR:-$APP_DIR/grid}/recheck_summary.json" \
  2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
if [[ $rc -eq 0 ]]; then
  :
elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
  echo "[INFO][recheck_topk] graceful stop (rc=$rc); letting outputs flush."
else
  echo "[ERROR][recheck_topk] failed with exit code $rc" >&2
  exit $rc
fi

# === materialize Phase-2 winner as fixed CLI for downstream ===
BEST_JSON="${GRID_DIR:-$APP_DIR/grid}/best_grid_config.json"

if ! BEST_JSON_PATH="$BEST_JSON" RECHECK_LOG="$LOG" python - <<'PY'
import json
import os
import sys

path = os.environ["BEST_JSON_PATH"]
log_path = os.environ.get("RECHECK_LOG")
hint = f"; see {log_path}" if log_path else ""

if not os.path.exists(path):
    print(f"[phase2][fatal] expected {path} but it was not created{hint}", file=sys.stderr)
    sys.exit(2)

try:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception as exc:  # noqa: BLE001 - surface the error to the shell
    print(f"[phase2][fatal] unable to parse {path}: {exc}{hint}", file=sys.stderr)
    sys.exit(2)

config = payload.get("config") if isinstance(payload, dict) else None
if not isinstance(config, dict) or not config:
    print(f"[phase2][fatal] {path} is missing a non-empty config{hint}", file=sys.stderr)
    sys.exit(3)

sys.exit(0)
PY
then
  exit "$?"
fi
