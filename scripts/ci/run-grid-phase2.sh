#!/usr/bin/env bash
set -euo pipefail

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

echo "[phase2] using sweep: $SWEEP_ID"
# Use the same wrapper you use elsewhere for agents (timeout, tee, graceful stop)
run_with_timeout "wandb_agent"    # uses SWEEP_ID & WANDB_COUNT internally:contentReference[oaicite:1]{index=1}

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
  python -u "$APP_DIR/scripts/recheck_topk_from_wandb.py" \
    --sweep "${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_SWEEP_ID2}" \
    --metric "$PHASE2_METRIC" \
    --direction "$PHASE2_DIRECTION" \
    --topk "$TOPK_RECHECK" \
    --extra_seeds "$EXTRA_SEEDS" \
    --program "$APP_DIR/scripts/train_jepa.py" \
    --subcmd "sweep-run" \
    --unlabeled "$PHASE2_UNLABELED_DIR" \
    --labeled   "$PHASE2_LABELED_DIR" \
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

