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

sanitize_yaml() {
        local f="$1"
        perl -0777 -i -pe 's/command:\s*\[[^\]]*\]/command:\n  - "${interpreter}"\n  - "${program}"\n  - "sweep-run"\n  - "${args}"/s' "$f"
        sed -i -E 's/\blabeled[_-]dir\b/labeled-dir/g; s/\bunlabeled[_-]dir\b/unlabeled-dir/g' "$f"
        dos2unix "$f" 2>/dev/null || true
    }

if [[ "$GRID_MODE_CLEAN" == "wandb" ]]; then
    echo "[grid] running wandb sweep agent"

    ensure_micromamba

    # make sure micromamba is reachable here (this subshell may not inherit PATH)
    export PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"
    # normalize to absolute path if 'command -v' returned just 'micromamba'
    : "${MMBIN:=${MAMBA_ROOT_PREFIX}/bin/micromamba}"
    if [[ "$(basename "$MMBIN")" = "micromamba" && ! "$MMBIN" = /* ]]; then
        MMBIN="${MAMBA_ROOT_PREFIX}/bin/micromamba"
    fi
    if [[ ! -x "$MMBIN" ]]; then
        echo "[wandb_agent][fatal] micromamba not found at $MMBIN" >&2
        exit 1
    fi

    # keep one umbrella group; never force a run id for agent trials
    unset WANDB_NAME WANDB_RUN_ID
    : "${WANDB_RUN_GROUP:=${GITHUB_RUN_ID:-pipeline-$(date -u +%Y%m%dT%H%M%SZ)}}"
    export WANDB_RUN_GROUP WANDB_RESUME=allow

    # -------- Phase 1 orchestration (two sweeps) --------

    # Create sweeps via the Python module inside the env that has wandb installed
    # (pass project/entity explicitly so it doesn't rely on local config)
    echo "[phase1] creating sweeps…jepa"

    JEPA_SPEC="${JEPA_SWEEP_SPEC:-$APP_DIR/sweeps/sweep_phase1_jepa.yaml}"
    sanitize_yaml "$JEPA_SPEC"
    JEPA_ID=$(
  "$MMBIN" run -n mjepa env \
    APP_DIR="$APP_DIR" JEPA_SPEC="$JEPA_SPEC" \
    WANDB_PROJECT="$WANDB_PROJECT" WANDB_ENTITY="$WANDB_ENTITY" \
    python - <<'PY' | tail -n1 | tr -d '\r\n '
import os, yaml, wandb
app = os.environ["APP_DIR"]
with open(os.environ["JEPA_SPEC"], "r") as f:
    spec = yaml.safe_load(f)

# Force absolute program + explicit command
spec["program"] = os.path.join(app, "scripts", "train_jepa.py")
spec["command"] = ["python", spec["program"], "sweep-run", "${args}"]

print("[JEPA command]", " ".join(spec["command"]))
sid = wandb.sweep(spec, project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"])
print(sid)
PY
)

    echo "$JEPA_ID" > "${GRID_DIR:-$APP_DIR/grid}/sweep_jepa.id"

    echo "[phase1] creating sweeps…contrastive"
    CONTRAST_SPEC="${CONTRAST_SWEEP_SPEC:-$APP_DIR/sweeps/sweep_phase1_contrastive.yaml}"
    sanitize_yaml "$CONTRAST_SPEC"
    CONTRAST_ID=$(
  "$MMBIN" run -n mjepa env \
    APP_DIR="$APP_DIR" CONTRAST_SPEC="$CONTRAST_SPEC" \
    WANDB_PROJECT="$WANDB_PROJECT" WANDB_ENTITY="$WANDB_ENTITY" \
    python - <<'PY' | tail -n1 | tr -d '\r\n '
import os, yaml, wandb
app = os.environ["APP_DIR"]
with open(os.environ["CONTRAST_SPEC"], "r") as f:
    spec = yaml.safe_load(f)

spec["program"] = os.path.join(app, "scripts", "train_jepa.py")
spec["command"] = ["python", spec["program"], "sweep-run", "${args}"]

print("[CTR command]", " ".join(spec["command"]))
sid = wandb.sweep(spec, project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"])
print(sid)
PY
)
    # hard-validate: must be exactly 8 lowercase letters/digits
    if [[ ! "$JEPA_ID" =~ ^[a-z0-9]{8}$ ]] || [[ ! "$CONTRAST_ID" =~ ^[a-z0-9]{8}$ ]]; then
        echo "[phase1][fatal] bad sweep ids: JEPA_ID='$JEPA_ID' CONTRAST_ID='$CONTRAST_ID'" >&2
        exit 1
    fi

    
    echo "[phase1] JEPA=$JEPA_ID  CONTRAST=$CONTRAST_ID"

    cd "$APP_DIR"

    # run JEPA agents (serial on 1 GPU; change CUDA_VISIBLE_DEVICES to fan out)
    SWEEP_ID="$JEPA_ID"
    echo "[phase1] JEPA → $SWEEP_ID"
    export WANDB_COUNT=${WANDB_COUNT:-30}
    export SWEEP_ID="${WANDB_ENTITY}/${WANDB_PROJECT}/${JEPA_ID}"
    run_with_timeout wandb_agent || exit 1

    # run Contrastive agents
    SWEEP_ID="$CONTRAST_ID"
    echo "[phase1] Contrastive → $SWEEP_ID"
    export WANDB_COUNT=${WANDB_COUNT:-30}
    export SWEEP_ID="${WANDB_ENTITY}/${WANDB_PROJECT}/${CONTRAST_ID}"
    run_with_timeout wandb_agent || exit 1
    
    # paired-effect report
    echo "[phase1] paired-effect"
    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
    python -u "$APP_DIR/scripts/ci/paired_effect_from_wandb.py" \
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
    SWEEP_ID2="$(wandb sweep -q "$PHASE2_PATH")"
    echo -n "$SWEEP_ID2" > "${GRID_DIR:-$APP_DIR/grid}/phase2_sweep_id.txt"
    echo "[phase2] created sweep: $SWEEP_ID2  (saved to $GRID_DIR/phase2_sweep_id.txt)"
        
else
    echo "[grid] running custom grid-search"
    #ensure the parm matches train_jepa_ci.yml
    run_stage grid_search
fi