#!/usr/bin/env bash
#set -x   # bash prints each command before executing
set -euo pipefail

grace_marker() {
  local stage="${1:?stage}" log_dir="${2:-${LOG_DIR:-}}"
  if [[ -z "$log_dir" ]]; then
    log_dir="${APP_DIR:-.}/logs"
  fi
  echo "${log_dir}/${stage}.graceful_stop"
}
mark_graceful_stop() {
  local stage="${1:?stage}" log_dir="${2:-${LOG_DIR:-}}"
  if [[ -z "$log_dir" ]]; then
    log_dir="${APP_DIR:-.}/logs"
  fi
  mkdir -p "$log_dir"
  : >"$(grace_marker "$stage" "$log_dir")"
}
was_graceful_stop() {
  local stage="${1:?stage}" log_dir="${2:-${LOG_DIR:-}}"
  if [[ -z "$log_dir" ]]; then
    log_dir="${APP_DIR:-.}/logs"
  fi
  [[ -f "$(grace_marker "$stage" "$log_dir")" ]]
}

# requires: common.sh (ensure_micromamba, build_argv_from_yaml, expand_array_vars, best_config_args)
# ---------- stage → dirs / subcommands / dependencies ----------
stage_dir() {
  case "$1" in
    grid) echo "$GRID_DIR" ;;
    pretrain) echo "$PRETRAIN_DIR" ;;
    finetune) echo "$FINETUNE_DIR" ;;
    bench|benchmark) echo "$BENCH_DIR" ;;
    tox21) echo "$TOX21_DIR" ;;
    report) echo "$REPORTS_DIR" ;;
    phase2_sweep|phase2_recheck|phase2_export) echo "${GRID_DIR}/$1" ;;
    *) echo "$EXP_ROOT/$1" ;;
  esac
}
stage_subcmd() {
  case "$1" in
    grid) echo "grid-search" ;;
    bench) echo "benchmark" ;;
    *) echo "$1" ;;
  esac
}
stage_needs() {
  local s="$1"
  local base=(
    "$APP_DIR/scripts/train_jepa.py"
    "$APP_DIR/scripts/ci/train_jepa_ci.yml"
    "$APP_DIR/scripts/ci/common.sh"
    "$APP_DIR/scripts/ci/run-${s}.sh"
  )
  
  case "$s" in
    grid)   printf '%s\n' "${base[@]}" ;;
    pretrain) printf '%s\n' "${base[@]}" "$GRID_DIR/best_grid_config.json" ;;
    finetune)
      local encoder_dep
      encoder_dep="$(resolve_encoder_checkpoint)"
      printf '%s\n' "${base[@]}" "$GRID_DIR/best_grid_config.json" "$encoder_dep"
      ;;
    bench|benchmark) printf '%s\n' "${base[@]}" "$GRID_DIR/best_grid_config.json" "$FINETUNE_DIR/seed_0/ft_best.pt" ;;
    tox21) printf '%s\n' "${base[@]}" "$GRID_DIR/best_grid_config.json" "$APP_DIR/scripts/ci/data/tox21/data.csv" ;;
    phase2_sweep)
      printf '%s\n' \
        "${base[@]}" \
        "$GRID_DIR/phase2_sweep_id.txt"
      ;;
    phase2_recheck)
      local sweep_stamp
      sweep_stamp="$(stage_dir phase2_sweep)"
      printf '%s\n' \
        "${base[@]}" \
        "$APP_DIR/scripts/ci/recheck_topk_from_wandb.py" \
        "$GRID_DIR/phase2_sweep_id.txt" \
        "${sweep_stamp}/.stamp"
      ;;
    phase2_export)
      local recheck_stamp
      recheck_stamp="$(stage_dir phase2_recheck)"
      printf '%s\n' \
        "${base[@]}" \
        "$GRID_DIR/best_grid_config.json" \
        "$GRID_DIR/recheck_summary.json" \
        "${recheck_stamp}/.stamp"
      ;;
    report)
      printf '%s\n' \
        "$APP_DIR/reports/build_wandb_report.py" \
        "$APP_DIR/reports/discover_schema.py" \
        "$APP_DIR/reports/wandb_utils.py" \
        "$APP_DIR/reports/plots_pretrain.py" \
        "$APP_DIR/reports/plots_regression.py" \
        "$APP_DIR/reports/plots_classification.py" \
        "$APP_DIR/reports/plots_repr.py" \
        "$APP_DIR/reports/plots_tox21.py" \
        "$APP_DIR/reports/plots_compare.py"
      ;;
    *) printf '%s\n' "${base[@]}" ;;
  esac
}

# ---------- phase-2 helpers ----------
phase2_step_diag() {
  local step="$1"
  echo "[ci] EXP_ID=${EXP_ID:-<unset>} EXP_ROOT=${EXP_ROOT:-<unset>} GRID_DIR=${GRID_DIR:-<unset>} STEP=${step}" >&2
}

restore_env_var() {
  local name="$1" previous="$2"
  if [[ -n "$previous" ]]; then
    printf -v "$name" '%s' "$previous"
    export "$name"
  else
    unset "$name"
  fi
}

run_phase2_sweep_stage() {
  local dir="$1" step="$2"

  phase2_step_diag "$step"

  local sweep_id_file="${GRID_DIR}/phase2_sweep_id.txt"
  if [[ ! -f "$sweep_id_file" ]]; then
    echo "[$step][fatal] sweep id file not found: $sweep_id_file" >&2
    return 2
  fi

  local sweep_id
  sweep_id="$(<"$sweep_id_file")"
  export WANDB_SWEEP_ID2="$sweep_id"
  export SWEEP_ID="${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_SWEEP_ID2}"

  local prev_log_dir="${LOG_DIR:-}"
  local step_log_dir="${dir}/logs"
  mkdir -p "$step_log_dir"
  export LOG_DIR="$step_log_dir"

  local prev_wall="${HARD_WALL_MINS:-}"
  local sweep_wall="${PHASE2_SWEEP_WALL_MINS:-920}"
  export HARD_WALL_MINS="$sweep_wall"

  mapfile -t GRID_VISIBLE_GPUS < <(visible_gpu_ids)
  local gpu_count="${#GRID_VISIBLE_GPUS[@]}"

  if [[ -z "${WANDB_COUNT:-}" ]]; then
    export WANDB_COUNT=100
  fi
  local phase2_total_count="$WANDB_COUNT"

  : "${PHASE2_LABELED_DIR:=$APP_DIR/data/katielinkmoleculenet_benchmark/train}"
  : "${PHASE2_UNLABELED_DIR:=$APP_DIR/data/ZINC-canonicalized}"

  if [[ ! -d "$PHASE2_LABELED_DIR" ]]; then
    echo "[$step][fatal] not a dir: $PHASE2_LABELED_DIR" >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    restore_env_var HARD_WALL_MINS "$prev_wall"
    return 2
  fi
  if [[ ! -d "$PHASE2_UNLABELED_DIR" ]]; then
    echo "[$step][fatal] not a dir: $PHASE2_UNLABELED_DIR" >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    restore_env_var HARD_WALL_MINS "$prev_wall"
    return 2
  fi

  local agent_workers=1
  if [[ -n "${PHASE2_AGENT_COUNT:-}" ]]; then
    if [[ "$PHASE2_AGENT_COUNT" =~ ^[0-9]+$ ]]; then
      agent_workers="$PHASE2_AGENT_COUNT"
    else
      echo "[$step][warn] ignoring non-numeric PHASE2_AGENT_COUNT='${PHASE2_AGENT_COUNT}'" >&2
      agent_workers=1
    fi
  elif (( gpu_count > 1 )); then
    agent_workers="$gpu_count"
  fi

  if (( gpu_count > 0 && agent_workers > gpu_count )); then
    agent_workers="$gpu_count"
  fi
  if (( agent_workers < 1 )); then
    agent_workers=1
  fi

  local launched=0

  if (( agent_workers == 1 || gpu_count <= 1 )); then
    export WANDB_COUNT="$phase2_total_count"
    if ! run_with_timeout wandb_agent; then
      restore_env_var LOG_DIR "$prev_log_dir"
      restore_env_var HARD_WALL_MINS "$prev_wall"
      return 1
    fi
    launched=1
  else
    local -a gpu_splits=()
    split_gpu_ids gpu_splits "$agent_workers" "${GRID_VISIBLE_GPUS[@]}"

    local -a agent_counts=()
    local base=$(( phase2_total_count / agent_workers ))
    local remainder=$(( phase2_total_count % agent_workers ))
    for ((i=0; i<agent_workers; ++i)); do
      local count="$base"
      if (( i < remainder )); then
        count=$((count + 1))
      fi
      agent_counts+=("$count")
    done

    local -a pids=()
    local -a labels=()
    for ((i=0; i<agent_workers; ++i)); do
      local count="${agent_counts[$i]}"
      if (( count <= 0 )); then
        continue
      fi
      (
        export LOG_DIR="${step_log_dir}/agent_${i}"
        mkdir -p "$LOG_DIR"
        if [[ -n "${gpu_splits[$i]:-}" ]]; then
          export CUDA_VISIBLE_DEVICES="${gpu_splits[$i]}"
        else
          unset CUDA_VISIBLE_DEVICES
        fi
        export WANDB_COUNT="$count"
        run_with_timeout wandb_agent
      ) &
      pids+=($!)
      labels+=("agent#$i")
      ((launched++))
    done

    if (( launched == 0 )); then
      export WANDB_COUNT="$phase2_total_count"
      if ! run_with_timeout wandb_agent; then
        restore_env_var LOG_DIR "$prev_log_dir"
        restore_env_var HARD_WALL_MINS "$prev_wall"
        return 1
      fi
      launched=1
    else
      local fail_rc=0
      for idx in "${!pids[@]}"; do
        local pid="${pids[$idx]}"
        if wait "$pid"; then
          :
        else
          local rc=$?
          echo "[$step][error] ${labels[$idx]} failed (rc=$rc)" >&2
          fail_rc=$rc
        fi
      done
      if (( fail_rc != 0 )); then
        restore_env_var LOG_DIR "$prev_log_dir"
        restore_env_var HARD_WALL_MINS "$prev_wall"
        return "$fail_rc"
      fi
    fi
  fi

  if find "$step_log_dir" -name 'wandb_agent.graceful_stop' -print -quit >/dev/null 2>&1; then
    mark_graceful_stop "$step"
  fi

  local metadata="${dir}/stage-outputs/${step}.json"
  python - <<'PY' "$metadata" "$sweep_id" "$phase2_total_count" "$agent_workers"
import json
import sys
from datetime import datetime, timezone

path, sweep_id, total, workers = sys.argv[1:5]

payload = {
    "sweep_id": sweep_id,
    "requested_trials": int(total),
    "agent_workers": int(workers),
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
import os
os.replace(tmp, path)
PY

  restore_env_var LOG_DIR "$prev_log_dir"
  restore_env_var HARD_WALL_MINS "$prev_wall"

  return 0
}

run_phase2_recheck_stage() {
  local dir="$1" step="$2"

  phase2_step_diag "$step"

  : "${TOPK_RECHECK:=5}"
  : "${EXTRA_SEEDS:=3}"
  : "${PHASE2_METRIC:=val_rmse}"
  : "${PHASE2_DIRECTION:=min}"
  : "${PHASE2_UNLABELED_DIR:=$APP_DIR/data/ZINC-canonicalized}"
  : "${PHASE2_LABELED_DIR:=$APP_DIR/data/katielinkmoleculenet_benchmark/train}"

  local sweep_id_file="${GRID_DIR}/phase2_sweep_id.txt"
  if [[ ! -f "$sweep_id_file" ]]; then
    echo "[$step][fatal] sweep id file not found: $sweep_id_file" >&2
    return 2
  fi

  local sweep_id
  sweep_id="$(<"$sweep_id_file")"
  export WANDB_SWEEP_ID2="$sweep_id"

  local prev_log_dir="${LOG_DIR:-}"
  local step_log_dir="${dir}/logs"
  mkdir -p "$step_log_dir"
  export LOG_DIR="$step_log_dir"

  local wall_mins="${PHASE2_RECHECK_WALL_MINS:-180}"
  local soft=$(( wall_mins * 60 ))
  local grace="${KILL_AFTER_SECS:-120}"
  local log_path="${step_log_dir}/recheck_topk.log"

  echo "[$step] wall budget=${wall_mins}m (${soft}s), grace=${grace}s"

  timeout --signal=SIGTERM --kill-after="$grace" "$soft" \
    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
    python -u "$APP_DIR/scripts/ci/recheck_topk_from_wandb.py" \
      --sweep "${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_SWEEP_ID2}" \
      --metric "${PHASE2_METRIC}" \
      --direction "${PHASE2_DIRECTION}" \
      --topk "${TOPK_RECHECK}" \
      --extra_seeds "${EXTRA_SEEDS}" \
      --program "$APP_DIR/scripts/train_jepa.py" \
      --subcmd "sweep-run" \
      --unlabeled-dir "${PHASE2_UNLABELED_DIR}" \
      --labeled-dir   "${PHASE2_LABELED_DIR}" \
      --out "${GRID_DIR}/recheck_summary.json" \
    2>&1 | tee "$log_path"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    :
  elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
    echo "[$step] graceful stop (rc=$rc); letting outputs flush." >&2
    mark_graceful_stop "$step"
    restore_env_var LOG_DIR "$prev_log_dir"
    return 0
  else
    restore_env_var LOG_DIR "$prev_log_dir"
    return "$rc"
  fi

  local metadata="${dir}/stage-outputs/${step}.json"
  python - <<'PY' "$metadata" "$sweep_id" "${GRID_DIR}/recheck_summary.json" "${GRID_DIR}/best_grid_config.json" "${PHASE2_METRIC}" "${PHASE2_DIRECTION}" "${TOPK_RECHECK}" "${EXTRA_SEEDS}"
import json
import os
import sys
from datetime import datetime, timezone

path, sweep_id, summary_path, best_path, metric, direction, topk, extra = sys.argv[1:9]

payload = {
    "sweep_id": sweep_id,
    "summary_path": summary_path,
    "best_config_path": best_path,
    "metric": metric,
    "direction": direction,
    "topk": int(topk),
    "extra_seeds": int(extra),
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
PY

  restore_env_var LOG_DIR "$prev_log_dir"
  return 0
}

run_phase2_export_stage() {
  local dir="$1" step="$2"

  phase2_step_diag "$step"

  local prev_log_dir="${LOG_DIR:-}"
  local step_log_dir="${dir}/logs"
  mkdir -p "$step_log_dir"
  export LOG_DIR="$step_log_dir"

  local best_json="${GRID_DIR}/best_grid_config.json"
  local summary_json="${GRID_DIR}/recheck_summary.json"

  if [[ ! -f "$best_json" ]]; then
    echo "[$step][fatal] expected ${best_json} but it was not created" >&2
    restore_env_var LOG_DIR "$prev_log_dir"
    return 3
  fi

  python - <<'PY' "$best_json"
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
if not isinstance(payload, dict) or not payload.get("config"):
    raise SystemExit(4)
PY

  local outputs_dir="${dir}/stage-outputs"
  mkdir -p "$outputs_dir"

  cp -f "$best_json" "${outputs_dir}/best_grid_config.json"
  if [[ -f "$summary_json" ]]; then
    cp -f "$summary_json" "${outputs_dir}/recheck_summary.json"
  fi

  local helper_dir="${outputs_dir}/helpers"
  mkdir -p "$helper_dir"

  while IFS= read -r rel; do
    [[ -z "$rel" || "$rel" == "." ]] && continue
    if [[ -f "${GRID_DIR}/${rel}" ]]; then
      cp -f "${GRID_DIR}/${rel}" "${helper_dir}/"
    fi
  done < <(cd "$GRID_DIR" && find . -maxdepth 1 -type f \( -name 'phase2_winner*' -o -name 'winner_*' -o -name 'phase2_cli*' \) -printf '%P\n' 2>/dev/null || true)

  local metadata="${dir}/stage-outputs/${step}.json"
  python - <<'PY' "$metadata" "$best_json" "$summary_json" "$helper_dir"
import json
import os
import sys
from datetime import datetime, timezone

path, best_path, summary_path, helper_dir = sys.argv[1:5]

helpers = []
if os.path.isdir(helper_dir):
    for entry in sorted(os.listdir(helper_dir)):
        helpers.append(os.path.join(helper_dir, entry))

payload = {
    "best_config_path": best_path,
    "summary_path": summary_path if os.path.exists(summary_path) else None,
    "helper_files": helpers,
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
PY

  printf '[%s] validated Phase-2 best configuration\n' "$step" | tee -a "${step_log_dir}/export.log" >/dev/null

  restore_env_var LOG_DIR "$prev_log_dir"
  return 0
}

# ---------- build & filter args ----------
build_stage_args() {
  local s="${1:?stage}"
  local section="$s"
  local subcmd="$s"
  local skip_best=0
  local skip_allowlist=0
  export TRAIN_JEPA_CI="$APP_DIR/scripts/ci/train_jepa_ci.yml"

  # YAML args
  #snake to kebab case
  if [ "$s" = "grid_search" ]; then
    section="grid_search"   # YAML key
    subcmd="grid-search"    # argparse subcommand
  fi

  if [ "$s" = "bench" ]; then
    section="bench"
    subcmd="benchmark"
  fi

  if [ "$s" = "report" ]; then
    skip_best=1
    skip_allowlist=1
  fi

  build_argv_from_yaml "$section"
  expand_array_vars ARGV
  #echo "ARGV after expansion: ${ARGV[@]}"

  # Best (from grid) → append last so it overrides YAML
  local -a BEST=()
  if (( ! skip_best )); then
    mapfile -t BEST < <(best_config_args "$section")
    expand_array_vars BEST
  fi

  local -a COMBINED=( "${ARGV[@]}" "${BEST[@]}" )

  # Dynamic allowlist from tool help (supports nargs='+')
  local -a OUT=()
  if (( skip_allowlist )); then
    OUT=("${COMBINED[@]}")
  else
    local -a ALLOWED
    mapfile -t ALLOWED < <(
      PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$MMBIN" run -n mjepa python "$APP_DIR/scripts/train_jepa.py" "$subcmd" --help |
          sed -n 's/.*\(--[a-z0-9-]\+\).*/\1/p' | sort -u
    )

    local i=0 j=0
    while (( i < ${#COMBINED[@]} )); do
      local f="${COMBINED[$i]}"
      if [[ "$f" == --* ]] && printf '%s\n' "${ALLOWED[@]}" | grep -qx -- "$f"; then
        OUT+=("$f")
        j=$((i+1))
        while (( j < ${#COMBINED[@]} )) && [[ "${COMBINED[$j]}" != --* ]]; do
          OUT+=("${COMBINED[$j]}"); ((j++))
        done
        i=$j; continue
      fi
      ((i++))
    done
  fi

  if [ "$s" = "report" ]; then
    local -a FILTERED=()
    local idx=0
    while (( idx < ${#OUT[@]} )); do
      local token="${OUT[$idx]}"
      if [[ "$token" == --* ]]; then
        local next=$((idx + 1))
        if (( next < ${#OUT[@]} )) && [[ "${OUT[$next]}" != --* ]]; then
          local value="${OUT[$next]}"
          if [[ -z "$value" ]]; then
            idx=$((next + 1))
            continue
          fi
          FILTERED+=("$token" "$value")
          idx=$((next + 1))
          continue
        fi
      fi
      FILTERED+=("$token")
      ((idx++))
    done
    OUT=("${FILTERED[@]}")
  fi

  STAGE_ARGS=("${OUT[@]}")
}

# ---------- dataset preflight (harmless if flags absent) ----------
# Trim CR and whitespace
clean_path() {
  echo "$1" | tr -d '\r\n' | xargs
}

stage_dataset_preflight() {
  local -n arr="$1"
  local getv; getv() { local k="$1" n=0; while (( n < ${#arr[@]} )); do [[ "${arr[$n]}" == "$k" ]] && { echo "${arr[$((n+1))]}"; return 0; }; ((n++)); done; return 1; }
  local UL="$(getv --unlabeled-dir || true)"
  local LBL="$(getv --labeled-dir  || true)"
  local DS="$(getv --dataset-dir   || true)"
  local CSV="$(getv --csv          || true)"

  UL=$(clean_path "$UL")
  LBL=$(clean_path "$LBL")
  CSV=$(clean_path "$CSV")
  DS=$(clean_path "$DS")

  if [[ -n "$UL" && ! -d "$UL" ]]; then
    echo "[warn] unlabeled-dir not found: $UL"
  fi
  if [[ -n "$LBL" && ! -d "$LBL" ]]; then
    echo "[warn] labeled-dir not found: $LBL"
  fi
  if [[ -n "$DS" && ! -d "$DS" ]]; then
    echo "[warn] dataset-dir not found: $DS"
  fi
  if [[ -n "$CSV" && ! -f "$CSV" ]]; then
    echo "[warn] csv file not found: $CSV"
  fi
  }

# ---------- timeout + SIGINT ----------
run_with_timeout() {
  local s="$1"; shift

    # --- JEPA mode: run_stage passes an array name ---
  if [[ "$s" != "wandb_agent" ]]; then
    if [ "$s" = "report" ]; then
      local BUDGET_MINS="${REPORT_TIME_BUDGET_MINS:-${HARD_WALL_MINS:-30}}"
      local SOFT=$((BUDGET_MINS*60))
      local GRACE="${KILL_AFTER_SECS:-60}"
      echo "[stage] wall budget=${BUDGET_MINS}m (${SOFT}s), grace=${GRACE}s"

      mkdir -p "$LOG_DIR"
      LOG="${LOG_DIR}/${s}.log"

      timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
        env PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
        python -m reports.build_wandb_report "${arr[@]}" \
        2>&1 | tee "$LOG_DIR/${s}.log"

      rc=${PIPESTATUS[0]}
      if [[ $rc -eq 0 ]]; then
        :
      elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
        echo "[INFO][$s] graceful stop (rc=$rc); not marking stage done; outputs should be flushed."
        mark_graceful_stop "$s"
        return 0
      else
        echo "[ERROR][$s] build_wandb_report failed with exit code $rc" >&2
        exit $rc
      fi
      return 0
    fi

    local -n arr="$1"; shift
    local subcmd; subcmd="$(stage_subcmd "$s")"

    #snake to kebab case  
    if [ "$s" = "grid_search" ]; then
      section="grid_search"   # YAML key
      subcmd="grid-search"    # argparse subcommand
    fi

    if [ "$s" = "bench" ]; then
      section="bench"
      subcmd="benchmark"
    fi

    # parsing flags from arr[@] inside stage.sh (e.g. --epochs 50, --batch-size 128),
    local getv; getv() {
      local k="$1" n=0
      while (( n < ${#arr[@]} )); do
        [[ "${arr[$n]}" == "$k" ]] && { echo "${arr[$((n+1))]}"; return 0; }
        ((n++))
      done
      return 1
    }

    # handle default timeouts differently
    local BUDGET_MINS="$(getv --time-budget-mins || true)"
    case "$s" in
      grid) : "${BUDGET_MINS:=${HARD_WALL_MINS:-240}}" ;;
      pretrain|finetune) : "${BUDGET_MINS:=${HARD_WALL_MINS:-120}}" ;;
      *) : "${BUDGET_MINS:=${HARD_WALL_MINS:-60}}" ;;
    esac
    local SOFT="$((BUDGET_MINS*60))"   # convert minutes → seconds
    local GRACE="${KILL_AFTER_SECS:-60}"
    echo "[stage] wall budget=${BUDGET_MINS}m (${SOFT}s), grace=${GRACE}s"

    mkdir -p "$LOG_DIR" 
    LOG="${LOG_DIR}/${s}.log"
    
    timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
      env PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
      "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -u "$APP_DIR/scripts/train_jepa.py" "$subcmd" "${arr[@]}" \
      2>&1 | tee "$LOG_DIR/${s}.log"
    rc=${PIPESTATUS[0]}
    # 0   = success
    # 124 = 'timeout' exceeded (we later sent SIGTERM/SIGKILL)
    # 143 = terminated by SIGTERM (128+15)
    # 137 = killed by SIGKILL   (128+9)
    if [[ $rc -eq 0 ]]; then
      :
    elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
      echo "[INFO][$s] graceful stop (rc=$rc); not marking stage done; outputs should be flushed."
      mark_graceful_stop "$s"
      return 0
    else
      echo "[ERROR][$s] train_jepa.py failed with exit code $rc" >&2
      exit $rc
    fi
  # --- WandB mode: run-grid passes a full cmd array --
  else
    ensure_micromamba

    # ensure the binary is callable in this subshell
    export PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"
    : "${MMBIN:=${MAMBA_ROOT_PREFIX}/bin/micromamba}"
    if [[ "$(basename "$MMBIN")" = "micromamba" && ! "$MMBIN" = /* ]]; then
      MMBIN="${MAMBA_ROOT_PREFIX}/bin/micromamba"
    fi
    if [[ ! -x "$MMBIN" ]]; then
      echo "[wandb_agent][fatal] micromamba not found at $MMBIN" >&2
      exit 1
    fi

   # --- ensure logging + ids are sane ---
    mkdir -p "$LOG_DIR"
    LOG="${LOG_DIR}/${s}.log"

    # build a full sweep path if only an id was provided
    SID="${SWEEP_ID}"
    if [[ -z "$SID" ]]; then echo "[wandb_agent][fatal] SWEEP_ID is empty" >&2; exit 1; fi

    if [[ "$SID" != */* ]]; then
      [[ -n "${WANDB_ENTITY:-}" && -n "${WANDB_PROJECT:-}" ]] \
        || { echo "[wandb_agent][fatal] SWEEP_ID not qualified and entity/project unset"; exit 1; }
      SID="${WANDB_ENTITY}/${WANDB_PROJECT}/${SID}"
    fi

    echo "[wandb_agent] using sweep: $SID"

    local -a cmd=("$@")
    local SOFT=$(( (${HARD_WALL_MINS:-240})*60 ))
    local GRACE="${KILL_AFTER_SECS:-60}"
    echo "[wandb_agent] wall budget=${SOFT}s, grace=${GRACE}s"

    timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
      "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -m wandb agent --count ${WANDB_COUNT:-50} "$SID" \
      2>&1 | tee "$LOG"

    rc=$?
    # If the agent “gracefully” exited 0 but clearly failed runs, force non-zero
    if grep -qE 'Detected [0-9]+ failed runs|error: the following arguments are required' "$LOG"; then
      echo "[ERROR][wandb_agent] runs failed; forcing non-zero exit"
      rc=1
    fi
    if [[ $rc -eq 0 ]]; then
      :
    elif [[ $rc -eq 124 || $rc -eq 130 || $rc -eq 143 || $rc -eq 137 ]]; then
      echo "[INFO][wandb_agent] graceful stop (rc=$rc); letting agent flush."
      mark_graceful_stop "$s"
      return 0
    else
      echo "[ERROR][wandb_agent] wandb agent failed with exit code $rc" >&2
      exit $rc
    fi
  fi
}

# ---------- one-call entry ----------
run_stage() {
  local s="${1:?stage}"

  if [[ -n "${MJEPACI_STAGE_SHIM:-}" && -x "${MJEPACI_STAGE_SHIM}" ]]; then
    local dir
    dir="$(stage_dir "$s")"
    mkdir -p "$dir" "$dir/stage-outputs"
    "$MJEPACI_STAGE_SHIM" "$s"
    mark_stage_done "$dir"
    return 0
  fi

  ensure_micromamba
  : "${WANDB_NAME:=$s}"; export WANDB_NAME
  : "${WANDB_JOB_TYPE:=$s}"; export WANDB_JOB_TYPE
  export WANDB_RUN_GROUP="$GITHUB_RUN_ID"

  local dir; dir="$(stage_dir "$s")"
  if needs_stage "$dir" $(stage_needs "$s"); then
    echo "[$s] starting"
    mkdir -p "$dir" "$dir/stage-outputs"

    local stage_log_dir="${LOG_DIR:-}"

    case "$s" in
      phase2_sweep)
        stage_log_dir="${dir}/logs"
        run_phase2_sweep_stage "$dir" "$s"
        ;;
      phase2_recheck)
        stage_log_dir="${dir}/logs"
        run_phase2_recheck_stage "$dir" "$s"
        ;;
      phase2_export)
        stage_log_dir="${dir}/logs"
        run_phase2_export_stage "$dir" "$s"
        ;;
      *)
        stage_log_dir="${LOG_DIR:-}"
        build_stage_args "$s"
        stage_dataset_preflight STAGE_ARGS
        run_with_timeout "$s" STAGE_ARGS
        ;;
    esac

    local -a grace_dirs=()
    if [[ -n "$stage_log_dir" ]]; then
      grace_dirs+=("$stage_log_dir")
    fi
    if [[ -n "${LOG_DIR:-}" && "${LOG_DIR:-}" != "$stage_log_dir" ]]; then
      grace_dirs+=("${LOG_DIR:-}")
    fi

    for candidate in "${grace_dirs[@]}"; do
      if [[ -n "$candidate" ]] && was_graceful_stop "$s" "$candidate"; then
        echo "[$s] stopped gracefully; leaving cache unstamped so it can resume."
        return 0
      fi
    done


    mark_stage_done "$dir"
    echo "[$s] completed"
  else
    echo "[$s] cache hit - skipping"
  fi
}