#!/usr/bin/env bash
#set -x   # bash prints each command before executing
set -euo pipefail

grace_marker() { echo "$LOG_DIR/${1}.graceful_stop"; }
mark_graceful_stop() { mkdir -p "$LOG_DIR"; : >"$(grace_marker "$1")"; }
was_graceful_stop() { [[ -f "$(grace_marker "$1")" ]]; }

# requires: common.sh (ensure_micromamba, build_argv_from_yaml, expand_array_vars, best_config_args)
# ---------- stage → dirs / subcommands / dependencies ----------
stage_dir() {
  case "$1" in
    grid) echo "$GRID_DIR" ;;
    pretrain) echo "$PRETRAIN_DIR" ;;
    finetune) echo "$FINETUNE_DIR" ;;
    bench|benchmark) echo "$BENCH_DIR" ;;
    tox21) echo "$TOX21_DIR" ;;
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
    finetune) printf '%s\n' "${base[@]}" "$GRID_DIR/best_grid_config.json" "$PRETRAIN_DIR/encoder.pt" ;;
    bench|benchmark) printf '%s\n' "${base[@]}" "$GRID_DIR/best_grid_config.json" "$FINETUNE_DIR/seed_0/ft_best.pt" ;;
    tox21) printf '%s\n' "${base[@]}" "$GRID_DIR/best_grid_config.json" "$APP_DIR/scripts/ci/data/tox21/data.csv" ;;
    *) printf '%s\n' "${base[@]}" ;;
  esac
}

# ---------- build & filter args ----------
build_stage_args() {
  local s="${1:?stage}"
  local section="$s"
  local subcmd="$s"
  export TRAIN_JEPA_CI="$APP_DIR/scripts/ci/train_jepa_ci.yml"

  # YAML args
  #snake to kebab case  
  if [ "$s" = "grid_search" ]; then
    section="grid_search"   # YAML key
    subcmd="grid-search"    # argparse subcommand
  fi

  build_argv_from_yaml "$section"
  expand_array_vars ARGV
  #echo "ARGV after expansion: ${ARGV[@]}"

  # Best (from grid) → append last so it overrides YAML
  local -a BEST
  mapfile -t BEST < <(best_config_args "$section")
  expand_array_vars BEST

  local -a COMBINED=( "${ARGV[@]}" "${BEST[@]}" )

  # Dynamic allowlist from tool help (supports nargs='+')
  local -a ALLOWED
  mapfile -t ALLOWED < <(
    PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
      "$MMBIN" run -n mjepa python "$APP_DIR/scripts/train_jepa.py" "$subcmd" --help |
        sed -n 's/.*\(--[a-z0-9-]\+\).*/\1/p' | sort -u
  )

  local -a OUT=()
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
    local -n arr="$1"; shift
    local subcmd; subcmd="$(stage_subcmd "$s")"

    #snake to kebab case  
    if [ "$s" = "grid_search" ]; then
      section="grid_search"   # YAML key
      subcmd="grid-search"    # argparse subcommand
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

    local -a cmd=("$@")
    local SOFT=$(( (${HARD_WALL_MINS:-240})*60 ))
    local GRACE="${KILL_AFTER_SECS:-60}"
    echo "[wandb_agent] wall budget=${SOFT}s, grace=${GRACE}s"

    timeout --signal=SIGTERM --kill-after="$GRACE" "$SOFT" \
      "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
      python -m wandb agent --count ${WANDB_COUNT:-50} "$SWEEP_ID" \
      2>&1 | tee "$LOG_DIR/${s}.log"

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
  ensure_micromamba
  : "${WANDB_NAME:=$s}"; export WANDB_NAME
  : "${WANDB_JOB_TYPE:=$s}"; export WANDB_JOB_TYPE
  export WANDB_RUN_GROUP="$GITHUB_RUN_ID"

  local dir; dir="$(stage_dir "$s")"
  if needs_stage "$dir" $(stage_needs "$s"); then
    echo "[$s] starting"
    build_stage_args "$s"
    stage_dataset_preflight STAGE_ARGS

    run_with_timeout "$s" STAGE_ARGS
    if was_graceful_stop "$s"; then
      echo "[$s] stopped gracefully; leaving cache unstamped so it can resume."
      return 0
    fi


    mark_stage_done "$dir"
    echo "[$s] completed"
  else
    echo "[$s] cache hit - skipping"
  fi
}