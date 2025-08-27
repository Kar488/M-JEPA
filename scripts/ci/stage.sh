#!/usr/bin/env bash
set -x   # bash prints each command before executing
set -euo pipefail

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
  local subcmd; subcmd="$(stage_subcmd "$s")"
  export TRAIN_JEPA_CI="$APP_DIR/scripts/ci/train_jepa_ci.yml"

  # YAML args
  #snake to kebab case 
  local section="$s"
  local subcmd="$s"
  if [ "$s" = "grid" ]; then
    section="grid_search"   # YAML key
    subcmd="grid-search"    # argparse subcommand
  fi

  build_argv_from_yaml "$section"
  expand_array_vars ARGV
  echo "ARGV after expansion: ${ARGV[@]}"

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
    if [[ "$f" == --* ]] && printf '%s\n' "${ALLOWED[@]}" | grep -qx "$f"; then
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
stage_dataset_preflight() {
  local -n arr="$1"
  local getv; getv() { local k="$1" n=0; while (( n < ${#arr[@]} )); do [[ "${arr[$n]}" == "$k" ]] && { echo "${arr[$((n+1))]}"; return 0; }; ((n++)); done; return 1; }
  local UL="$(getv --unlabeled-dir || true)"
  local LBL="$(getv --labeled-dir  || true)"
  local DS="$(getv --dataset-dir   || true)"
  local CSV="$(getv --csv          || true)"
  [[ -n "$UL"  && ! -d "$UL"  ]] && { echo "missing --unlabeled-dir $UL"; return 66; }
  [[ -n "$LBL" && ! -d "$LBL" ]] && { echo "missing --labeled-dir $LBL"; return 66; }
  [[ -n "$DS"  && ! -d "$DS"  ]] && { echo "missing --dataset-dir $DS"; return 66; }
  [[ -n "$CSV" && ! -f "$CSV" ]] && { echo "missing --csv file $CSV"; return 66; }
}

# ---------- timeout + SIGINT ----------
run_with_timeout() {
  local s="$1"; shift
  local -n arr="$1"; shift
  local subcmd; subcmd="$(stage_subcmd "$s")"

  local getv; getv() { local k="$1" n=0; while (( n < ${#arr[@]} )); do [[ "${arr[$n]}" == "$k" ]] && { echo "${arr[$((n+1))]}"; return 0; }; ((n++)); done; return 1; }
  local BUDGET_MINS="$(getv --time-budget-mins || true)"
  case "$s" in
    grid) : "${BUDGET_MINS:=${HARD_WALL_MINS:-240}}" ;;
    pretrain|finetune) : "${BUDGET_MINS:=${HARD_WALL_MINS:-120}}" ;;
    *) : "${BUDGET_MINS:=${HARD_WALL_MINS:-60}}" ;;
  esac
  local SOFT="${BUDGET_MINS}m"; local GRACE="${KILL_AFTER_SECS:-60}s"
  mkdir -p "$LOG_DIR"

  timeout --signal=SIGINT --kill-after="$GRACE" "$SOFT" \
    PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$MMBIN" run -n mjepa env PYTHONUNBUFFERED=1 \
    python -u "$APP_DIR/scripts/train_jepa.py" "$subcmd" "${arr[@]}" \
    2>&1 | tee "$LOG_DIR/${s}.log"
}

# ---------- one-call entry ----------
run_stage() {
  local s="${1:?stage}"
  ensure_micromamba
  : "${WANDB_NAME:=$s}"; export WANDB_NAME
  : "${WANDB_JOB_TYPE:=$s}"; export WANDB_JOB_TYPE

  local dir; dir="$(stage_dir "$s")"
  if needs_stage "$dir" $(stage_needs "$s"); then
    echo "[$s] starting"
    build_stage_args "$s"
    stage_dataset_preflight STAGE_ARGS
    run_with_timeout "$s" STAGE_ARGS
    mark_stage_done "$dir"
    echo "[$s] completed"
  else
    echo "[$s] cache hit - skipping"
  fi
}
