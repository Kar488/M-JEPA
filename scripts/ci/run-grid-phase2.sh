#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ci] error at line $LINENO: $BASH_COMMAND" >&2' ERR

if [[ "${GRID_EXP_ID+x}" == x ]]; then
  CI_PHASE2_INCOMING_GRID_EXP_ID_SET=1
  CI_PHASE2_INCOMING_GRID_EXP_ID="$GRID_EXP_ID"
else
  CI_PHASE2_INCOMING_GRID_EXP_ID_SET=0
  CI_PHASE2_INCOMING_GRID_EXP_ID=""
fi

if [[ "${PRETRAIN_EXP_ID+x}" == x ]]; then
  CI_PHASE2_INCOMING_PRETRAIN_EXP_ID_SET=1
  CI_PHASE2_INCOMING_PRETRAIN_EXP_ID="$PRETRAIN_EXP_ID"
else
  CI_PHASE2_INCOMING_PRETRAIN_EXP_ID_SET=0
  CI_PHASE2_INCOMING_PRETRAIN_EXP_ID=""
fi

export MJEPACI_STAGE="phase2"

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/stage.sh"
source "$(dirname "$0")/wandb_utils.sh"

ci_phase2_select_existing_grid() {
  local common_grid_id="$1"
  local common_pretrain_id="$2"
  local common_source_dir="$3"
  local out_id_var="$4"
  local out_dir_var="$5"

  local root="${EXPERIMENTS_ROOT%/}"
  local best_id="" best_dir="" best_mtime=-1

  consider_dir() {
    local cid="$1" dir="$2"
    [[ -n "$dir" ]] || return 0
    local sweep_file="${dir%/}/phase2_sweep_id.txt"
    [[ -f "$sweep_file" ]] || return 0
    local mtime
    mtime=$(stat -c '%Y' "$sweep_file" 2>/dev/null || echo 0)
    if (( mtime > best_mtime )); then
      best_mtime="$mtime"
      best_id="$cid"
      best_dir="${dir%/}"
    fi
  }

  if [[ -n "$common_source_dir" ]]; then
    local guess_id=""
    if [[ -n "$root" && "$common_source_dir" == "$root"/* ]]; then
      guess_id="${common_source_dir#$root/}"
      guess_id="${guess_id%%/*}"
    fi
    consider_dir "$guess_id" "$common_source_dir"
  fi

  if [[ -n "$root" && -d "$root" ]]; then
    declare -A seen_ids=()
    local -a id_candidates=()
    local candidate
    for candidate in "$common_grid_id" "$common_pretrain_id" "${PRETRAIN_STATE_ID:-}" "${RUN_ID:-}" "${EXP_ID:-}"; do
      [[ -n "$candidate" ]] || continue
      if [[ -z "${seen_ids[$candidate]:-}" ]]; then
        seen_ids[$candidate]=1
        id_candidates+=("$candidate")
      fi
    done

    for candidate in "${id_candidates[@]}"; do
      local dir="$root/$candidate/grid"
      consider_dir "$candidate" "$dir"
    done

    if [[ -z "$best_dir" ]]; then
      while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        local ts="${line%% *}"
        local path="${line#* }"
        local dir
        dir="$(dirname "$path")"
        local parent
        parent="$(dirname "$dir")"
        local cid=""
        if [[ "$parent" == "$root" ]]; then
          cid="$(basename "$dir")"
        else
          cid="$(basename "$parent")"
        fi
        consider_dir "$cid" "$dir"
        break
      done < <(find "$root" -maxdepth 3 -mindepth 3 -path '*/grid/phase2_sweep_id.txt' -printf '%T@ %p\n' 2>/dev/null | sort -nr)
    fi
  fi

  printf -v "$out_id_var" '%s' "$best_id"
  printf -v "$out_dir_var" '%s' "$best_dir"

  [[ -n "$best_dir" ]]
}

ci_phase2_locate_phase1_spec() {
  local out_id_var="$1" out_dir_var="$2"
  local root="${EXPERIMENTS_ROOT%/}"
  local best_id="" best_dir=""

  if [[ -n "${EXP_ID:-}" && -n "$root" ]]; then
    local candidate_spec="${root}/${EXP_ID}/grid/grid_sweep_phase2.yaml"
    if [[ -f "$candidate_spec" ]]; then
      printf -v "$out_id_var" '%s' "${EXP_ID}"
      printf -v "$out_dir_var" '%s' "${candidate_spec%/*}"
      return 0
    fi
  fi

  if [[ -d "$root" ]]; then
    local best_line
    best_line=$(find "$root" -maxdepth 3 -path '*/grid/grid_sweep_phase2.yaml' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1)
    if [[ -n "$best_line" ]]; then
      local spec_path
      spec_path="${best_line#* }"
      best_dir="$(dirname "$spec_path")"
      best_id="$(basename "$(dirname "$best_dir")")"
    fi
  fi

  if [[ -z "$best_dir" && -n "$root" && -n "${EXP_ID:-}" ]]; then
    best_id="${EXP_ID}"
    best_dir="${root}/${EXP_ID}/grid"
  fi

  printf -v "$out_id_var" '%s' "$best_id"
  printf -v "$out_dir_var" '%s' "$best_dir"

  [[ -n "$best_dir" ]]
}

ci_phase2_ensure_sweep_id() {
  local grid_root="${1%/}"
  local prior_source="${2%/}"
  local sweep_file="${grid_root}/phase2_sweep_id.txt"
  if [[ -f "$sweep_file" ]]; then
    return 0
  fi

  local -a spec_candidates=()
  spec_candidates+=("${grid_root}/grid_sweep_phase2.yaml")
  if [[ -n "$prior_source" && "$prior_source" != "$grid_root" ]]; then
    spec_candidates+=("${prior_source}/grid_sweep_phase2.yaml")
  fi
  if [[ -n "${GRID_DIR:-}" ]]; then
    spec_candidates+=("${GRID_DIR%/}/grid_sweep_phase2.yaml")
  fi

  local spec=""
  local candidate
  declare -A seen_specs=()
  for candidate in "${spec_candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    candidate="${candidate%/}"
    if [[ -n "${seen_specs[$candidate]:-}" ]]; then
      continue
    fi
    seen_specs[$candidate]=1
    if [[ -f "$candidate" ]]; then
      spec="$candidate"
      break
    fi
  done

  if [[ -z "$spec" ]]; then
    echo "[phase2][fatal] unable to locate grid_sweep_phase2.yaml to create a new sweep (searched under $grid_root and $prior_source)" >&2
    return 2
  fi

  mkdir -p "$grid_root"

  local sweep_id
  sweep_id="$(wandb_sweep_create "$spec")"
  if [[ ! "$sweep_id" =~ ^[a-z0-9]{8}$ ]]; then
    echo "[phase2][fatal] wandb sweep create returned unexpected id '$sweep_id'" >&2
    return 2
  fi

  printf '%s\n' "$sweep_id" >"$sweep_file"
  if [[ -n "${GRID_DIR:-}" && "${GRID_DIR%/}" != "$grid_root" ]]; then
    mkdir -p "${GRID_DIR%/}"
    printf '%s\n' "$sweep_id" >"${GRID_DIR%/}/phase2_sweep_id.txt"
  fi
  echo "[phase2] created new sweep id=$sweep_id (spec=$spec)" >&2
}

if declare -F ci_setup_vast_ssh_key >/dev/null 2>&1; then
  ci_setup_vast_ssh_key || true
fi

common_grid_exp_id="${GRID_EXP_ID:-}"
common_pretrain_exp_id="${PRETRAIN_EXP_ID:-}"
common_grid_source_dir="${GRID_SOURCE_DIR:-}"

new_grid_exp_id=""
new_pretrain_exp_id=""

ci_phase2_candidate_grid_id=""
ci_phase2_candidate_grid_dir=""

ci_phase2_force_reuse="$(normalize_bool "${FORCE_REUSE_PHASE2_IDS:-0}" 0)"
ci_phase2_force_reuse="$(normalize_bool "${CI_PHASE2_FORCE_REUSE_PHASE2_IDS:-${ci_phase2_force_reuse}}" "$ci_phase2_force_reuse")"

ci_phase2_select_existing_grid \
  "$common_grid_exp_id" \
  "$common_pretrain_exp_id" \
  "$common_grid_source_dir" \
  ci_phase2_candidate_grid_id \
  ci_phase2_candidate_grid_dir || true

freeze_active=0
if (( FROZEN )) && [[ "${CI_FORCE_UNFREEZE_GRID}" != "1" ]]; then
  freeze_active=1
fi

if (( freeze_active || ci_phase2_force_reuse )) && [[ -n "$ci_phase2_candidate_grid_dir" ]]; then
  new_grid_exp_id="$ci_phase2_candidate_grid_id"
  if [[ -z "$new_grid_exp_id" ]]; then
    new_grid_exp_id="$(basename "$(dirname "$ci_phase2_candidate_grid_dir")")"
  fi
  GRID_SOURCE_DIR="$ci_phase2_candidate_grid_dir"
  export GRID_SOURCE_DIR
  GRID_DIR="$ci_phase2_candidate_grid_dir"
  export GRID_DIR
  if [[ -n "$new_grid_exp_id" ]]; then
    candidate_parent="$(dirname "$ci_phase2_candidate_grid_dir")"
    if [[ -f "${candidate_parent}/pretrain_state.json" ]]; then
      new_pretrain_exp_id="$new_grid_exp_id"
    fi
  fi
  if [[ -n "$new_grid_exp_id" && "$new_grid_exp_id" != "$common_grid_exp_id" ]]; then
    echo "[phase2] reusing existing grid lineage ${new_grid_exp_id}" >&2
  fi
elif (( freeze_active || ci_phase2_force_reuse )); then
  echo "[phase2][warn] requested reuse of existing grid but no prior lineage with phase2 sweep outputs was found" >&2
fi

if [[ -z "$new_grid_exp_id" ]]; then
  phase1_grid_id=""
  phase1_grid_dir=""
  if ci_phase2_locate_phase1_spec phase1_grid_id phase1_grid_dir; then
    new_grid_exp_id="$phase1_grid_id"
    if [[ -n "$phase1_grid_dir" ]]; then
      GRID_SOURCE_DIR="$phase1_grid_dir"
      export GRID_SOURCE_DIR
    fi
  elif [[ -n "${EXP_ID:-}" ]]; then
    if [[ -n "${CI_PHASE2_INCOMING_GRID_EXP_ID:-}" ]] && [[ "${CI_PHASE2_INCOMING_GRID_EXP_ID}" != "${EXP_ID}" ]]; then
      echo "[phase2] ignoring stale GRID_EXP_ID=${CI_PHASE2_INCOMING_GRID_EXP_ID} in favour of EXP_ID=${EXP_ID}" >&2
    fi
    if [[ -n "${CI_PHASE2_INCOMING_PRETRAIN_EXP_ID:-}" ]] && [[ "${CI_PHASE2_INCOMING_PRETRAIN_EXP_ID}" != "${EXP_ID}" ]]; then
      echo "[phase2] ignoring stale PRETRAIN_EXP_ID=${CI_PHASE2_INCOMING_PRETRAIN_EXP_ID} in favour of EXP_ID=${EXP_ID}" >&2
    fi
    new_grid_exp_id="${EXP_ID}"
    if [[ -n "${EXPERIMENTS_ROOT:-}" ]]; then
      GRID_SOURCE_DIR="${EXPERIMENTS_ROOT%/}/${EXP_ID}/grid"
      export GRID_SOURCE_DIR
    fi
  fi
fi

if [[ -n "$new_grid_exp_id" ]]; then
  GRID_EXP_ID="$new_grid_exp_id"
  export GRID_EXP_ID
fi

if [[ -n "$new_pretrain_exp_id" ]]; then
  PRETRAIN_EXP_ID="$new_pretrain_exp_id"
  export PRETRAIN_EXP_ID
elif [[ -n "$new_grid_exp_id" && -n "${EXPERIMENTS_ROOT:-}" ]]; then
  candidate_pretrain_state="${EXPERIMENTS_ROOT%/}/${new_grid_exp_id}/pretrain_state.json"
  if [[ -f "$candidate_pretrain_state" ]]; then
    PRETRAIN_EXP_ID="$new_grid_exp_id"
    export PRETRAIN_EXP_ID
  fi
fi

if [[ -n "$new_grid_exp_id" ]] || [[ -n "$new_pretrain_exp_id" ]]; then
  echo "[phase2] binding EXP_ID=${EXP_ID:-<unset>} -> GRID_EXP_ID=${GRID_EXP_ID:-<unset>} PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID:-<unset>}" >&2
  echo "        (previous grid=${common_grid_exp_id:-<unset>} pretrain=${common_pretrain_exp_id:-<unset>})" >&2
  ci_phase2_refresh_lineage_bindings \
    "$new_pretrain_exp_id" \
    "$new_grid_exp_id" \
    "$common_pretrain_exp_id" \
    "$common_grid_exp_id"
elif [[ -z "${GRID_SOURCE_DIR:-}" && -n "${GRID_EXP_ID:-}" ]]; then
  GRID_SOURCE_DIR="${EXPERIMENTS_ROOT%/}/${GRID_EXP_ID}/grid"
  export GRID_SOURCE_DIR
fi

if (( !(freeze_active || ci_phase2_force_reuse) )) && [[ -n "${GRID_EXP_ID:-}" && -n "${EXPERIMENTS_ROOT:-}" ]]; then
  target_grid_dir="${EXPERIMENTS_ROOT%/}/${GRID_EXP_ID}/grid"
  if [[ -z "${GRID_DIR:-}" ]]; then
    GRID_DIR="$target_grid_dir"
    export GRID_DIR
  elif [[ -n "$common_grid_exp_id" && "${GRID_DIR%/}" == "${EXPERIMENTS_ROOT%/}/${common_grid_exp_id}/grid" ]]; then
    GRID_DIR="$target_grid_dir"
    export GRID_DIR
  fi
fi

if [[ -z "${GRID_SOURCE_DIR:-}" && -n "$ci_phase2_candidate_grid_dir" ]]; then
  GRID_SOURCE_DIR="$ci_phase2_candidate_grid_dir"
  export GRID_SOURCE_DIR
fi

unset CI_PHASE2_INCOMING_GRID_EXP_ID_SET CI_PHASE2_INCOMING_GRID_EXP_ID
unset CI_PHASE2_INCOMING_PRETRAIN_EXP_ID_SET CI_PHASE2_INCOMING_PRETRAIN_EXP_ID

if [[ -z "${STAGE_BIN:-}" ]]; then
  STAGE_BIN="run_stage"
fi
export STAGE_BIN

ci_print_env_diag "$STAGE_BIN"

if [[ -n "${GRID_SOURCE_DIR:-}" ]]; then
  grid_root="${GRID_SOURCE_DIR%/}"
  source_hint="${common_grid_source_dir%/}"
  if ! ci_phase2_ensure_sweep_id "$grid_root" "$source_hint"; then
    exit $?
  fi
  # At this point GRID_EXP_ID and PRETRAIN_EXP_ID have been set (either from
  # reusing an existing sweep or by creating a new one).  Persist them into
  # the GitHub Actions environment so that subsequent steps (collect artifacts,
  # pretrain, finetune) use the correct grid and pretrain experiment IDs.
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "GRID_EXP_ID=${GRID_EXP_ID}"      >> "$GITHUB_ENV"
    echo "EXP_ID=${GRID_EXP_ID}"           >> "$GITHUB_ENV"
    echo "PRETRAIN_EXP_ID=${PRETRAIN_EXP_ID}" >> "$GITHUB_ENV"
  fi
  unset grid_root source_hint || true
fi

steps=(phase2_sweep phase2_recheck phase2_export)
for step in "${steps[@]}"; do
  "$STAGE_BIN" "$step"
done
