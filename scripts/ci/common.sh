#!/usr/bin/env bash
# Common helpers for Vast GPU CI stages
set -euo pipefail

# --- centralised environment variables ---
: "${APP_DIR:=/srv/mjepa}"
: "${VENV_DIR:=/srv/mjepa/.venv}"
: "${MAMBA_ROOT_PREFIX:=/data/mjepa/micromamba}"
: "${CACHE_DIR:=/data/mjepa/cache/graphs}"
: "${RUN_ID:=$(date +%s)}"
: "${EXP_ROOT:=/data/mjepa/experiments/${RUN_ID}}"

# Auto-detect repo root from this script if APP_DIR is off
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # .../scripts/ci
_root_guess="$(cd "${_here}/../.." && pwd)"                  # repo root
if [ ! -d "${APP_DIR}/experiments" ] || [ ! -f "${APP_DIR}/scripts/train_jepa.py" ]; then
  if [ -d "${_root_guess}/experiments" ] && [ -f "${_root_guess}/scripts/train_jepa.py" ]; then
    APP_DIR="${_root_guess}"
  fi
fi
export APP_DIR

# Ensure project modules are discoverable when running from subdirectories.
export PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Determine an available Python interpreter. Prefer 'python', fallback to 'python3'.
python_bin() {
  if command -v python >/dev/null 2>&1; then
    echo python
  elif command -v python3 >/dev/null 2>&1; then
    echo python3
  else
    return 1
  fi
}

# Allow cache directories to be overridden by env vars supplied by the workflow
GRID_DIR="${GRID_CACHE_DIR:-$EXP_ROOT/grid}"
PRETRAIN_DIR="${PRETRAIN_CACHE_DIR:-$EXP_ROOT/pretrain}"
FINETUNE_DIR="${FINETUNE_CACHE_DIR:-$EXP_ROOT/finetune}"
BENCH_DIR="${BENCH_CACHE_DIR:-$EXP_ROOT/bench}"
TOX21_DIR="${TOX21_CACHE_DIR:-$EXP_ROOT/tox21}"
LOG_DIR="$EXP_ROOT/logs"
mkdir -p "$GRID_DIR" "$PRETRAIN_DIR" "$FINETUNE_DIR" "$BENCH_DIR" "$TOX21_DIR" "$LOG_DIR"

# --- micromamba bootstrap ---
ensure_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    MMBIN="$(command -v micromamba)"
  elif [ -x "$HOME/micromamba/bin/micromamba" ]; then
    MMBIN="$HOME/micromamba/bin/micromamba"
  else
    echo "micromamba not found" >&2
    return 1
  fi
  export MAMBA_ROOT_PREFIX
  export MMBIN
  eval "$("$MMBIN" shell hook -s bash)" || true
}

# --- cache stamp utilities ---
_stamp() { echo "$1/.stamp"; }
needs_stage() {
  local dir="$1"; shift
  local stamp=$(_stamp "$dir")
  if [ ! -f "$stamp" ]; then return 0; fi
  for f in "$@"; do
    [ ! -e "$f" ] && return 0
    [ "$f" -nt "$stamp" ] && return 0
  done
  return 1
}
mark_stage_done() { touch "$(_stamp "$1")"; }

# --- progress bar helper ---
progress_bar() {
  local pct="$1"
  local w=40
  local done=$((pct*w/100))
  printf '\r['
  printf '%0.s#' $(seq 1 $done)
  printf '%0.s-' $(seq $((done+1)) $w)
  printf '] %s%%' "$pct"
}

simulate_progress() {
  for p in 0 20 40 60 80 100; do
    progress_bar $p
    sleep 0.1
  done
  printf '\n'
}

# --- yaml argument helper ---
yaml_args() {
  # Usage: yaml_args <section>
  # Prints one argument per line: --key <value> OR --flag (for booleans)
  # Rules: 
  #  - Convert underscores to kebab-case flags (cache_dir -> --cache-dir)
  #  - Strings with spaces => wrap in double quotes
  #  - Env refs like ${VAR} or $VAR => keep for shell to expand (double-quote)
  #  - Lists => repeat the flag
  #  - true => boolean flag present; false => omitted
  # Determine a python interpreter (python, python3, etc.)
   local py; py=$(python_bin) || { echo "python not found" >&2; return 127; }
   "$py" - "$@" <<'PY'
import sys, os, yaml, re
section = sys.argv[1] if len(sys.argv) > 1 else "grid_search"
with open(os.environ.get("TRAIN_JEPA_CI"), "r") as f:
    cfg = yaml.safe_load(f) or {}
node = cfg.get(section, {})
env_ref = re.compile(r'^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$')

def emit(k, v):
    key = "--" + k.replace("_","-")
    if isinstance(v, bool):
        if v:
            print(key)
        return

    if isinstance(v, (int, float)):
        print(key); print(str(v)); return
    
    if isinstance(v, (list, tuple)):
        # keys that accept multiple values with a single flag (nargs='+')
        multi = {
            "methods","mask_ratios","contiguities","hidden_dims","num_layers_list",
            "gnn_types","ema_decays","add_3d_options","pretrain_batch_sizes",
            "finetune_batch_sizes","pretrain_epochs_options","finetune_epochs_options",
            "learning_rates","seeds","aug_rotate_options","aug_mask_angle_options",
            "aug_dihedral_options","temperatures"
        }
        if k in multi:
            print(key)
            for item in v: print(item)
        else:
            for item in v: emit(k, item)
        return

    if v is None:
        return
    s = str(v)
    print(key); print(s)

for k, v in (node or {}).items():
    emit(k, v)
PY
  }

build_argv_from_yaml() {
  # Returns an array named ARGV built from yaml_args output (one token per line)
  local section="${1:-grid_search}"
  local -a tmp
  # Read one token per line, preserve spaces via previous quoting
  mapfile -t tmp < <(yaml_args "$section")
  ARGV=("${tmp[@]}")
}

expand_array_vars() {
  local -n _arr="$1"
  local i
  for i in "${!_arr[@]}"; do
    [[ "${_arr[$i]}" == --* ]] && continue
    _arr[$i]=$(eval "echo ${_arr[$i]}")
  done
}

# --- inject best grid search configuration ---
best_config_args() {
  # Usage: best_config_args <stage>
  # Reads best_grid_config.json and prints CLI args for the given stage
  local stage="$1"
  local cfg="$GRID_DIR/best_grid_config.json"
  [ -f "$cfg" ] || return 0
  local py; py=$(python_bin) || { echo "python not found" >&2; return 127; }
  "$py" - "$cfg" "$stage" <<'PY'
import json, sys
path, stage = sys.argv[1], sys.argv[2]
cfg = json.load(open(path))

# --- Unified, refactor-safe mappings (kebab-case) ---
# 1) Dataset / loader / device knobs that multiple stages accept
dataset_loader = {
    "dataset_dir": "--dataset-dir",
    "unlabeled_dir": "--unlabeled-dir",
    "labeled_dir": "--labeled-dir",
    "csv": "--csv",
    "label_col": "--label-col",
    "smiles_col": "--smiles-col",
    "task_type": "--task-type",
    "cache_dir": "--cache-dir",
    "no_cache": "--no-cache",
    "num_workers": "--num-workers",
    "prefetch_factor": "--prefetch-factor",
    "persistent_workers": "--persistent-workers",
    "pin_memory": "--pin-memory",
    "bf16": "--bf16",
    "devices": "--devices",
    "device": "--device",
}

# 2) Model architecture & training hyperparams (shared)
model_common = {
    "gnn_types": "--gnn-types",
    "hidden_dims": "--hidden-dims",
    "num_layers_list": "--num-layers",
    "ema_decays": "--ema-decays",
    "contiguities": "--contiguities",
    "add_3d_options": "--add-3d-options",
    "lr": "--lr",
    "learning_rates": "--learning-rates",
    "use_scheduler": "--use-scheduler",
    "warmup_steps": "--warmup-steps",
    "temperature": "--temperature",
    "contrastive": "--contrastive",
}

# 3) Augmentations (JEPA vs. contrastive study toggles)
augment = {
    "aug_rotate_options": "--aug-rotate-options",
    "aug_mask_angle_options": "--aug-mask-angle-options",
    "aug_dihedral_options": "--aug-dihedral-options",
}

# 4) Logging / outputs
logging_io = {
    "out_csv": "--out-csv",
    "best_config_out": "--best-config-out",
    "ckpt_dir": "--ckpt-dir",
    "ckpt_every": "--ckpt-every",
    "output": "--output",
    "force_tqdm": "--force-tqdm",
    "use_wandb": "--use-wandb",
    "wandb_project": "--wandb-project",
    "wandb_tags": "--wandb-tags",
    "report_dir": "--report-dir",
}

# 5) Stage-specific knobs
grid_only = {
    "methods": "--methods",
    "mask_ratios": "--mask-ratios",
    "pretrain_batch_sizes": "--pretrain-batch-sizes",
    "finetune_batch_sizes": "--finetune-batch-sizes",
    "pretrain_epochs_options": "--pretrain-epochs-options",
    "finetune_epochs_options": "--finetune-epochs-options",
    "seeds": "--seeds",
    "sample_unlabeled": "--sample-unlabeled",
    "sample_labeled": "--sample-labeled",
    "n_rows_per_file": "--n-rows-per-file",
    "max_pretrain_batches": "--max-pretrain-batches",
    "target_pretrain_samples": "--target-pretrain-samples",
    "max_finetune_batches": "--max-finetune-batches",
    "time_budget_mins": "--time-budget-mins",
    "temperatures": "--temperatures",
    "force_refresh": "--force-refresh",
}

pretrain_only = {
    "mask_ratio": "--mask-ratio",
    "pretrain_batch_size": "--batch-size",
    "pretrain_epochs": "--epochs",
    "save_every": "--save-every",
}

finetune_only = {
    "finetune_batch_size": "--batch-size",
    "finetune_epochs": "--epochs",
    "metric": "--metric",
    "patience": "--patience",
    "jepa_encoder": "--jepa-encoder",
}

benchmark_only = {
    "jepa_encoder": "--jepa-encoder",
    "ft_ckpt": "--ft-ckpt",
}

tox21_only = {
    "task": "--task",
}

# 6) Final per-stage maps (compose without duplication)
maps = {
    "grid": {
        **dataset_loader, **model_common, **augment, **logging_io, **grid_only
    },
    "pretrain": {
        **dataset_loader, **model_common, **augment, **logging_io, **pretrain_only
    },
    "finetune": {
        **dataset_loader, **model_common, **augment, **logging_io, **finetune_only
    },
    "bench": {
        **dataset_loader, **model_common, **augment, **logging_io, **benchmark_only
    },
    "tox21": {
        **dataset_loader, **model_common, **augment, **logging_io, **tox21_only
    },
}

mapping = maps.get(stage, {})
for key, flag in mapping.items():
    if key not in cfg:
        continue
    val = cfg[key]
    if isinstance(val, bool):
        if val:
            print(flag)
    else:
        print(flag)
        print(val)
PY
}
