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
: "${WANDB_DIR:=/data/mjepa/wandb}"

if [[ -z "${APP_DIR:-}" ]]; then
  _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../scripts/ci
  _root="$(cd "${_here}/../.." && pwd)"                      # repo root guess
  if [[ -f "$_root/scripts/train_jepa.py" ]]; then
    APP_DIR="$_root"
  fi
fi
export APP_DIR
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

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

# Allow cache directories to be overridden by env vars supplied by the workflow. If Grid_Dir is not set in yaml it uses cache dir
: "${GRID_DIR:=${GRID_CACHE_DIR:-$EXP_ROOT/grid}}"
: "${PRETRAIN_DIR:=${PRETRAIN_CACHE_DIR:-$EXP_ROOT/pretrain}}"
: "${FINETUNE_DIR:=${FINETUNE_CACHE_DIR:-$EXP_ROOT/finetune}}"
: "${BENCH_DIR:=${BENCH_CACHE_DIR:-$EXP_ROOT/bench}}"
: "${TOX21_DIR:=${TOX21_CACHE_DIR:-$EXP_ROOT/tox21}}"

mkdir -p "$GRID_DIR" "$PRETRAIN_DIR" "$FINETUNE_DIR" "$BENCH_DIR" "$TOX21_DIR" "$LOG_DIR" "$WANDB_DIR"
export GRID_DIR PRETRAIN_DIR FINETUNE_DIR BENCH_DIR TOX21_DIR LOG_DIR

# --- micromamba bootstrap ---
ensure_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    MMBIN="$(command -v micromamba)"
  elif [ -x "$MAMBA_ROOT_PREFIX/bin/micromamba" ]; then
    MMBIN="$MAMBA_ROOT_PREFIX/bin/micromamba"
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
  if [[ ! -s "$cfg" ]]; then
    # fail fast for stages that require the winner
    case "$stage" in bench|pretrain|finetune|tox21)
      echo "[fatal] $stage needs winner but missing: $cfg" >&2; return 2;;
    esac
    return 0
  fi
  local py; py=$(python_bin) || { echo "python not found" >&2; return 127; }
  "$py" - "$cfg" "$stage" <<'PY'
import json, sys
path, stage = sys.argv[1], sys.argv[2]
raw = json.load(open(path))

# --- flatten possible W&B shapes ---
def _get(key, default=None):
    # 1) flat
    if isinstance(raw, dict) and key in raw:
        v = raw[key]
        if isinstance(v, dict) and "value" in v:  # sometimes already wrapped
            return v["value"]
        return v
    # 2) wandb-like: parameters.<key>.value
    p = raw.get("parameters") or raw.get("config") or {}
    if isinstance(p, dict) and key in p:
        v = p[key]
        if isinstance(v, dict) and "value" in v:
            return v["value"]
        return v
    return default

# Build a flat cfg the rest of the code expects
cfg = {}
for k in ["gnn_type","hidden_dim","num_layers","ema_decay","contiguity","add_3d",
          "lr","learning_rate","temperature","training_method","method"]:
    v = _get(k)
    if v is not None:
        cfg[k] = v

# normalise: learning_rate → lr (benchmark/help only exposes --lr)
if "lr" not in cfg and "learning_rate" in cfg:
    cfg["lr"] = cfg["learning_rate"]

# normalise: method → contrastive flag
tm = cfg.get("training_method") or cfg.get("method")
if isinstance(tm, str) and tm.lower().startswith("con"):
    cfg["contrastive"] = True

# Translate single-choice method to a boolean CLI toggle
tm = cfg.get("training_method") or cfg.get("method")
if isinstance(tm, str) and tm.lower().startswith("con"):
    print("--contrastive")

# --- Unified, refactor-safe mappings (kebab-case) ---
# 1) Dataset / loader / device knobs that multiple stages accept
dataset_loader = {
    "dataset_dir": "--dataset-dir",
    "unlabeled_dir": "--unlabeled-dir",
    "labeled_dir": "--labeled-dir",
    "test_dir":    "--test-dir",
    "report_stem": "--report-stem",
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

    # >>> add singular keys used by best_grid_config.json <<<
    # emit --lr universally so it isn't filtered by benchmark's allow-list
    "gnn_type": "--gnn-type",
    "hidden_dim": "--hidden-dim",
    "num_layers": "--num-layers",
    "ema_decay": "--ema-decay",
    "contiguity": "--contiguity",
    "add_3d": "--add-3d",
    "learning_rate": "--lr",

    # other common knobs
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
    "pretrain_epochs": "--pretrain-epochs",
    "finetune_epochs": "--finetune-epochs",
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
