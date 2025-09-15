#!/usr/bin/env bash
set -euxo pipefail

# ----------- inputs & defaults -----------
: "${APP_DIR:=/srv/mjepa}"
: "${MAMBA_ROOT_PREFIX:=~/micromamba}"
: "${WANDB_DIR:=/data/mjepa/wandb}"
: "${CACHE_DIR:=/data/mjepa/cache/graphs}"
: "${RUN_ID:=$(date +%s)}"
: "${EXP_ROOT:=/data/mjepa/experiments/${RUN_ID}}"
ENV_NAME="mjepa"

# ----------- persistent dirs -----------
mkdir -p /data/mjepa/experiments "$WANDB_DIR" "$CACHE_DIR"
mkdir -p "$EXP_ROOT"/{grid,pretrain,finetune,bench,tox21,logs}
ln -sfn "$EXP_ROOT" /data/mjepa/experiments/latest

# ----------- micromamba install / hook -----------
MM_PREFIX="$HOME/micromamba"
MM_BIN="$MM_PREFIX/bin/micromamba"
mkdir -p "$MM_PREFIX/bin"
if ! [ -x "$MM_BIN" ]; then
  case "$(uname -m)" in
    x86_64|amd64) CHAN=linux-64 ;;
    aarch64|arm64) CHAN=linux-aarch64 ;;
    *) echo "Unsupported arch $(uname -m)"; exit 1 ;;
  esac
  curl -fsSL "https://micro.mamba.pm/api/micromamba/${CHAN}/latest" \
  | bsdtar -xjf- -C "$HOME" bin/micromamba
fi
export MAMBA_ROOT_PREFIX
eval "$("$MM_BIN" shell hook -s bash)"

# ----------- env create if needed -----------
if ! micromamba env list | grep -q "^${ENV_NAME}\b"; then
  micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.10 rdkit=2023.09.5 scipy pip
fi

# ----------- base pip deps -----------
micromamba run -n "$ENV_NAME" python -m pip install -U pip setuptools wheel "numpy<2"
micromamba run -n "$ENV_NAME" python -m pip install deepchem==2.8.0

# ----------- Torch (CUDA if available) -----------
if ! micromamba run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import torch, sys; sys.exit(0 if torch.cuda.is_available() or torch.__version__ else 1)
PY
then
  micromamba run -n "$ENV_NAME" python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1
fi

# ----------- Optional PyG if required -----------
if ! micromamba run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch_scatter") and importlib.util.find_spec("torch_geometric") else 1)
PY
then
  TORCH_VER=$(micromamba run -n "$ENV_NAME" python -c "import torch; print(torch.__version__.split('+')[0])")
  CUDA_SUFFIX=$(micromamba run -n "$ENV_NAME" python -c "import torch; print('cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu')")
  micromamba run -n "$ENV_NAME" python -m pip install -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_SUFFIX}.html" torch-scatter==2.1.2 || true
  micromamba run -n "$ENV_NAME" python -m pip install torch-geometric==2.5.3 || true
fi

# ----------- project requirements (safe filter) -----------
if [ -f "$APP_DIR/requirements.txt" ]; then
  grep -viE '^(rdkit|torch(|-geometric|-scatter))([=<>].*)?$' "$APP_DIR/requirements.txt" > /tmp/req-safe.txt || true
  if [ -s /tmp/req-safe.txt ]; then
    micromamba run -n "$ENV_NAME" python -m pip install -r /tmp/req-safe.txt
  fi
fi

# ----------- sanity -----------
set +e
timeout 120 micromamba run -n "$ENV_NAME" python - <<'PY'
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
assert hasattr(MS, "MurckoScaffoldSmiles")
PY
status=$?
set -e
if [ "$status" -eq 0 ]; then
  echo "[prepare-env] sanity check passed"
else
  echo "[prepare-env][warn] sanity check failed or timed out"
fi

echo "[prepare_env] APP_DIR=$APP_DIR"
echo "[prepare_env] EXP_ROOT=$EXP_ROOT"

# --- Ensure yq is installed ---
if ! command -v yq >/dev/null 2>&1; then
  echo "[prepare-env] Installing yq..."
  installed_yq=0
  if command -v apt-get >/dev/null 2>&1; then
    for attempt in 1 2 3; do
      if timeout 300 sudo apt-get update && timeout 300 sudo apt-get install -y yq; then
        installed_yq=1
        break
      elif [ "$attempt" -lt 3 ]; then
        echo "[prepare-env] apt-get install failed (attempt $attempt), retrying..."
        sleep 5
      fi
    done
  elif command -v brew >/dev/null 2>&1; then
    for attempt in 1 2 3; do
      if brew install yq; then
        installed_yq=1
        break
      elif [ "$attempt" -lt 3 ]; then
        echo "[prepare-env] brew install failed (attempt $attempt), retrying..."
        sleep 5
      fi
    done
  else
    echo "[prepare-env][warn] Could not install yq automatically."
    echo "Please install manually: https://github.com/mikefarah/yq"
  fi

  if [ "$installed_yq" -ne 1 ] && ! command -v yq >/dev/null 2>&1; then
    echo "[prepare-env][warn] yq is not installed. Continuing without it."
  fi
fi
