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
: "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/nightly/cu128}"
: "${PYTORCH_PACKAGE_SPEC:=--pre torch}"
: "${TORCH_SCATTER_SOURCE:=git+https://github.com/pyg-team/pytorch_scatter.git@2.1.2}"

# ----------- persistent dirs -----------
mkdir -p /data/mjepa/experiments "$WANDB_DIR" "$CACHE_DIR"
mkdir -p "$EXP_ROOT"/{grid,pretrain,finetune,bench,tox21,logs}
ln -sfn "$EXP_ROOT" /data/mjepa/experiments/latest

# ----------- micromamba install / hook -----------
MM_PREFIX="$HOME/micromamba"
MM_BIN="$MM_PREFIX/bin/micromamba"
mkdir -p "$MM_PREFIX/bin"
# Ensure bsdtar is installed on the system
if ! command -v bsdtar >/dev/null 2>&1; then
  echo "Installing bsdtar…"
  sudo apt-get update -qq
  sudo apt-get install -y libarchive-tools
fi
if ! [ -x "$MM_BIN" ]; then
  case "$(uname -m)" in
    x86_64|amd64) CHAN=linux-64 ;;
    aarch64|arm64) CHAN=linux-aarch64 ;;
    *) echo "Unsupported arch $(uname -m)"; exit 1 ;;
  esac
  curl -fsSL "https://micro.mamba.pm/api/micromamba/${CHAN}/latest" \
  | bsdtar -xjf- -C "$MM_PREFIX" bin/micromamba
fi
export MAMBA_ROOT_PREFIX
eval "$("$MM_BIN" shell hook -s bash)"


# ----------- env create if needed -----------
if ! micromamba env list | grep -q "^${ENV_NAME}\b"; then
  micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.10 rdkit=2023.09.5 scipy pip
fi

# ----------- base pip deps -----------
micromamba run -n "$ENV_NAME" python -m pip install -U pip setuptools wheel "numpy<2" ninja cmake
micromamba run -n "$ENV_NAME" python -m pip install deepchem==2.8.0

# ----------- Install CUDA Toolkit 12.8 (for compiling extensions like PyG) -----------
# Add this block before the Torch section. Assumes Ubuntu 22.04/Debian-based (from your apt-get usage).
# If on 24.04, swap 'ubuntu2204' for 'ubuntu2404' in the wget URL.
if ! nvcc --version | grep -q "release 12.8"; then
  echo "[prepare-env] Installing CUDA Toolkit 12.8..."
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo rm cuda-keyring_1.1-1_all.deb  # Cleanup
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-8
  # Add to PATH (persist via ~/.bashrc or eval in script)
  export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
  # Verify
  nvcc --version
fi

# ----------- Torch (CUDA if available) -----------
if ! micromamba run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import torch, sys; sys.exit(0 if torch.cuda.is_available() or torch.__version__ else 1)
PY
then
  IFS=' ' read -r -a _torch_spec <<<"$PYTORCH_PACKAGE_SPEC"
  micromamba run -n "$ENV_NAME" python -m pip install --no-cache-dir \
    --index-url "$PYTORCH_INDEX_URL" \
    "${_torch_spec[@]}"
fi

# ----------- Optional PyG if required -----------
if ! micromamba run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch_scatter") and importlib.util.find_spec("torch_geometric") else 1)
PY
then
  RAW_TORCH_VER=$(micromamba run -n "$ENV_NAME" python -c "import torch; print(torch.__version__)")
  BASE_TORCH_VER=$(micromamba run -n "$ENV_NAME" python - <<'PY'
import re, torch
match = re.match(r"^(\d+\.\d+\.\d+)", torch.__version__)
print(match.group(1) if match else "")
PY
  )
  CUDA_SUFFIX=$(micromamba run -n "$ENV_NAME" python -c "import torch; print('cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu')")

  if [[ -n "$BASE_TORCH_VER" ]]; then
    micromamba run -n "$ENV_NAME" python -m pip install -f "https://data.pyg.org/whl/torch-${BASE_TORCH_VER}+${CUDA_SUFFIX}.html" torch-scatter==2.1.2 || true
  else
    echo "[prepare-env][info] building torch-scatter from source for Torch ${RAW_TORCH_VER}"
    GPU_ARCH="${TORCH_CUDA_ARCH_LIST:-}"
    if [ -z "$GPU_ARCH" ]; then
      GPU_ARCH=$(micromamba run -n "$ENV_NAME" python - <<'PY' 2>/dev/null || true
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
PY
)
    fi
    SCATTER_CMD=(python -m pip install --no-cache-dir --no-build-isolation "$TORCH_SCATTER_SOURCE")
    if [ -n "$GPU_ARCH" ]; then
      echo "[prepare-env][info] using TORCH_CUDA_ARCH_LIST=$GPU_ARCH for torch-scatter build"
      if [[ "$GPU_ARCH" != "0.0" ]]; then
        TORCH_CUDA_ARCH_LIST="$GPU_ARCH" FORCE_CUDA=1 micromamba run -n "$ENV_NAME" "${SCATTER_CMD[@]}" || true
      else
        TORCH_CUDA_ARCH_LIST="$GPU_ARCH" micromamba run -n "$ENV_NAME" "${SCATTER_CMD[@]}" || true
      fi
    else
      micromamba run -n "$ENV_NAME" "${SCATTER_CMD[@]}" || true
    fi
  fi

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
