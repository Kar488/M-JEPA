#!/usr/bin/env bash
set -euo pipefail

# ----------- inputs & defaults -----------
: "${APP_DIR:=/srv/mjepa}"
: "${RUN_ID:=$(date +%s)}"
export MJEPACI_STAGE="prepare-env"
: "${EXP_ID:=${RUN_ID}}"

CI_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${CI_SCRIPT_DIR}/common.sh"

if [[ -n "${PIP_CACHE_DIR:-}" ]]; then
  if mjepa_try_dir "${PIP_CACHE_DIR}"; then
    export PIP_CACHE_DIR
  else
    mjepa_log_warn "PIP_CACHE_DIR=${PIP_CACHE_DIR} not writable; disabling pip cache"
    unset PIP_CACHE_DIR
  fi
fi

EXP_ROOT="${EXPERIMENTS_ROOT%/}/${EXP_ID}"
ENV_NAME="mjepa"
: "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu128}"
: "${PYTORCH_PACKAGE_SPEC:=torch==2.8.*}"
: "${PYTORCH_NIGHTLY_INDEX_URL:=https://download.pytorch.org/whl/nightly/cu128}"
: "${PYTORCH_NIGHTLY_PACKAGE_SPEC:=--pre torch}"
: "${PYTORCH_ALLOW_NIGHTLY_FALLBACK:=1}"
: "${PYTORCH_FAIL_FAST_ON_BAD_CUDA:=1}"
: "${BUILD_SCATTER_FROM_SOURCE:=0}"

# ----------- persistent dirs -----------
ln -sfn "$EXP_ROOT" "${EXPERIMENTS_ROOT%/}/latest"

# ----------- driver / gpu sanity -----------
CUDA_EXPECTED=0
if command -v nvidia-smi >/dev/null 2>&1; then
  nv_output=""
  nv_status=0
  if ! nv_output=$(nvidia-smi -L 2>&1); then
    nv_status=$?
  fi

  if [ -z "$nv_output" ] || printf '%s\n' "$nv_output" | grep -qiE 'no devices were found'; then
    echo "[prepare-env][error] nvidia-smi reports no accessible GPU devices."
    echo "[prepare-env][error] Ensure a CUDA-capable GPU and matching drivers are available."
    [ "$nv_status" -ne 0 ] && echo "[prepare-env][error] Output: $nv_output"
    exit 1
  fi

  if ! printf '%s\n' "$nv_output" | grep -qiE '^GPU [0-9]+:'; then
    echo "[prepare-env][error] Unable to determine the available GPU devices from nvidia-smi -L."
    echo "[prepare-env][error] Output: $nv_output"
    exit 1
  fi

  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    detected_mask=$(printf '%s\n' "$nv_output" | sed -n 's/^GPU \([0-9]\+\):.*/\1/p' | paste -sd, -)
    if [ -n "$detected_mask" ]; then
      export CUDA_VISIBLE_DEVICES="$detected_mask"
      echo "[prepare-env] Defaulting CUDA_VISIBLE_DEVICES to detected GPU mask: $CUDA_VISIBLE_DEVICES"
    fi
  fi

  if [ "$nv_status" -ne 0 ]; then
    echo "[prepare-env][warn] nvidia-smi -L exited with status $nv_status; continuing with reported devices."
    echo "[prepare-env][warn] Output: $nv_output"
  fi

  CUDA_EXPECTED=1
else
  echo "[prepare-env][error] nvidia-smi not found; GPU-equipped host is required."
  exit 1
fi

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
  cuda_tmp_dir="$(mktemp -d "${RUNNER_TEMP:-/tmp}/cuda-keyring.XXXXXX")"
  cuda_keyring_path="${cuda_tmp_dir}/cuda-keyring_1.1-1_all.deb"
  curl -fsSL -o "$cuda_keyring_path" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i "$cuda_keyring_path"
  rm -rf "$cuda_tmp_dir"
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
# Run nvidia-smi to find usable GPUs
healthy_gpus=$(nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits 2>/dev/null \
    | awk '!/Unknown Error/ {print $1}' | paste -sd "," -)

if [ -z "$healthy_gpus" ]; then
  echo "[prepare-env][warn] No healthy GPUs detected by nvidia-smi; defaulting to all"
  export CUDA_VISIBLE_DEVICES=0,1
else
  echo "[prepare-env] Healthy GPUs detected: $healthy_gpus"
  export CUDA_VISIBLE_DEVICES="$healthy_gpus"
fi

# Optional: show for debugging
echo "[prepare-env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if ! micromamba run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import torch, sys; sys.exit(0 if torch.cuda.is_available() or torch.__version__ else 1)
PY
then
  IFS=' ' read -r -a _torch_spec <<<"$PYTORCH_PACKAGE_SPEC"
  echo "[prepare-env] Installing PyTorch from ${PYTORCH_INDEX_URL} (${_torch_spec[*]})"
  if ! micromamba run -n "$ENV_NAME" python -m pip install --no-cache-dir \
    --index-url "$PYTORCH_INDEX_URL" \
    "${_torch_spec[@]}"; then
    if [ "${PYTORCH_ALLOW_NIGHTLY_FALLBACK:-1}" -eq 1 ]; then
      IFS=' ' read -r -a _torch_nightly_spec <<<"$PYTORCH_NIGHTLY_PACKAGE_SPEC"
      echo "[prepare-env][warn] Stable PyTorch install failed, attempting nightly from ${PYTORCH_NIGHTLY_INDEX_URL} (${_torch_nightly_spec[*]})"
      micromamba run -n "$ENV_NAME" python -m pip install --no-cache-dir \
        --index-url "$PYTORCH_NIGHTLY_INDEX_URL" \
        "${_torch_nightly_spec[@]}"
    else
      echo "[prepare-env][error] Failed to install stable PyTorch and nightly fallback disabled (PYTORCH_ALLOW_NIGHTLY_FALLBACK=0)"
      exit 1
    fi
  fi
fi

TORCH_META=$(micromamba run -n "$ENV_NAME" python - <<'PY'
import shlex
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - defensive
    print("TORCH_SUPPORTED=0")
    print(f"TORCH_ERROR={shlex.quote('Torch import failed: ' + repr(exc))}")
    sys.exit(0)

raw = torch.__version__
base = raw.split('+')[0]
parts = base.split('.')
if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
    print("TORCH_SUPPORTED=0")
    print(f"TORCH_ERROR={shlex.quote('Unsupported Torch version: ' + raw)}")
    sys.exit(0)

major, minor = parts[:2]
page_version = f"{major}.{minor}.0"
major_minor = f"{major}.{minor}"
cuda_version = torch.version.cuda or ""
suffix = f"cu{cuda_version.replace('.', '')}" if cuda_version else "cpu"

print("TORCH_SUPPORTED=1")
print(f"TORCH_VERSION={shlex.quote(raw)}")
print(f"TORCH_MAJOR_MINOR={shlex.quote(major_minor)}")
print(f"TORCH_PAGE_VERSION={shlex.quote(page_version)}")
print(f"TORCH_CUDA_VERSION={shlex.quote(cuda_version)}")
print(f"TORCH_SCATTER_SUFFIX={shlex.quote(suffix)}")
print(f"TORCH_CUDA_AVAILABLE={1 if torch.cuda.is_available() else 0}")
PY
)
eval "$TORCH_META"

echo "[prepare-env] Torch version: ${TORCH_VERSION:-unknown} (cuda available: ${TORCH_CUDA_AVAILABLE:-0}, torch.version.cuda='${TORCH_CUDA_VERSION:-}')"

if [ "${PYTORCH_FAIL_FAST_ON_BAD_CUDA:-1}" -eq 1 ] && [ "$CUDA_EXPECTED" -eq 1 ]; then
  if [ "${TORCH_CUDA_AVAILABLE:-0}" -ne 1 ]; then
    echo "[prepare-env][error] CUDA-capable driver detected but torch.cuda.is_available()=False."
    echo "[prepare-env][error] Ensure matching NVIDIA drivers are installed and CUDA_VISIBLE_DEVICES is set correctly."
    exit 1
  fi
fi

# ----------- Optional PyG if required -----------
if ! micromamba run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch_scatter") else 1)
PY
then
  need_source_build=$BUILD_SCATTER_FROM_SOURCE
  case "$need_source_build" in
    0|1) ;;
    *)
      echo "[prepare-env][warn] BUILD_SCATTER_FROM_SOURCE should be 0 or 1; defaulting to 0"
      need_source_build=0
      ;;
  esac

  if [ "${TORCH_SUPPORTED:-0}" -eq 1 ]; then
    PAGE_URL="https://data.pyg.org/whl/torch-${TORCH_PAGE_VERSION}+${TORCH_SCATTER_SUFFIX}.html"
    echo "[prepare-env] Using PyG wheel page: ${PAGE_URL}"
    if ! micromamba run -n "$ENV_NAME" python -m pip install --no-cache-dir -f "$PAGE_URL" torch-scatter==2.1.2; then
      if [ "$BUILD_SCATTER_FROM_SOURCE" -eq 1 ]; then
        echo "[prepare-env][warn] Wheel install failed, falling back to source build because BUILD_SCATTER_FROM_SOURCE=1"
      else
        echo "[prepare-env][error] Failed to install torch-scatter wheel. Set BUILD_SCATTER_FROM_SOURCE=1 to attempt a source build or install a supported Torch release."
        echo "[prepare-env][error] If pip cached a broken download, try clearing with 'pip cache purge'."
        exit 1
      fi
    else
      need_source_build=0
    fi
  else
    if [ "$BUILD_SCATTER_FROM_SOURCE" -ne 1 ]; then
      echo "[prepare-env][error] ${TORCH_ERROR:-Unsupported Torch configuration}. Set BUILD_SCATTER_FROM_SOURCE=1 to force a source build or install a supported stable Torch release."
      exit 1
    fi
    need_source_build=1
  fi

  if [ "$need_source_build" -eq 1 ]; then
    echo "[prepare-env][info] Building torch-scatter from source"
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
    SCATTER_BUILD_CMD=(micromamba run -n "$ENV_NAME" python -m pip install --no-cache-dir --no-build-isolation --no-binary torch-scatter torch-scatter==2.1.2)
    if [ -n "$GPU_ARCH" ]; then
      echo "[prepare-env][info] using TORCH_CUDA_ARCH_LIST=$GPU_ARCH for torch-scatter build"
      TORCH_CUDA_ARCH_LIST="$GPU_ARCH" FORCE_CUDA=1 "${SCATTER_BUILD_CMD[@]}"
    else
      "${SCATTER_BUILD_CMD[@]}"
    fi
  fi

  micromamba run -n "$ENV_NAME" python -c "import torch_scatter" >/dev/null
  echo "[prepare-env] torch-scatter import check passed"
fi

if ! micromamba run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch_geometric") else 1)
PY
then
  micromamba run -n "$ENV_NAME" python -m pip install torch-geometric==2.5.3
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
