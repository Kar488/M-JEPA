#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_graph_visuals.py"

COMMON_SH="${SCRIPT_DIR}/common.sh"
if [[ -r "${COMMON_SH}" ]]; then
  # shellcheck source=scripts/ci/common.sh
  source "${COMMON_SH}"
fi

APP_ROOT="${APP_DIR:-/srv/mjepa}"
DEFAULT_DATASET_DIR="${APP_ROOT%/}/data/ZINC-canonicalized"
DEFAULT_GRAPH_VISUALS_DATASET="${APP_ROOT%/}/data/tox21/data.csv"
DATA_ROOT_REAL="${DATA_ROOT:-${APP_ROOT}}"
: "${DATASET_DIR:=${DEFAULT_DATASET_DIR}}"
: "${GRAPH_VISUALS_DATASET:=${DEFAULT_GRAPH_VISUALS_DATASET}}"

if declare -F ensure_micromamba >/dev/null 2>&1; then
  ensure_micromamba || true
fi

PYTHON=()
if [[ -n "${MMBIN:-}" ]]; then
  PYTHON=("${MMBIN}" run -n mjepa python)
elif command -v micromamba >/dev/null 2>&1; then
  PYTHON=(micromamba run -n mjepa python)
elif declare -F resolve_ci_python >/dev/null 2>&1; then
  resolve_ci_python PYTHON
fi
if [[ ${#PYTHON[@]} -eq 0 ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=(python)
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
  else
    PYTHON=(python3)
  fi
fi

DATASET_ARG=""
OUTPUT_ARG=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_ARG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_ARG="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift 1
      ;;
  esac
done

if [[ -z "${DATASET_ARG}" ]]; then
  DATASET_ARG="${GRAPH_VISUALS_DATASET:-${DATASET_DIR:-}}"
  for cache_hint in "${DATA_ROOT_REAL%/}/cache/graphs_10m" "${DATA_ROOT_REAL%/}/cache/graphs_250k"; do
    if [[ -n "${cache_hint}" && "${DATASET_ARG}" == "${cache_hint}" ]]; then
      echo "[graph-visuals] info: graph visuals dataset points to a cache (${DATASET_ARG}); defaulting to dataset corpus ${DEFAULT_GRAPH_VISUALS_DATASET}" >&2
      DATASET_ARG="${DEFAULT_GRAPH_VISUALS_DATASET}"
      break
    fi
  done
fi

if [[ -z "${DATASET_ARG}" ]]; then
  DATASET_ARG="${DEFAULT_GRAPH_VISUALS_DATASET}"
fi

if [[ -z "${OUTPUT_ARG}" && -n "${PRETRAIN_EXPERIMENT_ROOT:-}" ]]; then
  OUTPUT_ARG="${PRETRAIN_EXPERIMENT_ROOT%/}/graphs"
fi

if [[ -z "${OUTPUT_ARG}" ]]; then
  echo "[pretrain] skipping graph visualisations because PRETRAIN_EXPERIMENT_ROOT is unset" >&2
  exit 0
fi

mkdir -p "${OUTPUT_ARG}"

cmd=("${PYTHON[@]}" "${PYTHON_SCRIPT}" --output-dir "${OUTPUT_ARG}")
if [[ -n "${DATASET_ARG}" ]]; then
  cmd+=(--dataset-path "${DATASET_ARG}")
fi
if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
  cmd+=("${PASSTHROUGH[@]}")
fi

"${cmd[@]}"
