#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_graph_visuals.py"

COMMON_SH="${SCRIPT_DIR}/common.sh"
if [[ -r "${COMMON_SH}" ]]; then
  # shellcheck source=scripts/ci/common.sh
  source "${COMMON_SH}"
fi

PYTHON=()
if declare -F resolve_ci_python >/dev/null 2>&1; then
  resolve_ci_python PYTHON
fi
if [[ ${#PYTHON[@]} -eq 0 ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=(python)
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
  elif [[ -n "${MMBIN:-}" ]]; then
    PYTHON=("${MMBIN}" run -n mjepa python)
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
  DATASET_ARG="${DATASET_DIR:-}"
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
