#!/usr/bin/env bash
set -euo pipefail

: "${APP_DIR:=/srv/mjepa}"
: "${MJEPACI_STAGE:=prepare-env}"


CI_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${CI_SCRIPT_DIR}/common.sh"


if [[ -z "${EXPERIMENTS_ROOT:-}" ]]; then
  echo "[ensure-experiments-root][error] EXPERIMENTS_ROOT is not set" >&2
  exit 1
fi

if ! mjepa_try_dir "${EXPERIMENTS_ROOT}" "${EXPERIMENTS_ROOT}"; then
  mjepa_log_error "unable to provision EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT}"
  exit 1
fi

exp_slot="${EXPERIMENTS_ROOT%/}/${EXP_ID:-${RUN_ID:-}}"
if [[ -n "${exp_slot}" ]]; then
  mkdir -p "${exp_slot}"/stage-outputs
fi

echo "[ensure-experiments-root] using ${EXPERIMENTS_ROOT}"
