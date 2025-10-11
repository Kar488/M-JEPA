#!/usr/bin/env bash
# Resolve Phase-1 lineage identifiers on the Vast host and publish them to the caller's CI environment.
set -euo pipefail

missing=()
for var in SSH_KEY VAST_USER VAST_HOST VAST_PORT APP_DIR EXPERIMENTS_ROOT; do
  if [[ -z "${!var:-}" ]]; then
    missing+=("$var")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf 'publish_phase1_lineage: missing required environment: %s\n' "${missing[*]}" >&2
  exit 1
fi

DEFAULT_ID="${DEFAULT_ID:-${RUN_ID:-}}"
export DEFAULT_ID

if [[ -z "$DEFAULT_ID" ]]; then
  printf 'publish_phase1_lineage: DEFAULT_ID or RUN_ID must be provided.\n' >&2
  exit 1
fi

ssh_opts=(
  -o StrictHostKeyChecking=no
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -p "${VAST_PORT}"
)

tmp_key="$(mktemp)"
trap 'rm -f "$tmp_key"' EXIT
printf '%s\n' "$SSH_KEY" >"$tmp_key"
chmod 600 "$tmp_key"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
source "${script_dir}/stage.sh"

printf -v app_dir_q '%q' "$APP_DIR"
printf -v experiments_root_q '%q' "$EXPERIMENTS_ROOT"
printf -v default_id_q '%q' "$DEFAULT_ID"

remote_script=$(cat <<EOF
set -euo pipefail
cd ${app_dir_q}
export APP_DIR="$(pwd)"
source scripts/ci/common.sh
resolve_ci_python
"${PYTHON_CMD[@]}" scripts/ci/resolve_lineage_ids.py --root ${experiments_root_q} --default-id ${default_id_q}
EOF
)

if ! json_payload="$(
  ssh -i "$tmp_key" "${ssh_opts[@]}" "${VAST_USER}@${VAST_HOST}" 'bash -s' <<EOS
${remote_script}
EOS
)"; then
  json_payload='{}'
fi

echo "phase1-lineage: ${json_payload}"

python_inline "${json_payload}" <<'PY'
import json
import os
import sys

payload = {}
try:
    payload = json.loads(sys.argv[1])
except Exception:
    payload = {}

default_id = os.environ.get("DEFAULT_ID", "")
run_id = os.environ.get("RUN_ID", "")

grid_id = (payload.get("grid_exp_id") or default_id or run_id or "").strip()
pretrain_id = (payload.get("pretrain_exp_id") or grid_id or default_id or run_id or "").strip()

if not grid_id:
    sys.exit("Failed to resolve grid_exp_id from payload or defaults")

env_path = os.environ.get("CI_ENV_FILE") or os.environ.get("GITHUB_ENV")
if env_path:
    with open(env_path, "a", encoding="utf-8") as handle:
        handle.write(f"GRID_EXP_ID={grid_id}\n")
        handle.write(f"EXP_ID={grid_id}\n")
        handle.write(f"PRETRAIN_EXP_ID={pretrain_id}\n")

out_path = os.environ.get("CI_OUTPUT_FILE") or os.environ.get("GITHUB_OUTPUT")
if out_path:
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(f"grid_exp_id={grid_id}\n")
        handle.write(f"pretrain_exp_id={pretrain_id}\n")

print(f"Resolved grid_exp_id={grid_id}, pretrain_exp_id={pretrain_id}")
PY

mapfile -t _resolved_ids < <(
  python_inline "${json_payload}" <<'PY'
import json
import os
import sys

payload = {}
try:
    payload = json.loads(sys.argv[1])
except Exception:
    payload = {}

default_id = os.environ.get("DEFAULT_ID", "")
run_id = os.environ.get("RUN_ID", "")

grid_id = (payload.get("grid_exp_id") or default_id or run_id or "").strip()
pretrain_id = (payload.get("pretrain_exp_id") or grid_id or default_id or run_id or "").strip()

print(grid_id)
print(pretrain_id)
PY
)

grid_exp_id="${_resolved_ids[0]:-}"
pretrain_exp_id="${_resolved_ids[1]:-${grid_exp_id}}"

src_grid="${EXPERIMENTS_ROOT%/}/${grid_exp_id}/grid"
dst_grid="${EXPERIMENTS_ROOT%/}/${pretrain_exp_id}/grid"

if [[ -f "${src_grid}/phase2_sweep_id.txt" ]] && [[ ! -f "${dst_grid}/phase2_sweep_id.txt" ]]; then
  mkdir -p "$dst_grid"
  for f in phase2_sweep_id.txt best_grid_config.json recheck_summary.json grid_state.json; do
    if [[ -f "${src_grid}/$f" ]]; then
      cp -f "${src_grid}/$f" "${dst_grid}/$f"
    fi
  done
fi
