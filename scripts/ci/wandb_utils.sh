#!/usr/bin/env bash
set -euo pipefail

# Ensure micromamba is available (common.sh defines this)
type ensure_micromamba >/dev/null 2>&1 || source "$(dirname "$0")/common.sh"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "[grid][fatal] missing required command: $1" >&2
        exit 127
    }
}

sanitize_sweep_yaml() {
  local f="$1"
  # force block-list command and hyphenated flags
  perl -0777 -i -pe 's/command:\s*\[[^\]]*\]/command:\n  - "python"\n  - "${program}"\n  - "sweep-run"\n  - "${args}"/s' "$f"
  sed -i -E 's/\blabeled[_-]dir\b/labeled-dir/g; s/\bunlabeled[_-]dir\b/unlabeled-dir/g' "$f"
  dos2unix "$f" 2>/dev/null || true
}

sanitize_sweep_yaml_copy() {
  # $1 = source yaml, $2 = dest yaml
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  # Fix flow-list command → block list with explicit 'python'
  perl -0777 -i -pe 's/command:\s*\[[^\]]*\]/command:\n  - "python"\n  - "${program}"\n  - "sweep-run"\n  - "${args}"/s' "$dst"
  # Hyphenated flags expected by your CLI
  sed -i -E 's/\blabeled[_-]dir\b/labeled-dir/g; s/\bunlabeled[_-]dir\b/unlabeled-dir/g' "$dst"
}

qualify_sweep_id() {
  local id="$1"
  if [[ "$id" == */* ]]; then echo "$id"; else echo "${WANDB_ENTITY}/${WANDB_PROJECT}/${id}"; fi
}

wandb_sweep_create() {
  local spec="$1"
  [[ -f "$spec" ]] || { echo "[fatal] missing sweep spec: $spec" >&2; return 2; }

  ensure_micromamba

  local tmp
  tmp="$(mktemp -d)"
  local copy="$tmp/spec.yaml"
  sanitize_sweep_yaml_copy "$spec" "$copy"
  
  
  "$MMBIN" run -n mjepa env APP_DIR="$APP_DIR" SPEC="$copy" WANDB_PROJECT="$WANDB_PROJECT" WANDB_ENTITY="$WANDB_ENTITY" \
    python - <<'PY' | tail -n1 | tr -d '\r\n '
import os, yaml, wandb, os.path as p
with open(os.environ["SPEC"], "r") as f:
    spec = yaml.safe_load(f)
spec["program"] = p.join(os.environ["APP_DIR"], "scripts", "train_jepa.py")
spec["command"] = ["python", spec["program"], "sweep-run", "${args}"]
sid = wandb.sweep(spec, project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"])
print(sid)
PY
}
