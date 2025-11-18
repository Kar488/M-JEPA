#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENTS_ROOT:?EXPERIMENTS_ROOT is not set}"
: "${MJEPA_DIR_MODE:=0775}"

owner_uid="${MJEPA_DIR_OWNER_UID:-}" 
owner_gid="${MJEPA_DIR_OWNER_GID:-}"

if [[ -z "$owner_uid" ]]; then
  if [[ -n "${SUDO_UID:-}" ]]; then
    owner_uid="$SUDO_UID"
  else
    owner_uid="$(stat -Lc '%u' "$EXPERIMENTS_ROOT" 2>/dev/null || true)"
  fi
fi

if [[ -z "$owner_gid" ]]; then
  if [[ -n "${SUDO_GID:-}" ]]; then
    owner_gid="$SUDO_GID"
  else
    owner_gid="$(stat -Lc '%g' "$EXPERIMENTS_ROOT" 2>/dev/null || true)"
  fi
fi

if [[ -z "$owner_uid" ]]; then
  owner_uid="$(id -u)"
fi
if [[ -z "$owner_gid" ]]; then
  owner_gid="$owner_uid"
fi

printf '[repair-experiments-root] ensuring %s owned by %s:%s with mode %s\n' \
  "$EXPERIMENTS_ROOT" "$owner_uid" "$owner_gid" "$MJEPA_DIR_MODE"

install -d -m "$MJEPA_DIR_MODE" -o "$owner_uid" -g "$owner_gid" "$EXPERIMENTS_ROOT"

printf '[repair-experiments-root] repaired %s\n' "$EXPERIMENTS_ROOT"
