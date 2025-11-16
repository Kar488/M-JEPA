#!/usr/bin/env bash
set -euo pipefail

log_path="${MJEPA_FAKE_SUDO_LOG:-}"
if [[ -n "$log_path" ]]; then
  printf '%s\n' "$*" >> "$log_path"
fi

require_tty="${MJEPA_FAKE_SUDO_REQUIRE_TTY:-0}"
if [[ "$require_tty" == "1" ]] && [[ ! -t 0 ]]; then
  exit 1
fi

args=()
for arg in "$@"; do
  if [[ "$arg" == "-n" ]]; then
    continue
  fi
  args+=("$arg")
done

fix_path="${MJEPA_FAKE_SUDO_FIX_PATH:-}"
if [[ -n "$fix_path" ]]; then
  case "${args[0]:-}" in
    mkdir)
      chmod u+w "$fix_path" 2>/dev/null || true
      ;;
    chown)
      exit 0
      ;;
    chmod)
      chmod u+w "$fix_path" 2>/dev/null || true
      exit 0
      ;;
  esac
fi

exec "${args[@]}"
