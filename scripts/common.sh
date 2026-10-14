#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${LINGBOT_PATHS_FILE:-$SCRIPT_ROOT/configs/paths.env}"
: "${LINGBOT_ROOT:?}" "${LINGBOT_ENV:?}" "${ROBOTWIN_ENV:?}"
if [[ -n "${LINGBOT_STORAGE_MOUNT:-}" ]]; then
  mountpoint -q "$LINGBOT_STORAGE_MOUNT" || { echo "DATA_DISK_NOT_MOUNTED: $LINGBOT_STORAGE_MOUNT" >&2; exit 2; }
  [[ "$(findmnt -n -o UUID --target "$LINGBOT_STORAGE_MOUNT")" = "$LINGBOT_STORAGE_UUID" ]] || { echo "DATA_DISK_UUID_MISMATCH" >&2; exit 2; }
fi
[[ "$LINGBOT_ROOT" = /* && "$LINGBOT_ROOT" != / ]] || exit 2
mkdir -p "$LINGBOT_ROOT"/{logs,cache/tmp,cache/pip,cache/conda,cache/xdg,cache/huggingface,cache/torch,cache/torch_extensions}
log_run() {
  exec > >(tee -a "$LINGBOT_ROOT/logs/$1-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
  trap 'rc=$?; printf "FAILED line=%s exit=%s\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR
}
