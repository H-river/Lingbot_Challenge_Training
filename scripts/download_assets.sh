#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run download_assets
# Explicit scopes: base, clean [--task NAME], simulation. No wildcard download.
python3 "$SCRIPT_ROOT/scripts/download_assets.py" "$@"
