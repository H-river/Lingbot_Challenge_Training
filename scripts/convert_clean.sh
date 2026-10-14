#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run convert_clean
"$LINGBOT_ENV/bin/python" "$SCRIPT_ROOT/scripts/convert_clean.py" "$@"
