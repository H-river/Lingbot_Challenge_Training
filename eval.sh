#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/scripts/common.sh"
exec python3 "$LINGBOT_ROOT/reproduction/step2000/replay500.py" run "$@"
