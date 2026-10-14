#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run check_cpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 "$SCRIPT_ROOT/scripts/check_cpu.py"
