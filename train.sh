#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec bash "$ROOT/scripts/launch_clean_training.sh" "${1:?Usage: bash train.sh /path/to/reviewed-clean-config.yaml}"
