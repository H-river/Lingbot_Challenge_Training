#!/usr/bin/env bash
# Reproduction chain: base -> 8k (actual original config/provenance required)
# -> augmentation finetuning (configs/train_8xA100_aug_v1.yaml / v2.yaml).
# The augmentation configs are BLOCKED drafts until 8k provenance is reconciled.
# This launcher only accepts the base model; resume restores optimizer state.
# A reviewed weights-only launch path is still required; see AUG_REPORT.md.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec bash "$ROOT/scripts/launch_clean_training.sh" "${1:?Usage: bash train.sh /path/to/reviewed-clean-config.yaml}"
