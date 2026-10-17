#!/usr/bin/env bash
set -Eeuo pipefail
source /root/lingbot_challenge/scripts/common.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
export QWEN3VL_PATH="$LINGBOT_ROOT/models/Qwen3-VL-4B-Instruct-config"
cd "$LINGBOT_ROOT/repos/lingbot-vla-v2"
exec "$LINGBOT_ENV/bin/python" -u -m deploy.lingbot_vla_v2_policy_base --model_path "$LINGBOT_ROOT/eval_step2000/checkpoints/global_step_2000/hf_ckpt" --use_length 50 --chunk_ret True --port "${1:-9430}" --use_bf16 True --use_fp32 False --use_compile False
