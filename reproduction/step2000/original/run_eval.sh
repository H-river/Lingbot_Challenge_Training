#!/usr/bin/env bash
set -Eeuo pipefail
source /root/lingbot_challenge/scripts/common.sh
exec python3 -u /root/lingbot_challenge/results/step2000_bf16/run_eval.py
