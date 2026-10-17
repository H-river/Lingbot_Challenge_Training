#!/bin/bash
set -e
cd /root/lingbot_challenge
source scripts/common.sh
exec python3 results/step2000_bf16/run_scene_recovery.py
