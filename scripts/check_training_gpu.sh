#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run check_training_gpu
nvidia-smi
nvidia-smi topo -m
"$LINGBOT_ENV/bin/python" - <<'PY'
import torch,flash_attn
from flash_attn import flash_attn_func
assert torch.version.hip is None and torch.version.cuda=='12.8'
assert torch.cuda.is_available(), 'Platform must supply compatible driver/GPU; do not install host driver'
for i in range(torch.cuda.device_count()):
 with torch.cuda.device(i):
  print('GPU',i,torch.cuda.get_device_name(i),torch.cuda.get_device_capability(i))
  a=torch.randn(128,128,device='cuda',dtype=torch.bfloat16);b=a@a.T;assert torch.isfinite(b).all()
  q=torch.randn(1,32,4,64,device='cuda',dtype=torch.bfloat16);y=flash_attn_func(q,q,q);assert y.shape==q.shape and torch.isfinite(y).all();torch.cuda.synchronize()
import lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2
print('TRAINING_GPU_MATH_FLASH_AND_MODEL_IMPORT_PASS; no model weights, no simulation')
PY
