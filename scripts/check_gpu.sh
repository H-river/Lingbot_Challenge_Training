#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run check_gpu
nvidia-smi
for env in "$LINGBOT_ENV" "$ROBOTWIN_ENV"; do
 "$env/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available()
a=torch.randn(128,128,device='cuda');b=a@a.T
assert torch.isfinite(b).all();torch.cuda.synchronize()
print('PASS GPU matmul',torch.__version__,torch.version.cuda,torch.cuda.get_device_name())
PY
done
"$LINGBOT_ENV/bin/python" - <<'PY'
import torch,flash_attn
from flash_attn import flash_attn_func
q=torch.randn(1,32,4,64,device='cuda',dtype=torch.float16)
y=flash_attn_func(q,q,q);assert y.shape==q.shape and torch.isfinite(y).all()
torch.cuda.synchronize();print('PASS flash-attn',flash_attn.__version__)
PY
"$ROBOTWIN_ENV/bin/python" - <<'PY'
import torch,curobo,warp as wp
wp.init(); w=wp.zeros(4,device='cuda'); wp.synchronize(); print('PASS Warp GPU allocation',wp.__version__)
from curobo.curobolib import geom,kinematics
from pytorch3d.ops import knn_points
x=torch.randn(1,16,3,device='cuda');y=knn_points(x,x,K=1)
assert torch.isfinite(y.dists).all();torch.cuda.synchronize()
print('PASS cuRobo compiled imports and PyTorch3D GPU operation')
PY
echo 'GPU math/extension checks passed. No model inference, simulation rollout, rendering or training was performed.'
