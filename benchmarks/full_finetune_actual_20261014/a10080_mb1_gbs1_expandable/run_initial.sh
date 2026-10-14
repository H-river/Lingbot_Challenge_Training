#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
export LINGBOT_ROOT=${LINGBOT_ROOT:-/root/lingbot_challenge}
source "$LINGBOT_ROOT/scripts/common.sh"
log_run full_finetune_single_a100_actual
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$LINGBOT_ROOT/repos/lingbot-vla-v2:$LINGBOT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="/root/lingbot_challenge/benchmarks/full_finetune_actual_20261014/a10080_mb1_gbs1_r2/cache/inductor_a100"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG="$HERE/full_clean50_smoke.yaml"
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/audit_training_ready.py"
"$LINGBOT_ENV/bin/python" - "$CFG" "$HERE" <<'PY'
import ast,datetime,hashlib,json,math,pathlib,shutil,sys,torch,yaml
from safetensors import safe_open
r=pathlib.Path('/root/lingbot_challenge');c=yaml.safe_load(pathlib.Path(sys.argv[1]).read_text());here=pathlib.Path(sys.argv[2]);a=c['train']
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.cuda.get_device_properties(0).total_memory>75*1024**3
assert a['global_batch_size']==a['micro_batch_size']==a['gradient_accumulation_steps']==1
assert not any(a[k] for k in ['freeze_vit','freeze_vision_encoder','train_expert_only'])
assert a['max_steps']==11 and a['save_steps']==11 and not a['save_hf_weights']
assert c['eval']['enabled'] is False and not c['eval']['command']
assert c['model']['model_path']==str(r/'models/lingbot-vla-v2-6b')
output=pathlib.Path(a['output_dir']);assert output.parent==r/'checkpoints'
assert not output.exists() or not any(output.iterdir()),'Never overwrite an existing run'
memtotal=next(int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:'))
assert memtotal>40*1024**3,'This run is intended for the actual 48GiB host; export/resume remains pending'
index=json.loads((r/'models/lingbot-vla-v2-6b/model.safetensors.index.json').read_text())['weight_map']
param_numel=0;adam_numel=0
for shard in sorted(set(index.values())):
 with safe_open(r/'models/lingbot-vla-v2-6b'/shard,framework='pt',device='cpu') as f:
  for key in f.keys():
   shape=f.get_slice(key).get_shape();n=math.prod(shape);param_numel+=n
   if len(shape) not in (2,3) or any(s in key.lower() for s in ['embed_tokens','embedding','lm_head','output_layer']):adam_numel+=n
# FP32 model + one Muon momentum; Adam subset needs another moment. Add a generous metadata/IO margin.
estimated=(param_numel*8+adam_numel*4)/1024**3;free=shutil.disk_usage(output.parent).free/1024**3
assert free>estimated+10,f'CHECKPOINT_SPACE_PENDING: estimate {estimated:.2f}GiB +10GiB reserve, free {free:.2f}'
proof={'GPU':torch.cuda.get_device_name(0),'RAM_GiB':memtotal/1024**3,'checkpoint_estimated_GiB':estimated,'checkpoint_free_GiB':free,'estimate_scope':'header-only conservative parameter count; no full CPU model load','config_sha256':hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(),'norm_sha256':hashlib.sha256((r/'configs/clean_norm/robotwin.json').read_bytes()).hexdigest(),'trainer_sha256':hashlib.sha256((r/'repos/lingbot-vla-v2/tasks/vla/train_lingbotvla.py').read_bytes()).hexdigest()}
(here/'prelaunch_proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
(here/'RUNNING.json').write_text(json.dumps({'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output_dir':str(output),'scope':'11 full-training steps plus DCP; HF/resume pending','training_started':True},indent=2))
PY
cd "$LINGBOT_ROOT/repos/lingbot-vla-v2"
timeout --signal=TERM --kill-after=30s 20m "$LINGBOT_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 "$HERE/train_with_memory_record.py" "$CFG" 2>&1 | tee "$HERE/initial.log"
"$LINGBOT_ENV/bin/python" "$HERE/verify_training_log.py" "$CFG" "$HERE/initial.log" initial
"$LINGBOT_ENV/bin/python" "$HERE/verify_dcp.py" "$CFG"
echo FULL_TRAINING_AND_DCP_PASS_HF_AND_RESUME_PENDING
