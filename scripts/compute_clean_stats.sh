#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run compute_clean_stats
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$LINGBOT_ENV/bin/python" - <<'PY'
import os,pathlib,json,hashlib
r=pathlib.Path(os.environ['LINGBOT_ROOT']);p=r/'configs/clean_episodes.manifest.json';m=json.loads(p.read_text())
assert m['complete_2500'] and len(m['episodes'])==2500 and len(m['tasks'])==50
allowed=set(m['tasks']);lines=(r/'configs/train_clean.txt').read_text().splitlines()
assert len(lines)==50
for line in lines:
 robot,path=line.split();d=pathlib.Path(path)
 assert robot=='robotwin' and d.name in allowed and d.parent==r/'data/clean_processed/lerobot'
 assert (d/'meta/info.json').is_file(), f'LeRobot conversion missing: {d}'
 info=json.loads((d/'meta/info.json').read_text());assert info['total_episodes']==50
 assert not any(t in path for t in ['randomized','unseen'])
(r/'configs/clean_norm').mkdir(exist_ok=True)
(r/'configs/clean_norm/input_provenance.json').write_text(json.dumps({'episode_manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'train_list_sha256':hashlib.sha256((r/'configs/train_clean.txt').read_bytes()).hexdigest(),'source':m['source'],'revision':m['revision'],'script':'official compute_norm_stats.py with import-only CPU adapter; see docs/patches/cpu-norm-imports.patch','data_ratio':1.0},indent=2)+'\n')
PY
"$LINGBOT_ENV/bin/python" "$SCRIPT_ROOT/scripts/prepare_offline_stats.py"
cd "$LINGBOT_ROOT"
# Single process, CPU-only statistics; no train.sh, torchrun, or simulator.
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 CUDA_VISIBLE_DEVICES='' "$LINGBOT_ENV/bin/python" scripts/generated_offline_norm_stats.py repos/lingbot-vla-v2/configs/vla/norm_compute/post_data.yaml \
 --data.datasets_type vla --data.data_name multi \
 --data.train_path "$LINGBOT_ROOT/configs/train_clean.txt" \
 --data.robot_config_root "$LINGBOT_ROOT/configs/robot_configs" \
 --data.robot_name robotwin --data.data_ratio_for_norm_compute 1.0 \
 --data.norm_path "$LINGBOT_ROOT/configs/clean_norm/robotwin.json" \
 --data.num_workers 1 --train.micro_batch_size 16
