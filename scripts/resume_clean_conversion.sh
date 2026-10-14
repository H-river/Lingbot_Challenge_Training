#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run resume_clean_conversion_handoff
mkdir -p "$LINGBOT_ROOT/cache/locks" "$LINGBOT_ROOT/results/conversion_validation"
exec 9>"$LINGBOT_ROOT/cache/locks/clean-handoff-conversion.lock"
flock -n 9 || { echo 'Another handoff conversion is running'; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Serial: the imported official adapter is generated at one shared path.
"$LINGBOT_ENV/bin/python" - <<'PY'
import os,pathlib,json,subprocess,datetime,hashlib
r=pathlib.Path(os.environ['LINGBOT_ROOT']);m=json.loads((r/'configs/clean_episodes.manifest.json').read_text());assert m['source']=='TianxingChen/RoboTwin2.0' and m['revision']=='3dc3b798668feb99ac61cc9086d84cbcc3d79186';assert m['complete_2500'] and len(m['tasks'])==50 and len(m['episodes'])==2500
state={'state':'RUNNING','completed_tasks':[],'phase':'conversion','manifest_sha256':hashlib.sha256((r/'configs/clean_episodes.manifest.json').read_bytes()).hexdigest(),'simulation_started':False,'training_started':False}
def save():
 state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=r/'results/clean_handoff_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2));q.replace(p)
try:
 for task in m['tasks']:
  state.update(current_task=task,phase='conversion');save()
  subprocess.run(['bash',str(r/'scripts/convert_clean.sh'),'--task',task],check=True,cwd=r)
  state['phase']='validate';save();subprocess.run([os.environ['LINGBOT_ENV']+'/bin/python',str(r/'scripts/validate_converted_clean.py'),task],check=True,cwd=r)
  state['completed_tasks'].append(task);save()
 state.update(current_task=None,phase='normalization');save()
 if (r/'configs/clean_norm/robotwin.json').exists():
  p=json.loads((r/'configs/clean_norm/input_provenance.json').read_text());assert p['episode_manifest_sha256']==state['manifest_sha256'];assert p['train_list_sha256']==hashlib.sha256((r/'configs/train_clean.txt').read_bytes()).hexdigest();print('EXISTING_FULL_CLEAN_NORM: source matched, preserved')
 else:subprocess.run(['bash',str(r/'scripts/compute_clean_stats.sh')],check=True,cwd=r)
 subprocess.run([os.environ['LINGBOT_ENV']+'/bin/python',str(r/'scripts/audit_training_ready.py')],check=True,cwd=r)
 state.update(state='DATA_TRAIN_READY_50',phase='await_gpu_checkpoint_validation');save()
except BaseException as e:
 state.update(state='FAILED',error=type(e).__name__+': '+str(e));save();raise
PY
