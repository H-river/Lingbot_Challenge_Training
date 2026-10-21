#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run launch_clean_training
cfg=${1:?Usage: launch_clean_training.sh config.yaml};[[ -f "$cfg" ]]
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
export PYTHONPATH="$LINGBOT_ROOT/repos/lingbot-vla-v2:$LINGBOT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/audit_training_ready.py"
readarray -t launch < <("$LINGBOT_ENV/bin/python" - "$cfg" <<'PY'
import os,sys,pathlib,yaml,torch,shutil
r=pathlib.Path(os.environ['LINGBOT_ROOT']);c=yaml.safe_load(pathlib.Path(sys.argv[1]).read_text());n=torch.cuda.device_count();assert n>0 and torch.cuda.is_available();a=c['train'];assert a['global_batch_size']==a['micro_batch_size']*n*a['gradient_accumulation_steps'];assert c['data']['train_path']==str(r/'configs/train_clean.txt') and c['data']['norm_stats_file']==str(r/'configs/clean_norm/robotwin.json');assert c.get('eval',{}).get('enabled') is False and not c.get('eval',{}).get('command');assert c['model']['model_path']==str(r/'models/lingbot-vla-v2-6b')
output=pathlib.Path(a['output_dir']);assert output.is_relative_to(r/'checkpoints') and output!=r/'checkpoints';assert not output.exists() or not any(output.iterdir()),'Output exists: choose fresh directory; no implicit resume'
output.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(output.parent).free>60*1024**3,'CHECKPOINT_SPACE_PENDING: require >60 GiB for formal-save test'
# ORACLE control experiment (evaluation textures in the pool): explicit opt-in, labelled output, sibling marker. Never submit.
oracle='ORACLE' in str((((c.get('data') or {}).get('aug') or {}).get('texture') or {}).get('source',''))
assert not oracle or os.environ.get('LINGBOT_UNSEEN_TEXTURE_ORACLE')=='NOT_FOR_SUBMISSION','ORACLE config: set LINGBOT_UNSEEN_TEXTURE_ORACLE=NOT_FOR_SUBMISSION to confirm this run is a non-submitted control experiment'
assert not oracle or 'ORACLE' in output.name,'ORACLE config: output_dir name must contain ORACLE'
assert oracle or 'ORACLE' not in output.name,'Non-oracle run must not use an ORACLE output_dir'
oracle and (output.parent/(output.name+'.ORACLE_NOT_FOR_SUBMISSION')).write_text('Trained with RoboTwin EVALUATION (unseen) background textures. Control experiment only. Do NOT submit, export or fine-tune from this checkpoint.\n')
entry=r/'scripts/train_frozen_clean.py' if a['train_expert_only'] else r/'repos/lingbot-vla-v2/tasks/vla/train_lingbotvla.py';assert entry.is_file();print(n);print(entry)
PY
)
[[ ${#launch[@]} == 2 && ${launch[0]} =~ ^[1-9][0-9]*$ ]] || { echo 'PRELAUNCH_VALIDATION_FAILED'; exit 2; }
cd "$LINGBOT_ROOT/repos/lingbot-vla-v2"
"$LINGBOT_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node="${launch[0]}" "${launch[1]}" "$cfg"
