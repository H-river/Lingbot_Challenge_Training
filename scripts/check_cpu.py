#!/usr/bin/env python3
import os,pathlib,json,subprocess,sys,hashlib
r=pathlib.Path(os.environ['LINGBOT_ROOT']);results=[]
def record(name,status,detail):
 results.append({'name':name,'status':status,'detail':detail});print(status,name,detail,flush=True)
def run(name,cmd,gpu=False):
 try:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=120);detail=(p.stdout+p.stderr).strip()
  status='PASS' if p.returncode==0 else 'FAIL'
  if p.returncode and name=='lingbotvla pip check' and (r/'configs/accept_upstream_metadata_conflicts').is_file():
   known=r/'configs/known_lingbot_pip_check.txt'
   if known.is_file() and set(detail.splitlines()) <= set(known.read_text().splitlines()):status='KNOWN_METADATA_CONFLICT' 
  if p.returncode and gpu and any(x in detail.lower() for x in ['no nvidia driver','cuda driver','libcuda.so','cuda is not available','no cuda gpus','libcudart.so']):status='GPU_PENDING'
  record(name,status,detail[-5000:])
 except Exception as e:record(name,'FAIL',repr(e))
for name,version,modules in [
 ('lingbotvla','3.12',['torch','torchvision','torchaudio','torchcodec','transformers','datasets','h5py','cv2','accelerate','mlflow','lerobot','moge','moge.model.v2','mdm','mdm.model.v2','utils3d','flash_attn','lingbotvla.data.vla_data.base_dataset','lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2']),
 ('robotwin','3.10',['torch','torchvision','h5py','cv2','sapien','mplib','open3d','transforms3d'])]:
 py=r/'envs'/name/'bin/python'
 if not py.exists():record(name,'FAIL','environment missing');continue
 run(name+' Python',[str(py),'-c',f'import sys; print(sys.version); assert sys.version.startswith({version!r})'])
 run(name+' pip check',[str(py),'-m','pip','check'])
 for mod in modules:run(name+' import '+mod,[str(py),'-c',f'import importlib; m=importlib.import_module({mod!r}); print(getattr(m,"__version__","import OK"))'],True)
 run(name+' torch backend',[str(py),'-c','import torch; print(torch.__version__,"CUDA build",torch.version.cuda,"available",torch.cuda.is_available()); assert torch.version.cuda is not None'])
manifest=r/'configs/clean_episodes.manifest.json'
if not manifest.exists():manifest=r/'configs/clean_episodes_adjust_bottle.manifest.json'
if manifest.exists():
 m=json.loads(manifest.read_text());allowed=json.loads((r/'configs/clean_archives.manifest.json').read_text());tasks={x['task']:x for x in allowed['archives']}
 try:
  assert m['source']==allowed['repository'] and m['revision']==allowed['revision']
  assert len({(e['task'],e['episode']) for e in m['episodes']})==len(m['episodes'])
  for e in m['episodes']:
   assert e['task'] in tasks and 0<=e['episode']<50 and e['archive_sha256']==tasks[e['task']]['sha256']
   assert e['instruction_field']=='seen'
   for key in ['path','instructions_path']:
    p=r/e[key];assert p.resolve().is_relative_to((r/'data/clean_raw/extracted').resolve());assert p.is_file()
    assert not any(x in str(p).lower() for x in ['randomized','unseen'])
   assert set(json.loads((r/e['instructions_path']).read_text()))=={'seen'}
  record('clean provenance manifest','PASS',f'{len(m["episodes"])} episode paths verified')
  record('full clean 2500','PASS' if m['complete_2500'] and len(m['episodes'])==2500 and len(m['tasks'])==50 else 'DATA_PENDING',f'{len(m["tasks"])} tasks, {len(m["episodes"])} episodes')
 except Exception as e:record('clean provenance manifest','FAIL',repr(e))
 ep=m['episodes'][0]
 code='''import h5py,numpy as np,cv2,sys
with h5py.File(sys.argv[1]) as f:
 a=[f['joint_action/'+k][:] for k in ['left_arm','left_gripper','right_arm','right_gripper']]
 state=np.concatenate([x.reshape(len(x),-1) for x in a],axis=1)
 assert state.shape[1]==14 and len(state)>1 and np.isfinite(state).all()
 print('state/action',state[:-1].shape,state[1:].shape)
 for cam in ['head_camera','left_camera','right_camera']:
  img=cv2.imdecode(np.frombuffer(f['observation/'+cam+'/rgb'][0],np.uint8),cv2.IMREAD_COLOR)
  assert img is not None and img.ndim==3;print(cam,img.shape)
'''
 run('clean episode read/image/state/action',[str(r/'envs/robotwin/bin/python'),'-c',code,str(r/ep['path'])])
else:record('clean episode manifest','DATA_PENDING','No verified episode manifest')
run('clean configuration and training-path isolation',[str(r/'envs/robotwin/bin/python'),'-c',"""import pathlib,sys,yaml,json
r=pathlib.Path(sys.argv[1]);c=yaml.safe_load((r/'configs/competition_clean.template.yaml').read_text());a=json.loads((r/'configs/clean_archives.manifest.json').read_text());tasks={e['task'] for e in a['archives']}
assert c['model']['model_path']==str(r/'models/lingbot-vla-v2-6b')
assert c['data']['norm_stats_file']==str(r/'configs/clean_norm/robotwin.json')
lines=(r/'configs/train_clean.txt').read_text().splitlines();assert len(lines)==50
paths=[pathlib.Path(line.split()[1]) for line in lines];assert {p.name for p in paths}==tasks
assert all(p.parent==r/'data/clean_processed/lerobot' for p in paths)
assert not any(x in str(paths) for x in ['randomized','unseen'])
rc=yaml.safe_load((r/'configs/robot_configs/robotwin.yaml').read_text());assert rc['norm_stats']==c['data']['norm_stats_file']
print('50 unique clean tasks; allowed base; isolated clean stats paths')
""",str(r)])

if (r/'scripts/generated_offline_arguments.py').is_file():
 run('official dataclass configuration parsing without model import',[str(r/'envs/lingbotvla/bin/python'),'-c',"""import sys,os
os.environ.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1')
sys.path.insert(0,sys.argv[1]+'/scripts')
from generated_offline_arguments import Arguments
from lingbotvla.utils.arguments import parse_args
sys.argv=['check',sys.argv[1]+'/configs/competition_clean.template.yaml']
a=parse_args(Arguments);print(a.model.model_path,a.data.train_path,a.train.chunk_size)
""",str(r)])
else:record('official dataclass configuration parsing','DATA_PENDING','Generate offline argument definitions first')

assets=json.loads((r/'configs/assets.manifest.json').read_text())
for key,dest in [('models','models/lingbot-vla-v2-6b'),('qwen','models/Qwen3-VL-4B-Instruct-config'),('moge','models/MoGe-2-vitb-normal')]:
 missing=[f['rfilename'] for f in assets[key]['files'] if not (r/dest/f['rfilename']).exists() or (r/dest/f['rfilename']).stat().st_size!=f['size']]
 record(key+' files','ASSET_PENDING' if missing else 'PASS',missing)
run('Qwen config/tokenizer (no model load)',[str(r/'envs/lingbotvla/bin/python'),'-c','from transformers import AutoConfig,AutoTokenizer; import sys; c=AutoConfig.from_pretrained(sys.argv[1],local_files_only=True); t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True); print(c.model_type,len(t))',str(r/'models/Qwen3-VL-4B-Instruct-config')])
for repo,rev in [('lingbot-vla-v2','be969b8fd117fb70550c5d4bf4bc328211b5b1b6'),('RoboTwin','13c3c47ff4312dd62484bcd51be034af55c062d1')]:
 p=subprocess.run(['git','-C',str(r/'repos'/repo),'rev-parse','HEAD'],text=True,capture_output=True)
 record(repo+' revision','PASS' if p.returncode==0 and p.stdout.strip()==rev else 'FAIL',p.stdout+p.stderr)
for name in ['lingbotvla.pip-freeze.txt','robotwin.pip-freeze.txt','lingbotvla.conda-explicit.txt','robotwin.conda-explicit.txt','git-revisions.txt']:
 record('version export '+name,'PASS' if (r/'docs/versions'/name).is_file() else 'FAIL',name)

for archive in ['embodiments.zip','objects.zip','background_texture.zip']:
 marker=r/'assets/simulation'/(archive+'.prepared.json')
 record('simulation asset '+archive,'PASS' if marker.is_file() else 'ASSET_PENDING','Prepared without simulation' if marker.is_file() else 'Download/extraction not complete')
for package in ['curobo','pytorch3d']:
 run('deferred GPU extension inventory '+package,[str(r/'envs/robotwin/bin/python'),'-c',f'import importlib.util; print(importlib.util.find_spec({package!r}))'])
 record(package+' compiled execution','GPU_PENDING','Intentionally deferred until actual GPU/driver architecture detection; setup_gpu.sh not executed')
sample=r/'data/clean_processed/smoke_lerobot/adjust_bottle'
if (sample/'conversion_provenance.json').is_file():
 run('converted clean LeRobot episode and video decode',[str(r/'envs/lingbotvla/bin/python'),'-c',"""import sys,torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
p=Path(sys.argv[1]);d=LeRobotDataset(p.name,root=p);assert d.meta.total_episodes==1
f=d[0]
for k in ['observation.state','action']:
 assert f[k].shape==(14,) and torch.isfinite(f[k]).all();print(k,f[k].shape)
for k in ['observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist']:
 assert f[k].shape==(3,480,640) and torch.isfinite(f[k]).all();print(k,f[k].shape)
print('frames',len(d))
""",str(sample)])
else:record('converted clean LeRobot sample','DATA_PENDING','Sample conversion not verified')
norm=r/'configs/clean_norm/robotwin.json'
record('clean-only normalization','PASS' if norm.is_file() and (norm.parent/'input_provenance.json').is_file() else 'DATA_PENDING','Only a fully verified clean manifest may produce these statistics')

(r/'results/check_cpu.json').write_text(json.dumps(results,indent=2)+'\n')
sys.exit(1 if any(x['status']=='FAIL' for x in results) else 3 if any(x['status']!='PASS' for x in results) else 0)
