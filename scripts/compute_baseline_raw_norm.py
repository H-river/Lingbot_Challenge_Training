"""Clean-only baseline normalization, no image conversion or simulator calls.
Matches official next-frame actions, fixed 50-frame chunks with terminal padding,
and robot feature slicing. Uses pinned official RunningStats, batch size 256.
This is a disclosed baseline preprocessing configuration, not trained weights.
"""
import os,json,hashlib,sys
from pathlib import Path
import numpy as np
import h5py
import torch
from lingbotvla.utils import normalize
r=Path(os.environ['LINGBOT_ROOT']);p=r/'configs/clean_episodes.manifest.json';m=json.loads(p.read_text())
assert m['complete_2500'] and len(m['episodes'])==2500 and len(m['tasks'])==50
allowed=json.loads((r/'configs/clean_archives.manifest.json').read_text())
assert m['source']==allowed['repository'] and m['revision']==allowed['revision']
by_task={a['task']:a for a in allowed['archives']}; assert set(m['tasks'])==set(by_task)
columns={'arm.position':[0,1,2,3,4,5,7,8,9,10,11,12],'effector.position':[6,13]}
keys=[prefix+k for prefix in ['observation.state.','action.'] for k in columns]
stats={k:normalize.RunningStats() for k in keys};chunk=50

def read_raw(e):
 assert e['task'] in by_task and e['archive_sha256']==by_task[e['task']]['sha256']
 assert e['instruction_field']=='seen'
 path=r/e['path'];assert path.resolve().is_relative_to((r/'data/clean_raw/extracted').resolve())
 with h5py.File(path) as f:
  v=[f['joint_action/'+k][:] for k in ['left_arm','left_gripper','right_arm','right_gripper']]
 raw=np.concatenate([x.reshape(len(x),-1) for x in v],axis=1).astype(np.float32)
 assert raw.shape[1]==14 and np.isfinite(raw).all()
 return raw

def values(raw,idx):
 action=raw[1:];future=np.minimum(idx[:,None]+np.arange(chunk)[None,:],len(action)-1)
 return {**{'observation.state.'+k:raw[:-1][idx][:,c] for k,c in columns.items()},**{'action.'+k:action[future][:,:,c] for k,c in columns.items()}}

# Exact feature comparison against the native reader on the completed 50-episode task.
sys.path.insert(0,str(r/'scripts'))
from generated_offline_arguments import Arguments
from lingbotvla.utils.arguments import parse_args
from lingbotvla.data.vla_data.base_dataset import VLADataset
sys.argv=['check',str(r/'configs/competition_clean.template.yaml')]
a=parse_args(Arguments)
d=VLADataset(str(r/'data/clean_processed/lerobot/adjust_bottle'),'robotwin',a.data,str(r/'configs/robot_configs'),chunk_size=chunk,do_nomalize=False,return_item=True,disabled_image_features=True)
offset=0;checks=0
for e in sorted((e for e in m['episodes'] if e['task']=='adjust_bottle'),key=lambda e:e['episode']):
 raw=read_raw(e);n=len(raw)-1
 for i in sorted(set([0,min(49,n-1),n-1])):
  expected=values(raw,np.array([i]));item=d[offset+i]
  for key in keys:np.testing.assert_array_equal(np.asarray(item[key]),expected[key][0])
  checks+=1
 offset+=n
assert offset==len(d)
print('NATIVE_INPUT_EQUIVALENCE_PASS',checks,flush=True)
del d
frames=0;seen=set()
for e in sorted(m['episodes'],key=lambda e:(e['task'],e['episode'])):
 identity=(e['task'],e['episode']);assert identity not in seen and 0<=e['episode']<50;seen.add(identity)
 raw=read_raw(e);n=len(raw)-1
 for start in range(0,n,256):
  batch=values(raw,np.arange(start,min(n,start+256)))
  for key,x in batch.items():stats[key].update(x.reshape(-1,x.shape[-1]))
 frames+=n
 if e['episode']==49:print('NORM_TASK',e['task'],'frames_total',frames,flush=True)
assert len(seen)==2500
result={k:v.get_statistics() for k,v in stats.items()}
for k,obj in result.items():
 for field in ['mean','std','q01','q99','q02','q98','min','max']:assert np.isfinite(getattr(obj,field)).all(),(k,field)
 assert stats[k]._count==frames*(50 if k.startswith('action.') else 1)
out=r/'configs/baseline_clean_norm';out.mkdir(exist_ok=True)
assert not (out/'robotwin.json').exists(),'Refusing to overwrite existing baseline statistics'
normalize.save(out/'robotwin.json',result,frames)
provenance={'source':m['source'],'revision':m['revision'],'episode_manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'episodes':2500,'tasks':50,'frames':frames,'chunk_size':chunk,'native_input_equivalence_checks':checks,'implementation':'Pinned official RunningStats on clean raw joint arrays; next-frame actions, terminal clamping, official arm/gripper slicing','batch_size':256,'order':'task name then episode index','quantiles':'Official online 5000-bin histogram; batching differs from full dataset statistics, so quantiles need not be bit-identical','raw_image_read':False,'full_raw_files_rehashed_this_run':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_started':False}
(out/'input_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('BASELINE_CLEAN_NORM_COMPLETE',frames,flush=True)
