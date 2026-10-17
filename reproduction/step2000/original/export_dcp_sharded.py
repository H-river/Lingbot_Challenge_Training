import os,json,time,gc,fcntl,hashlib
from pathlib import Path
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from safetensors.torch import save_file
from safetensors import safe_open
R=Path('/root/lingbot_challenge/results/step2000_bf16'); C=Path('/root/deliverables/trained_ckpt/global_step_2000')
lock=open(R/'export.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
reader=FileSystemReader(C/'model');m=reader.read_metadata()
for name in ('model','optimizer'):
 md=FileSystemReader(C/name).read_metadata()
 for v in md.storage_data.values():
  p=C/name/v.relative_path
  assert p.exists() and p.stat().st_size>=v.offset+v.length,str(p)
O=C/'hf_ckpt';O.mkdir(exist_ok=True)
assert not (O/'model.safetensors.index.json').exists(),'Already exported; validate existing output'
keys=sorted(m.state_dict_metadata);groups=[];group=[];size=0
for k in keys:
 v=m.state_dict_metadata[k];n=v.size.numel()*torch.empty((),dtype=v.properties.dtype).element_size()
 if group and size+n>1_500_000_000:groups.append(group);group=[];size=0
 group.append(k);size+=n
groups.append(group)
weight_map={};total=0;manifest=[];started=time.time()
for i,group in enumerate(groups,1):
 sd={};_load_state_dict(sd,storage_reader=reader,planner=_EmptyStateDictLoadPlanner(keys=set(group)),no_dist=True)
 state=sd['state']['model'];expected={k.removeprefix('state.model.') for k in group}
 assert set(state)==expected,(len(state),len(expected))
 for k,t in state.items():
  assert tuple(t.shape)==tuple(m.state_dict_metadata['state.model.'+k].size)
  assert torch.isfinite(t).all(),k
  total+=t.numel()*t.element_size()
 name=f'model-{i:05d}-of-{len(groups):05d}.safetensors';tmp=O/(name+'.partial')
 save_file(state,str(tmp),metadata={'format':'pt'})
 with safe_open(tmp,framework='pt') as f:
  assert set(f.keys())==expected
  for k in expected:assert torch.equal(f.get_tensor(k),state[k]),k
 os.replace(tmp,O/name)
 digest=hashlib.sha256()
 with (O/name).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
 manifest.append({'file':name,'sha256':digest.hexdigest(),'keys':len(expected)})
 weight_map.update({k:name for k in expected})
 (R/'export_progress.json').write_text(json.dumps({'state':'EXPORTING','shards_done':i,'shards_total':len(groups),'keys_done':len(weight_map),'elapsed_seconds':time.time()-started}))
 print('EXPORTED',i,len(groups),len(weight_map),flush=True)
 del sd,state;gc.collect()
assert len(weight_map)==len(keys)
(O/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':weight_map},indent=2))
(R/'export_verified.json').write_text(json.dumps({'state':'EXPORTED_VERIFIED','source':str(C/'model'),'output':str(O),'keys':len(keys),'total_bytes':total,'exact_tensor_roundtrip':True,'finite':True,'dtype':'source_preserved','shards':manifest,'elapsed_seconds':time.time()-started},indent=2))
print('EXPORT_VERIFIED',len(keys),flush=True)
