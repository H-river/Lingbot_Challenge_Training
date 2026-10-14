#!/usr/bin/env python3
"""Inspect trusted local DCP and compare selected projection tensors, not the entire model."""
import json,math,pathlib,sys,torch,yaml
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
c=yaml.safe_load(pathlib.Path(sys.argv[1]).read_text());a=c['train'];out=pathlib.Path(a['output_dir']);cp=out/'checkpoints'/f"global_step_{a['max_steps']}"
for folder in ['model','optimizer']:assert (cp/folder/'.metadata').is_file()
state=torch.load(cp/'extra_state/extra_state_rank_0.pt',map_location='cpu',weights_only=False)
assert state['global_step']==11 and all(k in state for k in ['lr_scheduler','train_dataloader','torch_rng_state','environ_meter'])
base=pathlib.Path(c['model']['model_path']);index=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
reader=dcp.FileSystemReader(cp/'model');metadata=reader.read_metadata()
numel=sum(math.prod(v.size) for v in metadata.state_dict_metadata.values() if hasattr(v,'size'));assert numel>6_000_000_000
groups={'VLM_language':lambda k:'.qwenvl.' in k and '.language_model.layers.' in k,'VLM_vision':lambda k:'.qwenvl.' in k and '.visual.blocks.' in k,'action_expert':lambda k:'.qwen_expert.model.layers.' in k}
updates={}
for group,pred in groups.items():
 rows=[]
 for key in sorted(k for k in index if pred(k) and k.endswith('weight') and ('proj' in k or '.attn.' in k))[:8]:
  meta=metadata.state_dict_metadata.get('state.model.'+key)
  if meta is None:continue
  shape=list(meta.size)
  if len(shape)!=2:continue
  tensor=torch.empty(shape,dtype=meta.properties.dtype,device='cpu');target={'state':{'model':{key:tensor}}}
  dcp.load(target,storage_reader=reader)
  cut=(slice(0,min(128,shape[0])),slice(0,min(128,shape[1])))
  with safe_open(base/index[key],framework='pt',device='cpu') as f:before=f.get_slice(key)[cut].to(tensor.dtype)
  after=tensor[cut];assert torch.isfinite(after).all();delta=(after.float()-before.float()).abs()
  rows.append({'key':key,'sampled_elements':delta.numel(),'changed_elements':int((delta>0).sum()),'max_abs_change':float(delta.max())});del tensor,target
  if rows[-1]['changed_elements']>0:break
 assert rows and any(x['changed_elements']>0 for x in rows),f'{group} update not established by sampled tensors'
 updates[group]=rows
report={'state':'FULL_TRAINING_DCP_AND_SAMPLED_UPDATES_VERIFIED','steps':11,'DCP_numel':numel,'checkpoint':str(cp),'updates':updates,'HF_export':'PENDING','GPU_resume':'PENDING','full_6B_CPU_load':False}
(out/'dcp_verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
