#!/usr/bin/env python3
"""Use pinned official DCP->HF conversion; CPU state merge for export, no inference."""
import datetime,hashlib,json,os,pathlib,shutil,sys,time
import torch,yaml
from torch.distributed.checkpoint import FileSystemReader
from lingbotvla.checkpoint import ckpt_to_state_dict
from lingbotvla.models import save_model_weights
cfg=pathlib.Path(sys.argv[1]);c=yaml.safe_load(cfg.read_text());out=pathlib.Path(c['train']['output_dir']);cp=out/'checkpoints'/f"global_step_{c['train']['max_steps']}";dest=cp/'hf_ckpt'
assert not dest.exists(),'Never overwrite an existing HF export'
meta=FileSystemReader(cp/'model').read_metadata()
numel=sum(__import__('math').prod(v.size) for v in meta.state_dict_metadata.values() if hasattr(v,'size'))
expected=numel*2;free=shutil.disk_usage(cp).free;assert free>expected+5*1024**3,'HF_EXPORT_SPACE_PENDING: keep 5 GiB reserve'
started=time.monotonic();print('OFFICIAL_HF_EXPORT_START',json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'estimated_bytes':expected,'free_bytes':free,'CPU_state_merge_for_export':True,'inference':False}),flush=True)
state=ckpt_to_state_dict(cp,out,ckpt_manager='dcp')
save_model_weights(dest,state,save_dtype=torch.bfloat16)
del state
for f in (out/'model_assets').iterdir():
    if f.is_file():shutil.copyfile(f,dest/f.name)
report={'state':'OFFICIAL_BF16_HF_EXPORT_CREATED','HF_path':str(dest),'elapsed_s':time.monotonic()-started,'CPU_state_merge_for_export':True,'CPU_inference':False,'checkpoint_source':str(cp),'bytes':sum(f.stat().st_size for f in dest.iterdir() if f.is_file())}
(out/'HF_export_receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
