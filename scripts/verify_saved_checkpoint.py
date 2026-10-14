import os,sys,pathlib,json,torch,hashlib
from safetensors import safe_open
cfg=json.loads(json.dumps(__import__('yaml').safe_load(pathlib.Path(sys.argv[1]).read_text())));steps=cfg['train']['max_steps'];out=pathlib.Path(cfg['train']['output_dir']);cp=out/'checkpoints'/f'global_step_{steps}';assert (cp/'model/.metadata').is_file() and (cp/'optimizer/.metadata').is_file(),'DCP model/optimizer metadata missing'
n=int(os.environ.get('WORLD_SIZE','2'))
for rank in range(n):
 p=cp/'extra_state'/f'extra_state_rank_{rank}.pt';assert p.is_file();state=torch.load(p,map_location='cpu',weights_only=False) # Only this workflow's trusted, locally generated extra state, not external checkpoints.
 assert state['global_step']==steps and all(k in state for k in ['lr_scheduler','train_dataloader','torch_rng_state','environ_meter'])
hf=cp/'hf_ckpt';assert hf.is_dir();index=hf/'model.safetensors.index.json'
files=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else [p.name for p in hf.glob('*.safetensors')];assert files;keys=0;numel=0
for name in files:
 p=hf/name;assert p.is_file() and p.stat().st_size>0
 with safe_open(p,framework='pt',device='cpu') as f:
  for key in f.keys():
   keys+=1;shape=f.get_slice(key).get_shape();size=1
   for dim in shape:size*=dim
   numel+=size
assert numel>6_000_000_000,'Unexpected incomplete full VLA HF export'
report={'state':'CHECKPOINT_STRUCTURE_AND_EXTRA_STATE_VERIFIED','checkpoint':str(cp),'HF_checkpoint':str(hf),'steps':steps,'ranks':n,'HF_shards':files,'HF_parameter_keys':keys,'HF_numel':numel,'full_6B_CPU_load':False,'GPU_resume_execution':'SEPARATE_TEST_REQUIRED'};(out/'checkpoint_structure_verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
