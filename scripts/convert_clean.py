#!/usr/bin/env python3
"""Official RoboTwin conversion with a small LeRobot v0.4.2 API adapter.
No simulation imports, no remote datasets, and no overwrite of existing datasets.
"""
import os,pathlib,json,importlib.util,sys,difflib,hashlib,argparse,shutil,numpy as np
r=pathlib.Path(os.environ['LINGBOT_ROOT']);p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--sample',action='store_true');a=p.parse_args()
manifest=r/'configs/clean_episodes.manifest.json'
if not manifest.exists():manifest=r/f'configs/clean_episodes_{a.task}.manifest.json'
m=json.loads(manifest.read_text());eps=[e for e in m['episodes'] if e['task']==a.task];assert len(eps)==50 and {e['episode'] for e in eps}==set(range(50))
def sha256_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
for e in eps:
 assert sha256_file(r/e['path'])==e['sha256'], f'Raw episode changed: {e["path"]}'
 assert e['instruction_field']=='seen'
 pp=r/e['instructions_path'];assert set(json.loads(pp.read_text()))=={'seen'}
 assert hashlib.sha256(pp.read_bytes()).hexdigest()==e['instructions_sha256']
root_out=r/'data/clean_processed'/('smoke_lerobot' if a.sample else 'lerobot')
output=root_out/a.task
if output.exists():
 prov=output/'conversion_provenance.json'
 if prov.is_file() and json.loads(prov.read_text()).get('input_manifest_sha256')==hashlib.sha256(manifest.read_bytes()).hexdigest():print('ALREADY_CONVERTED',output);sys.exit(0)
 raise SystemExit(f'Refusing existing/incomplete output: {output}; inspect before manual recovery')
rawdir=r/'data/clean_raw/extracted'/a.task/'aloha-agilex_clean_50'
mid=r/'data/clean_processed/hdf5'/('sample' if a.sample else 'full')/a.task
if mid.exists():raise SystemExit(f'Intermediate directory exists: {mid}; inspect before recovery')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
source=r/'repos/RoboTwin/policy/pi0'
first=load('official_clean_hdf5',source/'scripts/process_data.py')
# Bound each task's working budget before invoking the unmodified offline converter.
selected=eps[:1] if a.sample else eps
raw_bytes=sum((r/e['path']).stat().st_size for e in selected)
required=raw_bytes*6+20*1024**3
if shutil.disk_usage(root_out.parent).free<required:raise SystemExit(f'SPACE_PENDING: conservative task working budget requires {required} free bytes')
first.data_transform(str(rawdir),1 if a.sample else 50,str(mid))
src=source/'examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py'
old=src.read_text();new=old.replace('from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME','from lerobot.utils.constants import HF_LEROBOT_HOME').replace('from lerobot.common.datasets.lerobot_dataset import LeRobotDataset','from lerobot.datasets.lerobot_dataset import LeRobotDataset')
new=new.replace('shutil.rmtree(HF_LEROBOT_HOME / repo_id)',"raise RuntimeError('Existing output must never be deleted')")
new=new.replace('repo_id=repo_id,\n        fps=50,','repo_id=repo_id,\n        root=Path(os.environ["LINGBOT_CONVERT_ROOT"]) / repo_id,\n        fps=50,')
new=new.replace('    dataset = create_empty_dataset(', '    hdf5_files.sort(key=lambda p: int(Path(p).parent.name.split("_")[-1]))\n    dataset = create_empty_dataset(')
new=new.replace('    # dataset.consolidate()','    dataset.finalize()')
patchdir=r/'docs/patches';patchdir.mkdir(exist_ok=True)
(patchdir/'robotwin-converter-v042.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(src),tofile='generated-v042-adapter.py')))
adapter=r/'scripts/generated_robotwin_v042_adapter.py';adapter.write_text(new)
os.environ['LINGBOT_CONVERT_ROOT']=str(root_out);os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_LEROBOT_HOME']=str(root_out)
second=load('official_clean_lerobot',adapter);second.HF_LEROBOT_HOME=root_out
np.random.seed(0)
second.port_aloha(raw_dir=mid,repo_id=a.task,mode='video',dataset_config=second.DatasetConfig(image_writer_processes=0,image_writer_threads=2),push_to_hub=False)
provenance={'source_revision':'13c3c47ff4312dd62484bcd51be034af55c062d1','lerobot':'v0.4.2','input_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'episodes':1 if a.sample else 50,'sample_only':a.sample,'instruction_field':'seen','seed':0,'adapter_sha256':hashlib.sha256(new.encode()).hexdigest()}
(output/'conversion_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print('CONVERTED',output)
