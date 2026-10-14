#!/usr/bin/env python3
"""Verify exact release archives; extract HDF5 and seen instructions only."""
import os,pathlib,json,hashlib,zipfile,re,shutil,argparse
root=pathlib.Path(os.environ['LINGBOT_ROOT']);p=argparse.ArgumentParser();p.add_argument('--task');a=p.parse_args()
m=json.loads((root/'configs/clean_archives.manifest.json').read_text());records=[];tasks=[]
for entry in m['archives']:
 if a.task and a.task!=entry['task']:continue
 path=root/'data/clean_raw/archives'/entry['path']
 if not path.exists():
  if a.task:raise SystemExit(f'MISSING {path}')
  continue
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 assert h.hexdigest()==entry['sha256'],f'Archive SHA mismatch {path}'
 with zipfile.ZipFile(path) as z:
  names=z.namelist();assert len(names)==len(set(names)),'Duplicate archive members'
  for name in names:
   rp=pathlib.PurePosixPath(name);assert not rp.is_absolute() and '..' not in rp.parts
  data_names={n for n in names if n.endswith('.hdf5')}
  expect={f'aloha-agilex_clean_50/data/episode{i}.hdf5' for i in range(50)}
  assert data_names==expect,f'Episode roster mismatch {entry["task"]}'
  required=list(expect)+[f'aloha-agilex_clean_50/instructions/episode{i}.json' for i in range(50)]
  needed=sum(z.getinfo(n).file_size for n in required)
  out=root/'data/clean_raw/extracted'/entry['task']
  out.mkdir(parents=True,exist_ok=True)
  assert shutil.disk_usage(out).free>needed+20*1024**3,'SPACE_PENDING'
  for i in range(50):
   dn=f'aloha-agilex_clean_50/data/episode{i}.hdf5';jn=f'aloha-agilex_clean_50/instructions/episode{i}.json'
   dest=out/dn;dest.parent.mkdir(parents=True,exist_ok=True)
   # CRC is verified by ZipFile while streaming. Do not load pickle trajectories.
   temp=dest.with_suffix('.hdf5.tmp');dh=hashlib.sha256()
   with z.open(dn) as inp,temp.open('wb') as f:
    for block in iter(lambda:inp.read(4*1024**2),b''):f.write(block);dh.update(block)
   temp.replace(dest)
   instr_raw=z.read(jn);instr=json.loads(instr_raw)
   assert isinstance(instr.get('seen'),list) and instr['seen'] and all(isinstance(s,str) and s.strip() for s in instr['seen'])
   ip=out/jn;ip.parent.mkdir(parents=True,exist_ok=True);ip.write_text(json.dumps({'seen':instr['seen']},ensure_ascii=False)+'\n')
   records.append({'task':entry['task'],'episode':i,'path':str(dest.relative_to(root)),'sha256':dh.hexdigest(),'instructions_path':str(ip.relative_to(root)),'instructions_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),'release_instructions_sha256':hashlib.sha256(instr_raw).hexdigest(),'instruction_field':'seen','archive_sha256':entry['sha256']})
  tasks.append(entry['task']);print('VERIFIED_AND_EXTRACTED',entry['task'],50,flush=True)
if not records:raise SystemExit('No verified downloaded clean archives')
# Each invocation inventories only its selected scope, never claims full completion.
name='clean_episodes'+('_'+a.task if a.task else '')+'.manifest.json'
result={'source':m['repository'],'revision':m['revision'],'tasks':tasks,'complete_2500':len(tasks)==50 and len(records)==2500,'episodes':records}
(root/'configs'/name).write_text(json.dumps(result,indent=2)+'\n')
print(name,len(records),'complete_2500',result['complete_2500'])
