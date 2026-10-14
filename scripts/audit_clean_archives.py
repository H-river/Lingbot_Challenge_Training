#!/usr/bin/env python3
"""Inventory exact release episodes without expanding the dataset or reading pickle."""
import os,pathlib,json,hashlib,zipfile
r=pathlib.Path(os.environ['LINGBOT_ROOT']);m=json.loads((r/'configs/clean_archives.manifest.json').read_text());episodes=[];missing=[];expanded=0
for e in m['archives']:
 p=r/'data/clean_raw/archives'/e['path']
 if not p.is_file():missing.append(e['task']);continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==e['sha256'],f'Archive SHA mismatch: {p}'
 with zipfile.ZipFile(p) as z:
  names=z.namelist();assert len(names)==len(set(names))
  for name in names:
   q=pathlib.PurePosixPath(name);assert not q.is_absolute() and '..' not in q.parts
  assert {n for n in names if n.endswith('.hdf5')}=={f'aloha-agilex_clean_50/data/episode{i}.hdf5' for i in range(50)}
  for i in range(50):
   member=f'aloha-agilex_clean_50/data/episode{i}.hdf5';info=z.getinfo(member)
   ip=f'aloha-agilex_clean_50/instructions/episode{i}.json';raw=z.read(ip);instructions=json.loads(raw)
   assert isinstance(instructions.get('seen'),list) and instructions['seen'] and all(isinstance(s,str) and s.strip() for s in instructions['seen'])
   expanded+=info.file_size+len(raw)
   episodes.append({'task':e['task'],'episode':i,'archive_path':str(p.relative_to(r)),'archive_sha256':e['sha256'],'member':member,'uncompressed_bytes':info.file_size,'zip_crc32':info.CRC,'instructions_member':ip,'release_instructions_sha256':hashlib.sha256(raw).hexdigest(),'allowed_instruction_field':'seen'})
 print('RELEASE_INVENTORY_VERIFIED',e['task'],flush=True)
result={'source':m['repository'],'revision':m['revision'],'complete_2500':not missing and len(episodes)==2500,'expanded_selected_bytes':expanded,'missing_tasks':missing,'episodes':episodes,'note':'Archive hashes and exact release member rosters verified; not a claim that all episodes are extracted or converted.'}
(r/'configs/clean_release_episodes.manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print('RELEASE_MANIFEST',len(episodes),'missing_tasks',missing,'expanded_selected_bytes',expanded)
raise SystemExit(3 if missing else 0)
