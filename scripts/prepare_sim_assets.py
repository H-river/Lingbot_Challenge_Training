#!/usr/bin/env python3
# Asset file preparation only: imports no simulator or planner.
import os,pathlib,json,zipfile,hashlib,shutil,stat,argparse
r=pathlib.Path(os.environ['LINGBOT_ROOT']);p=argparse.ArgumentParser();p.add_argument('--archive',choices=['embodiments.zip','objects.zip','background_texture.zip']);a=p.parse_args()
m=json.loads((r/'configs/assets.manifest.json').read_text())['simulation_assets'];dest=r/'assets/simulation';dest.mkdir(exist_ok=True)
for entry in m['files']:
 name=entry['rfilename']
 if a.archive and name!=a.archive:continue
 zpath=r/'assets/archives'/name;done=dest/(name+'.prepared.json')
 if done.exists():
  state=json.loads(done.read_text());assert state['sha256']==entry['lfs']['sha256']
  assert all((dest/top).exists() for top in state['top_level'])
  print('ALREADY_PREPARED',name);continue
 if not zpath.exists():raise SystemExit(f'ASSET_PENDING {zpath}')
 h=hashlib.sha256()
 with zpath.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==entry['lfs']['sha256']
 stage=dest/(name+'.extracting')
 if stage.exists():raise SystemExit(f'Interrupted extraction requires inspection: {stage}')
 with zipfile.ZipFile(zpath) as z:
  members=[i for i in z.infolist() if not i.filename.startswith('__MACOSX/') and pathlib.PurePosixPath(i.filename).name!='.DS_Store']
  need=sum(i.file_size for i in members)
  assert shutil.disk_usage(dest).free>need+20*1024**3,'SPACE_PENDING: extraction reserve'
  stage.mkdir()
  for i in members:
   rp=pathlib.PurePosixPath(i.filename);assert not rp.is_absolute() and '..' not in rp.parts
   assert not stat.S_ISLNK(i.external_attr>>16),'Archive symlink requires manual review'
   target=stage.joinpath(*rp.parts)
   if i.is_dir():target.mkdir(parents=True,exist_ok=True);continue
   target.parent.mkdir(parents=True,exist_ok=True)
   with z.open(i) as f,target.open('wb') as out:shutil.copyfileobj(f,out,4*1024**2)
 tops=[c.name for c in stage.iterdir()]
 for child in stage.iterdir():
  assert not (dest/child.name).exists(),f'Asset destination exists: {child.name}'
  child.rename(dest/child.name)
 stage.rmdir()
 for top in tops:
  link=r/'repos/RoboTwin/assets'/top
  if link.is_symlink():assert link.resolve()==(dest/top).resolve()
  elif link.exists():raise SystemExit(f'Refusing to replace repository asset: {link}')
  else:link.symlink_to(dest/top,target_is_directory=(dest/top).is_dir())
 done.write_text(json.dumps({'source':m['repo'],'revision':m['revision'],'sha256':h.hexdigest(),'top_level':tops,'uncompressed_bytes':need},indent=2)+'\n')
 print('PREPARED',name,need,flush=True)
# Exact path substitution from upstream update_embodiment_config_path.py.
# Fail on errors rather than swallowing them as that script does.
changes=[]
for tmp in (dest/'embodiments').rglob('*_tmp.yml'):
 original=tmp.read_text();new=original.replace('${ASSETS_PATH}',str(r/'repos/RoboTwin')).replace('$ASSETS_PATH',str(r/'repos/RoboTwin'))
 target=tmp.with_name(tmp.name.replace('_tmp.yml','.yml'))
 if target.exists() and target.read_text()!=new:
  backup=target.with_name(target.name+'.upstream')
  if backup.exists():raise SystemExit(f'Existing modified asset config requires inspection: {target}')
  backup.write_bytes(target.read_bytes())
 target.write_text(new);changes.append({'path':str(target),'sha256':hashlib.sha256(new.encode()).hexdigest()})
(r/'docs/asset_path_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
