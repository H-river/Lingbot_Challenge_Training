import os,pathlib,json,hashlib,math,datetime,collections
r=pathlib.Path(os.environ['LINGBOT_ROOT']);mfile=r/'configs/clean_episodes.manifest.json';m=json.loads(mfile.read_text());sha=hashlib.sha256(mfile.read_bytes()).hexdigest();archive=json.loads((r/'configs/clean_archives.manifest.json').read_text());allowed={a['task']:a for a in archive['archives']};assert len(allowed)==50 and m['complete_2500'] and len(m['episodes'])==2500 and set(m['tasks'])==set(allowed)
assert m['source']==archive['repository']=='TianxingChen/RoboTwin2.0' and m['revision']==archive['revision']=='3dc3b798668feb99ac61cc9086d84cbcc3d79186'
lines=(r/'configs/train_clean.txt').read_text().splitlines();assert len(lines)==50;seen=set();total_frames=0
for line in lines:
 robot,path=line.split();d=pathlib.Path(path);task=d.name;assert robot=='robotwin' and task in allowed and task not in seen;seen.add(task)
 assert d==r/'data/clean_processed/lerobot'/task and d.resolve().is_relative_to((r/'data/clean_processed/lerobot').resolve())
 eps=sorted((e for e in m['episodes'] if e['task']==task),key=lambda a:a['episode']);assert len(eps)==50 and [e['episode'] for e in eps]==list(range(50))
 for e in eps:
  assert e['archive_sha256']==allowed[task]['sha256'] and e['instruction_field']=='seen'
  raw=r/e['path'];instr=r/e['instructions_path'];assert raw.is_file() and instr.is_file();assert raw.resolve().is_relative_to((r/'data/clean_raw/extracted').resolve());assert set(json.loads(instr.read_text()))=={'seen'}
 info=json.loads((d/'meta/info.json').read_text());prov=json.loads((d/'conversion_provenance.json').read_text());valid=json.loads((r/'results/conversion_validation'/f'{task}.json').read_text())
 assert info['total_episodes']==prov['episodes']==valid['episodes']==50 and info['fps']==50 and not prov['sample_only'];assert prov['input_manifest_sha256']==valid['manifest_sha256']==sha;assert valid['all_state_action_exact'] and valid['native_video_boundary_decode'];assert info['total_frames']==valid['frames'];total_frames+=info['total_frames']
 # v3 output paths can span multiple parquet/video files; native validation has decoded each camera's boundary frames.
 assert any((d/'data').rglob('*.parquet')) and any((d/'videos').rglob('*.mp4'))
norm=r/'configs/clean_norm/robotwin.json';stats=json.loads(norm.read_text());p=json.loads((norm.parent/'input_provenance.json').read_text());assert p['episode_manifest_sha256']==sha and p['train_list_sha256']==hashlib.sha256((r/'configs/train_clean.txt').read_bytes()).hexdigest();assert p['source']==m['source'] and p['revision']==m['revision']
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 if isinstance(x,(int,float)):return math.isfinite(x)
 return True
assert finite(stats)
report={'state':'DATA_TRAIN_READY_50','tasks':50,'episodes':2500,'frames':total_frames,'manifest_sha256':sha,'train_list_sha256':hashlib.sha256((r/'configs/train_clean.txt').read_bytes()).hexdigest(),'norm_path':str(norm),'norm_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),'native_video_scope':'per task first/last dataset frames via validator, not all frames','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_training_or_simulation_tested':False}
(r/'results/data_train_ready_50.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
