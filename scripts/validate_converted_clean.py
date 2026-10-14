"""Validate converted state/action against pinned clean release; no simulation."""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import h5py
import pyarrow.parquet as pq
r=Path(os.environ['LINGBOT_ROOT']); task=sys.argv[1]
m=json.loads((r/'configs/clean_episodes.manifest.json').read_text())
eps=sorted((e for e in m['episodes'] if e['task']==task),key=lambda e:e['episode'])
assert len(eps)==50 and [e['episode'] for e in eps]==list(range(50))
d=r/'data/clean_processed/lerobot'/task
info=json.loads((d/'meta/info.json').read_text())
assert info['total_episodes']==50 and info['fps']==50
prov=json.loads((d/'conversion_provenance.json').read_text())
assert prov['episodes']==50 and not prov['sample_only']
assert prov['input_manifest_sha256']==hashlib.sha256((r/'configs/clean_episodes.manifest.json').read_bytes()).hexdigest()
import pyarrow as pa
table=pa.concat_tables([pq.read_table(p) for p in sorted((d/'data').rglob('*.parquet'))]).to_pandas()
task_table=pq.read_table(d/"meta/tasks.parquet").to_pandas()
text_by_id={int(row.task_index):str(text) for text,row in task_table.iterrows()}
frames=0
for e in eps:
 rows=table[table.episode_index==e['episode']].sort_values('frame_index')
 seen=json.loads((r/e['instructions_path']).read_text())['seen']
 assert rows.task_index.nunique()==1
 assert text_by_id[int(rows.task_index.iloc[0])] in seen
 with h5py.File(r/e['path']) as f:
  values=[f['joint_action/'+k][:] for k in ['left_arm','left_gripper','right_arm','right_gripper']]
  raw=np.concatenate([v.reshape(len(v),-1) for v in values],axis=1).astype(np.float32)
 assert len(rows)==len(raw)-1
 assert rows.frame_index.tolist()==list(range(len(rows)))
 for key,expected in [('observation.state',raw[:-1]),('action',raw[1:])]:
  actual=np.stack(rows[key]);assert actual.shape==expected.shape and np.isfinite(actual).all()
  np.testing.assert_array_equal(actual,expected)
 frames+=len(rows)
assert frames==info['total_frames']==len(table)
# Read both ends through the installed native loader, exercising all three videos.
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
ds=LeRobotDataset(task,root=d)
for idx in [0,len(ds)-1]:
 item=ds[idx]
 for cam in ['cam_high','cam_left_wrist','cam_right_wrist']:
  image=item['observation.images.'+cam]
  assert image.shape==(3,480,640) and torch.isfinite(image).all()
report={'task':task,'episodes':50,'frames':frames,'all_state_action_exact':True,'native_video_boundary_decode':True,'video_scope':'first and last dataset frames; not full video decode','manifest_sha256':prov['input_manifest_sha256']}
(r/'results/conversion_validation').mkdir(exist_ok=True)
(r/'results/conversion_validation'/f'{task}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
