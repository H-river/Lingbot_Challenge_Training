#!/usr/bin/env python3
"""Offline real-clean visual/mask audit and CPU image-pipeline throughput."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image,ImageDraw
from build_distractor_library import load_augment


def main(a):
    aug=load_augment(a.source);torch.set_num_threads(1);cv2.setNumThreads(1)
    a.output.mkdir(parents=True,exist_ok=False)
    records=[];samples=[];rng=np.random.default_rng(42)
    cameras={'camera_top':'head_camera','camera_wrist_left':'left_camera','camera_wrist_right':'right_camera'}
    config={'profile':'v2','teacher_uses_clean':True,'color':{'p':.5},'geometric':{'p':.3,'min_area':.9,'max_shift':.04},'texture':{'p':.3,'source':a.texture_source,'seen_dir':str(a.seen_dir),'regions':['table','wall']},'distractor':{'p':.3,'library_dir':str(a.library),'max_count':3,'cameras':['camera_top']}}
    for m in sorted((a.root/'configs').glob('clean_episodes_*.manifest.json')):
        entries=json.loads(m.read_text())['episodes'];task=entries[0]['task'];coverage=[]
        for ei in sorted(rng.choice(50,2,replace=False)):
            entry=next(e for e in entries if e['episode']==ei);raw=aug.safe_path(a.root/entry['path'])
            with h5py.File(raw) as h:
                frame=int(rng.integers(len(h['observation/head_camera/rgb'])))
                current={};future={}
                for cam,source in cameras.items():
                    for index,target in [(frame,current),(min(frame+49,len(h[f'observation/{source}/rgb'])-1),future)]:
                        im=cv2.imdecode(np.frombuffer(h[f'observation/{source}/rgb'][index],np.uint8),cv2.IMREAD_COLOR)
                        im=cv2.cvtColor(cv2.resize(im,(256,256)),cv2.COLOR_BGR2RGB)
                        target[cam]=torch.from_numpy(im.transpose(2,0,1).copy())
            colors=aug.episode_colors(str(raw));row={'task':task,'episode':int(ei),'frame':frame,'cameras':{}}
            for cam,im in current.items():
                row['cameras'][cam]={name:float(aug.region_mask(aug.uint8_hwc(im),color).mean()) for name,color in colors[cam].items()}
                row['cameras'][cam]['ambiguous_surface_colors']=bool(np.linalg.norm(colors[cam]['table']-colors[cam]['wall'])<24)
            canvas=Image.new('RGB',(6*256,3*280),'white');draw=ImageDraw.Draw(canvas)
            for pi,profile in enumerate(['none','v1','v2']):
                c={**config,'profile':profile}
                if profile=='none':student,teacher=current,current
                else:student,teacher,_,_=aug.SampleAugment(c,42,0,len(samples),str(raw),task).apply_pair(current,future)
                for ci,cam in enumerate(cameras):
                    for side,images in enumerate([student,teacher]):
                        x=(ci*2+side)*256;y=pi*280
                        draw.text((x+4,y+3),f'{profile} {cam} {"student" if side==0 else "teacher"}',fill='black')
                        canvas.paste(Image.fromarray(aug.uint8_hwc(images[cam])),(x,y+24))
            canvas.save(a.output/f'{task}_{ei:02d}.png')
            records.append(row);samples.append((current,future,str(raw),task))
        print(task,flush=True)
    timing={}
    for profile in ['none','v1','v2']:
        durations=[]
        for repeat in range(3):
            start=time.perf_counter()
            for i,(current,future,raw,task) in enumerate(samples):
                if profile=='none':student,teacher=current,current
                else:student,teacher,_,_=aug.SampleAugment({**config,'profile':profile},42,repeat,i,raw,task).apply_pair(current,future)
                # Match image normalization cost common to all profiles.
                torch.stack([im.float()/127.5-1 for im in student.values()])
            durations.append(time.perf_counter()-start)
        timing[profile]={'samples_per_second':len(samples)/float(np.median(durations)),'seconds_per_100':durations}
    result={'samples':len(samples),'seed':42,'texture_source':a.texture_source,'mask_records':records,'throughput':timing,'throughput_scope':'Decoded real clean current/future RGB image pipeline + student normalization, excludes decoder/tokenizer/state/action collation','ambiguous_surface_tasks':sorted({r['task'] for r in records if any(c['ambiguous_surface_colors'] for c in r['cameras'].values())}),'low_table_mask_tasks':sorted({r['task'] for r in records if r['cameras']['camera_top']['table']<.2})}
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(timing),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','output','library','seen-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--texture-source',choices=['procedural','robotwin_seen'],default='robotwin_seen')
    main(p.parse_args())
