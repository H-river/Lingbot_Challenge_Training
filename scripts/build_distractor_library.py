#!/usr/bin/env python3
"""Deterministic, conservative cutouts from SHA-verified release clean first frames."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image


def load_augment(source):
    spec=importlib.util.spec_from_file_location('aug',source/'lingbotvla/data/vla_data/augment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def candidates(image, aug):
    colors=aug.estimate_colors(image)
    table=aug.region_mask(image,colors['table'],background_only=False).astype(np.uint8)
    # Only enclosed holes within the table: border-connected arms are excluded.
    flood=table.copy();cv2.floodFill(flood,np.zeros((table.shape[0]+2,table.shape[1]+2),np.uint8),(0,0),1)
    holes=1-flood
    n,labels,stats,_=cv2.connectedComponentsWithStats(holes)
    h,w=table.shape
    for label in range(1,n):
        x,y,bw,bh,area=stats[label]
        if not 30<=area<=h*w*.04 or x<5 or y<5 or x+bw>w-5 or y+bh>h-5:
            continue
        # Conservative arm exclusion: lower quarter, elongated components, black parts.
        if y+bh>h*.75 or max(bw,bh)>min(bw,bh)*4:
            continue
        support=labels[y:y+bh,x:x+bw]==label
        rgb=image[y:y+bh,x:x+bw]
        if np.mean(rgb[support].mean(-1)<45)>.15:
            continue
        chroma=rgb.max(-1).astype(np.int16)-rgb.min(-1).astype(np.int16)
        if np.mean(chroma[support]>30)<.20:
            continue  # conservative: reject white robot/shadow scraps, including achromatic objects
        support &= np.linalg.norm(rgb.astype(np.float32)-colors['table'],axis=-1)>40
        if support.sum()<30:
            continue
        alpha=(support.astype(np.uint8)*255)
        yield [int(x),int(y),int(bw),int(bh)],np.dstack([rgb,alpha])


def build(root,out,source):
    aug=load_augment(source)
    out.mkdir(parents=True,exist_ok=False)
    items=[]
    for manifest in sorted((root/'configs').glob('clean_episodes_*.manifest.json')):
        for entry in sorted(json.loads(manifest.read_text())['episodes'],key=lambda e:e['episode']):
            path=aug.safe_path(root/entry['path'])
            digest=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
            if digest.hexdigest()!=entry['sha256']:raise ValueError(f'Raw SHA mismatch: {path}')
            with h5py.File(path) as h:
                image=cv2.imdecode(np.frombuffer(h['observation/head_camera/rgb'][0],np.uint8),cv2.IMREAD_COLOR)
            image=cv2.cvtColor(cv2.resize(image,(320,240)),cv2.COLOR_BGR2RGB)
            for i,(bbox,patch) in enumerate(candidates(image,aug)):
                name=f'{entry["task"]}_{entry["episode"]:02d}_{i:02d}.png'
                target=out/name;Image.fromarray(patch).save(target,compress_level=9)
                items.append({'file':name,'task':entry['task'],'episode':entry['episode'],'bbox':bbox,'frame_size':[320,240], 'source_sha256':entry['sha256'],'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    result={'source':'release_clean_first_head_frames','items':items}
    (out/'manifest.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'cutouts':len(items),'directory':str(out)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True,help='Patched lingbot-vla-v2 source')
    a=p.parse_args();build(a.root,a.output,a.source)
