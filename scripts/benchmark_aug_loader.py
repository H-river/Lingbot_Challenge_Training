#!/usr/bin/env python3
"""CPU raw-clean image DataLoader benchmark, including HDF5/JPEG decoding.
Not a full model/LeRobot/tokenizer benchmark: reports that boundary explicitly.
"""
import argparse,json,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
from torch.utils.data import DataLoader,Dataset
from build_distractor_library import load_augment


class Images(Dataset):
    def __init__(self,args,profile):
        self.args,self.profile=args,profile;self.aug=load_augment(args.source)
        self.rows=json.loads(args.audit.read_text())['mask_records']
        self.entries={}
        for p in (args.root/'configs').glob('clean_episodes_*.manifest.json'):
            for e in json.loads(p.read_text())['episodes']:self.entries[e['task'],e['episode']]=e
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];e=self.entries[row['task'],row['episode']];path=self.args.root/e['path']
        current,future={},{}
        with h5py.File(path) as h:
            for cam,src in [('camera_top','head_camera'),('camera_wrist_left','left_camera'),('camera_wrist_right','right_camera')]:
                data=h[f'observation/{src}/rgb']
                for index,out in [(row['frame'],current),(min(row['frame']+49,len(data)-1),future)]:
                    im=cv2.imdecode(np.frombuffer(data[index],np.uint8),cv2.IMREAD_COLOR)
                    im=cv2.cvtColor(cv2.resize(im,(256,256)),cv2.COLOR_BGR2RGB)
                    out[cam]=torch.from_numpy(im.transpose(2,0,1).copy())
        if self.profile!='none':
            config={'profile':self.profile,'texture':{'source':'robotwin_seen','seen_dir':str(self.args.seen_dir)},'distractor':{'library_dir':str(self.args.library)}}
            current,_,_,_=self.aug.SampleAugment(config,42,0,i,str(path),e['task']).apply_pair(current,future)
        return torch.stack([t.float()/127.5-1 for t in current.values()])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','audit','library','seen-dir','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);cv2.setNumThreads(1);results={}
    for profile in ['none','v1','v2']:
        dataset=Images(a,profile);times=[]
        for repeat in range(3):
            start=time.perf_counter()
            for batch in DataLoader(dataset,batch_size=1,num_workers=0):pass
            times.append(time.perf_counter()-start)
        results[profile]={'seconds':times,'samples_per_second':len(dataset)/float(np.median(times))}
    with a.output.open('x') as f:json.dump({'samples':len(dataset),'workers':0,'scope':'raw clean HDF5/JPEG + current/future RGB augmentation + student normalization; excludes LeRobot H264/tokenizer/state/action/model','results':results},f,indent=2)
    print(json.dumps(results))
