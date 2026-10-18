"""CPU augmentation contracts. Run against an applied patch, not pristine vendor."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

SOURCE=Path(os.environ['AUG_SOURCE'])
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
aug=module('aug_under_test',SOURCE/'lingbotvla/data/vla_data/augment.py')


def config(profile='v1'):
    return {'profile':profile,'teacher_uses_clean':True,'color':{'p':1},'geometric':{'p':1},'texture':{'p':0},'distractor':{'p':0}}


def fixture():
    return torch.arange(3*32*40).reshape(3,32,40).remainder(256).to(torch.uint8)


class Samples(Dataset):
    def __len__(self):return 12
    def __getitem__(self,index):
        obj=aug.SampleAugment(config(),42,3,index,'unused','task')
        return obj.apply_pair({'camera_top':fixture()})[0]['camera_top']


class Contracts(unittest.TestCase):
    def test_workers(self):
        reference=None
        for workers in (0,2,4):
            value=torch.cat(list(DataLoader(Samples(),batch_size=2,num_workers=workers)),0)
            if reference is None:reference=value
            else:self.assertTrue(torch.equal(reference,value))

    def test_teacher_temporal(self):
        obj=aug.SampleAugment(config(),42,3,1,'unused','task');im=fixture()
        s,t,fs,ft=obj.apply_pair({'camera_top':im},{'camera_top':im})
        self.assertTrue(torch.equal(s['camera_top'],fs['camera_top']))
        self.assertTrue(torch.equal(t['camera_top'],ft['camera_top']))
        expected=aug.tensor_image(aug.geometry(aug.uint8_hwc(im),obj.geom),im)
        self.assertTrue(torch.equal(t['camera_top'],expected))
        self.assertFalse(torch.equal(s['camera_top'],t['camera_top']))

    def test_v2_temporal_and_distractors(self):
        import hashlib,json
        from PIL import Image
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cut=np.zeros((8,8,4),np.uint8);cut[:,:,:3]=[150,30,20];cut[:,:,3]=255
            Image.fromarray(cut).save(root/'cut.png')
            (root/'manifest.json').write_text(json.dumps({'source':'release_clean_first_head_frames','items':[{'file':'cut.png','task':'other','sha256':hashlib.sha256((root/'cut.png').read_bytes()).hexdigest()}]}))
            c=config('v2');c['texture']={'p':1,'source':'procedural'};c['distractor']={'p':1,'library_dir':str(root),'max_count':1}
            im=torch.full((3,64,64),200,dtype=torch.uint8)
            colors={'camera_top':{'table':np.array([200,200,200]),'wall':np.array([80,80,80])}}
            with patch.object(aug,'episode_colors',return_value=colors):
                obj=aug.SampleAugment(c,42,3,1,'unused','task')
                student,teacher,fs,ft=obj.apply_pair({'camera_top':im},{'camera_top':im})
            self.assertTrue(torch.equal(student['camera_top'],fs['camera_top']))
            self.assertTrue(torch.equal(teacher['camera_top'],ft['camera_top']))

    def test_global_rng_independent(self):
        a=Samples()[5];torch.manual_seed(999);np.random.seed(999)
        self.assertTrue(torch.equal(a,Samples()[5]))

    def test_profile_gate(self):
        for profile in ('none','v1','v2'):
            self.assertFalse(aug.enabled(config(profile),policy_eval=True))
        self.assertFalse(aug.enabled({}))

    def test_unseen(self):
        with self.assertRaises(ValueError):aug.safe_path('/tmp/assets/unseen/image.png')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'unseen').mkdir();(root/'seen').symlink_to(root/'unseen')
            with self.assertRaises(ValueError):aug.safe_path(root/'seen')

    def test_procedural_does_not_open_seen(self):
        from unittest.mock import patch
        with patch.object(aug,'texture_files',side_effect=AssertionError('read seen')):
            value=aug.make_texture({'source':'procedural','seen_dir':'/unseen'},aug.sample_rng(1,2,3))
            self.assertEqual(value.shape,(256,256,3))

    def test_state_labels_unchanged(self):
        c=config();c['state_noise']={'enabled':True,'std':.1}
        obj=aug.SampleAugment(c,1,1,1,'unused','task')
        state=torch.ones(14);absolute=torch.ones(50,14)*2
        labels=absolute-state;before=labels.clone()
        noisy=obj.perturb_state(state)
        self.assertTrue(torch.equal(labels,before));self.assertFalse(torch.equal(noisy,state))

    def test_none_original_prepare_images(self):
        original=os.environ.get('AUG_ORIGINAL_SOURCE')
        if not original:self.fail('AUG_ORIGINAL_SOURCE required for bitwise regression')
        old=module('old_transform',Path(original)/'lingbotvla/data/vla_data/transform.py')
        new=module('new_transform',SOURCE/'lingbotvla/data/vla_data/transform.py')
        observation={'state':torch.ones(14),'image':{k:fixture() for k in ['camera_top','camera_wrist_left','camera_wrist_right']}}
        before=old.prepare_images(None,copy.deepcopy(observation),train=False,use_depth_align=True,return_augment_params=True)
        after=new.prepare_images(None,copy.deepcopy(observation),train=False,use_depth_align=True,return_augment_params=True)
        def check(a,b):
            if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
            elif isinstance(a,np.ndarray):self.assertTrue(np.array_equal(a,b))
            elif isinstance(a,(tuple,list)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):check(x,y)
            else:self.assertEqual(a,b)
        check(before,after)

    def test_feature_transform_inference_and_labels(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from lingbotvla.data.vla_data import utils
        def feature():
            f=utils.FeatureTransform.__new__(utils.FeatureTransform)
            f.org_features={'actions':['action.arm']};f.actions_convert_from_state={}
            f.actions=['action.arm'];f.action_subtract_state={'action.arm':True};f.action_relative_type={}
            f.chunk_size=2;f.normalizer=None;f.return_item_befor_padding=False
            f.model_config=SimpleNamespace(max_state_dim=2,max_action_dim=2)
            f.disabled_image_features=False;f.image_augment=False;f.use_depth_align=True;f.use_future_image=False
            f.feature_config=SimpleNamespace(images=['camera_top']);f.tokenizer=None
            f.processor=SimpleNamespace(image_processor=lambda x:{'pixel_values':x.float()/127.5-1})
            f.convert_features=lambda item,w_action:copy.deepcopy(item)
            f.pad_and_concat=lambda item,w_action:{'state':item['observation.state.arm'],'action':item['action.arm'],'image':{'camera_top':fixture()},'prompt':item['task'],'action_is_pad':torch.zeros(2),'state_joint_mask':torch.ones(2),'action_joint_mask':torch.ones(2),'chunk_joint_mask':torch.ones(2,2)}
            return f
        item={'action.arm':torch.full((2,2),2.),'observation.state.arm':torch.ones(2),'action.arm_is_pad':torch.zeros(2),'task':'original'}
        with patch.object(utils,'compute_image_token_count',return_value=10), patch.object(utils,'prepare_language',side_effect=lambda model,tokenizer,b:(b['prompt'],None)):
            baseline=feature().apply(copy.deepcopy(item),policy_eval=True)
            for profile in ('none','v1','v2'):
                class Bomb:
                    @property
                    def config(self):raise AssertionError('Inference touched augmentation')
                actual=feature().apply(copy.deepcopy(item),policy_eval=True,aug_context=Bomb())
                for key in ('images','pil_images','state'):
                    self.assertTrue(torch.equal(baseline[key],actual[key]))
                self.assertEqual(actual['lang_tokens'],'original')
            clean=feature().apply(copy.deepcopy(item))
            old_utils=module('lingbotvla.data.vla_data._original_utils',Path(os.environ['AUG_ORIGINAL_SOURCE'])/'lingbotvla/data/vla_data/utils.py')
            with patch.object(old_utils,'compute_image_token_count',return_value=10), patch.object(old_utils,'prepare_language',side_effect=lambda model,tokenizer,b:(b['prompt'],None)):
                original=old_utils.FeatureTransform.apply(feature(),copy.deepcopy(item))
            self.assertEqual(set(clean),set(original))
            for key in clean:
                if isinstance(clean[key],torch.Tensor):self.assertTrue(torch.equal(clean[key],original[key]),key)
                else:self.assertEqual(clean[key],original[key])
            c=config();c['state_noise']={'enabled':True,'std':.2}
            context=aug.SampleAugment(c,42,1,2,'unused','task');context.seen=['swapped']
            perturbed=feature().apply(copy.deepcopy(item),aug_context=context)
            self.assertTrue(torch.equal(clean['actions'],perturbed['actions']))
            self.assertFalse(torch.equal(clean['state'],perturbed['state']))
            self.assertEqual(perturbed['lang_tokens'],'swapped')

    def test_enclosed_reflection_is_not_background(self):
        import cv2
        image=np.full((128,128,3),220,np.uint8)
        cv2.circle(image,(64,64),32,(30,30,30),-1)
        cv2.circle(image,(64,64),26,(220,220,220),-1)
        mask=aug.region_mask(image,np.array([220,220,220]))
        self.assertFalse(bool(mask[64,64]));self.assertTrue(bool(mask[5,5]))

    def test_brightness_range(self):
        values=[]
        for i in range(400):
            a=aug.SampleAugment({'profile':'v1','color':{'p':1}},42,0,i,'unused','task')
            values.append(a.color[0])
        self.assertGreaterEqual(min(values),.75);self.assertLessEqual(max(values),1.1)
        self.assertGreater(max(values),1.05);self.assertLess(min(values),.8)
        a=aug.SampleAugment({'profile':'v1','color':{'p':1,'brightness':[.9,.9]}},42,0,0,'unused','task')
        self.assertAlmostEqual(a.color[0],.9)

    @staticmethod
    def shadow_scene():
        """White-ish table, a soft shadow touching lit table, and a grey disk with a hard dark rim."""
        import cv2
        table=np.array([240,238,236],np.float32)
        image=np.tile(table,(160,200,1))
        shade=np.zeros((160,200),np.float32);cv2.rectangle(shade,(10,90),(90,150),1,-1)
        shade=cv2.GaussianBlur(shade,(0,0),4)                       # soft shadow edge
        image*=(1-.25*shade)[...,None]                              # 25% darker, same chroma
        cv2.circle(image,(150,60),26,(25,25,25),-1)                  # hard dark rim
        cv2.circle(image,(150,60),20,tuple(float(x) for x in table*.7),-1)  # grey object, shadow-like colour
        return np.clip(image,0,255).astype(np.uint8),table

    def test_shadow_aware_grows_into_shadow_not_objects(self):
        image,table=self.shadow_scene()
        strict=aug.region_mask(image,table);shadow=aug.shadow_region_mask(image,table)
        core=(slice(105,140),slice(20,80))
        self.assertLess(strict[core].mean(),.1)                     # conservative mask skips shadow
        self.assertGreater(shadow[core].mean(),.9)                  # shadow-aware mask covers it
        self.assertFalse(shadow[50:70,140:160].any())               # grey object enclosed by hard edge is protected
        self.assertTrue(np.all(shadow[strict]))                     # never removes conservative pixels

    def test_shadow_aware_only_on_configured_cameras(self):
        image,table=self.shadow_scene()
        colors={'table':table}
        a=aug.SampleAugment({'profile':'v2','texture':{'p':0}},42,0,0,'unused','task')
        head=a.surface_masks('camera_top',image,colors)['table']
        wrist=a.surface_masks('camera_wrist_left',image,colors)['table']
        self.assertTrue(np.array_equal(wrist,aug.region_mask(image,table)))
        self.assertTrue(np.array_equal(head,aug.shadow_region_mask(image,table)))
        b=aug.SampleAugment({'profile':'v2','texture':{'p':0,'shadow_aware_cameras':[]}},42,0,0,'unused','task')
        self.assertTrue(np.array_equal(b.surface_masks('camera_top',image,colors)['table'],aug.region_mask(image,table)))

    def test_seam_filled_but_arm_kept(self):
        wall_c,table_c=np.array([200,200,225.]),np.array([245,245,245.])
        image=np.zeros((120,160,3),np.uint8);image[:50]=wall_c;image[50:]=table_c
        image[46:62]=255                                            # bright edge band left unmasked
        image[40:70,70:90]=20                                       # dark arm crossing the boundary
        wall=np.zeros((120,160),bool);wall[:44]=True
        table=np.zeros((120,160),bool);table[64:]=True
        t,w=aug.fill_seam(table,wall,image,{'table':table_c,'wall':wall_c})
        band=np.zeros((120,160),bool);band[46:62]=True;band[:,68:92]=False
        self.assertTrue(np.all((t|w)[band]))                        # seam closed
        self.assertFalse(np.any((t|w)[44:64,72:88]))                # arm untouched
        self.assertFalse(np.any(t&w))

    @staticmethod
    def texture_dirs(root):
        from PIL import Image
        dirs={}
        for name,colour in (('seen',(200,30,30)),('unseen',(30,30,200))):
            d=root/'background_texture'/name;d.mkdir(parents=True)
            for i in range(2):Image.fromarray(np.full((16,16,3),colour,np.uint8)).save(d/f'{i}.png')
            dirs[name]=str(d)
        return dirs

    def test_oracle_requires_opt_in_and_keeps_other_guards(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            d=self.texture_dirs(Path(tmp))
            c=config('v2');c['texture']={'p':1,'source':aug.ORACLE_SOURCE,'seen_dir':d['seen'],'unseen_dir':d['unseen']}
            aug.oracle_unseen_files.cache_clear();aug.read_oracle_texture.cache_clear()
            with patch.dict(os.environ,{},clear=False):
                os.environ.pop(aug.ORACLE_ENV,None)
                with self.assertRaises(PermissionError):aug.SampleAugment(c,42,0,0,'unused','task')
                with self.assertRaises(PermissionError):aug.read_oracle_texture(d['unseen']+'/0.png')
            with patch.dict(os.environ,{aug.ORACLE_ENV:'yes'}):
                aug.oracle_unseen_files.cache_clear()
                with self.assertRaises(PermissionError):aug.SampleAugment(c,42,0,0,'unused','task')
            with patch.dict(os.environ,{aug.ORACLE_ENV:aug.ORACLE_TOKEN}):
                with self.assertRaises(ValueError):aug.texture_files(d['unseen'])          # seen loader still rejects unseen
                with self.assertRaises(ValueError):aug.safe_path(d['unseen'])              # global guard unchanged
                bad=dict(c);bad['profile']='v1'
                with self.assertRaises(ValueError):aug.SampleAugment(bad,42,0,0,'unused','task')
            aug.oracle_unseen_files.cache_clear()

    def test_oracle_pool_and_v2_alignment(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            d=self.texture_dirs(Path(tmp))
            v2=config('v2');v2['texture']={'p':.3,'source':'robotwin_seen','seen_dir':d['seen']};v2['distractor']={'p':0}
            v3=copy.deepcopy(v2);v3['texture'].update(source=aug.ORACLE_SOURCE,unseen_dir=d['unseen'])
            with patch.dict(os.environ,{aug.ORACLE_ENV:aug.ORACLE_TOKEN}):
                aug.oracle_unseen_files.cache_clear()
                pools=set()
                for i in range(200):
                    a,b=aug.SampleAugment(v2,42,0,i,'unused','task'),aug.SampleAugment(v3,42,0,i,'unused','task')
                    self.assertEqual((a.geom,a.color,a.distractor_on,bool(a.textures)),(b.geom,b.color,b.distractor_on,bool(b.textures)))
                    for t in b.textures.values():pools.add('unseen' if t[...,2].mean()>t[...,0].mean() else 'seen')
                self.assertEqual(pools,{'seen','unseen'})
            aug.oracle_unseen_files.cache_clear()

    # ----------------------------- V2b -----------------------------
    @staticmethod
    def v2b_fixture(root):
        import hashlib,json,cv2
        from PIL import Image
        cut=np.zeros((10,10,4),np.uint8);cut[...,:3]=[150,30,20];cut[...,3]=255
        Image.fromarray(cut).save(root/'c.png')
        (root/'manifest.json').write_text(json.dumps({'source':'release_clean_first_head_frames','items':[{'file':'c.png','task':'other','sha256':hashlib.sha256((root/'c.png').read_bytes()).hexdigest()}]}))
        img=np.full((120,160,3),235,np.uint8);img[:25]=(200,200,225);cv2.circle(img,(110,85),10,(30,160,40),-1)
        cams=['camera_top','camera_wrist_left','camera_wrist_right']
        colors={c:{'table':np.array([235.,235,235]),'wall':np.array([200.,200,225])} for c in cams}
        cfg={'profile':'v2','teacher_uses_clean':True,'color':{'p':0},'geometric':{'p':0},
             'texture':{'p':1,'mode':'overlay','source':'procedural','regions':['table','wall']},
             'distractor':{'p':1,'placement':'v2b','library_dir':str(root),'max_count':3,'cameras':cams,'near_object_p':.5,'idle_focus_p':0}}
        return img,cams,colors,cfg

    def test_v2b_default_configs_keep_old_path(self):
        self.assertFalse(aug.SampleAugment(config('v2'),42,0,0,'unused','task').v2b_enabled())
        c=config('v2');c['texture']={'p':0,'mode':'overlay'}
        self.assertTrue(aug.SampleAugment(c,42,0,0,'unused','task').v2b_enabled())

    def test_v2b_overlay_teacher_clean_temporal_and_whole_image(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            img,cams,colors,cfg=self.v2b_fixture(Path(tmp))
            cfg['distractor']['p']=0
            t={c:torch.from_numpy(img.transpose(2,0,1).copy()) for c in cams}
            with patch.object(aug,'episode_colors',side_effect=AssertionError('overlay-only must not read HDF5')):
                s,te,fs,ft=aug.SampleAugment(cfg,42,0,3,'unused','task').apply_pair(t,{k:v.clone() for k,v in t.items()})
            for c in cams:
                self.assertTrue(torch.equal(te[c],t[c]));self.assertTrue(torch.equal(s[c],fs[c]));self.assertTrue(torch.equal(te[c],ft[c]))
                diff=(s[c].int()-t[c].int()).abs().sum(0)>0
                self.assertGreater(diff.float().mean().item(),.9)            # texture covers the whole image, objects included

    def test_overlay_formula_and_wrist_without_wall(self):
        img=np.full((40,60,3),200,np.uint8);img[20:30,20:30]=50
        tex={'table':np.full((256,256,3),100,np.uint8),'wall':np.full((256,256,3),10,np.uint8)}
        out=aug.overlay_texture(img,tex,(None,0.0,.4,3.0)).astype(float)
        lum=img.astype(float).mean(-1,keepdims=True)/img.mean()
        self.assertTrue(np.allclose(out,np.clip(.6*img+.4*100*lum,0,255).astype(np.uint8),atol=1))
        self.assertTrue(np.all(out[25,25]<out[5,5]))                       # dark object stays darker than background
        top=aug.overlay_texture(img,tex,(0.5,0.0,.5,1.0))
        self.assertLess(top[2,5].mean(),top[38,5].mean())                   # wall texture above the line
        nowall=[aug.overlay_params(aug.sample_rng(1,0,i,'w'),'camera_wrist_left',{'wrist_wall_p':0.3})[0] is None for i in range(400)]
        self.assertTrue(.6<np.mean(nowall)<.8)
        heads=[aug.overlay_params(aug.sample_rng(1,0,i,'h'),'camera_top',{})[0] for i in range(100)]
        self.assertTrue(all(.05<=y<=.35 for y in heads))

    def test_idle_arm_detection(self):
        T=10;arm=torch.zeros(T,12);grip=torch.zeros(T,2)
        arm[:,6:]=torch.linspace(0,.5,T)[:,None]                             # right arm moves, left still
        self.assertEqual(aug.detect_idle_arm({'action.arm.position':arm,'action.effector.position':grip},{}),'left')
        g2=grip.clone();g2[5:,0]=.5                                          # left gripper closes -> not idle
        self.assertIsNone(aug.detect_idle_arm({'action.arm.position':arm,'action.effector.position':g2},{}))
        both=torch.zeros(T,12)
        self.assertIsNone(aug.detect_idle_arm({'action.arm.position':both,'action.effector.position':grip},{}))
        pad=torch.zeros(T,dtype=torch.bool);pad[6:]=True;moving_tail=torch.zeros(T,12);moving_tail[6:,:6]=1;moving_tail[:,6:]=arm[:,6:]
        self.assertEqual(aug.detect_idle_arm({'action.arm.position':moving_tail,'action.effector.position':grip,'action_is_pad':pad},{}),'left')

    def test_v2b_idle_focus_cameras_and_side(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            img,cams,colors,cfg=self.v2b_fixture(Path(tmp))
            cfg['distractor'].update(p=0,idle_focus_p=1.0,max_count=4);cfg['texture']['p']=0
            t={c:torch.from_numpy(img.transpose(2,0,1).copy()) for c in cams}
            with patch.object(aug,'episode_colors',return_value=colors):
                for i in range(20):
                    a=aug.SampleAugment(cfg,42,0,i,'unused','task');a.idle_arm='left'
                    s,_,_,_=a.apply_pair(t)
                    self.assertTrue(torch.equal(s['camera_wrist_right'],t['camera_wrist_right']))   # active arm view untouched
                    changed=((s['camera_top'].int()-t['camera_top'].int()).abs().sum(0)>0).numpy()
                    self.assertFalse(changed[:,80+8:].any())   # objects in idle (left) half; only soft shadow may bleed a few px
            a=aug.SampleAugment(cfg,42,0,0,'unused','task');a.idle_arm=None
            with patch.object(aug,'episode_colors',return_value=colors):
                s,_,_,_=a.apply_pair(t)
            self.assertTrue(all(torch.equal(s[c],t[c]) for c in cams))                               # no idle arm, p=0 -> nothing

    def test_v2b_distractor_textured_and_shadowed(self):
        import cv2
        img=np.full((60,80,3),230,np.uint8)
        patch=np.zeros((10,10,4),np.uint8);patch[...,:3]=[150,30,20];patch[...,3]=255
        shadowed=aug.paste_distractors_v2b(img,[(30,20,patch,0.4)])
        self.assertLess(shadowed[30,35].mean(),229)                          # contact shadow under the object
        self.assertTrue(np.array_equal(shadowed[22,35],[150,30,20]))
        tex={'table':np.full((256,256,3),100,np.uint8)}
        both=aug.overlay_texture(shadowed,tex,(None,0.0,.4,3.0))
        self.assertFalse(np.array_equal(both[22,35],[150,30,20]))           # pasted object gets the overlay like real ones

    def test_v2b_near_object_hard_negatives(self):
        img=np.full((120,160,3),235,np.uint8);img[50:70,100:120]=(30,160,40)
        table=~((img==[30,160,40]).all(-1))
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,cfg=self.v2b_fixture(Path(tmp))
            d=cfg['distractor'];d.update(near_object_p=1.0,max_count=1)
            dist=[]
            for i in range(30):
                for x,y,p,_ in aug.plan_distractors_v2b(d,'task','camera_top',[table],img,aug.sample_rng(2,0,i,'n')):
                    dist.append(np.hypot(x+p.shape[1]/2-110,y+p.shape[0]/2-60))
            self.assertGreater(len(dist),20);self.assertLess(np.median(dist),40)

    def test_shortcut_probe_idle_frame_selection(self):
        probe=module('shortcut_probe',Path(__file__).resolve().parents[1]/'scripts/shortcut_probe.py')
        T=120;joint=np.zeros((T,14),np.float32)
        joint[:,7:13]=np.linspace(0,1,T)[:,None]                           # right arm moves the whole time
        joint[60:,0:6]=np.linspace(0,1,T-60)[:,None]                        # left arm starts moving at t=60
        frames=probe.find_idle_frames(joint,chunk=50,stride=10)
        self.assertIn((0,'left'),frames);self.assertIn((10,'left'),frames)
        self.assertFalse(any(t+50>60 for t,s in frames if s=='left'))      # chunk must not overlap left-arm motion
        self.assertFalse(any(s=='right' for _,s in frames))

    def test_library_reproducible(self):
        import hashlib,json,h5py,cv2
        builder=module('library_builder',Path(__file__).resolve().parents[1]/'scripts/build_distractor_library.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();raw=root/'episode0.hdf5'
            image=np.full((240,320,3),220,np.uint8);image[100:116,100:116]=[20,80,220]
            encoded=cv2.imencode('.png',image)[1].tobytes()
            with h5py.File(raw,'w') as f:f.create_dataset('observation/head_camera/rgb',data=np.array([encoded],dtype=f'S{len(encoded)}'))
            (root/'configs/clean_episodes_fixture.manifest.json').write_text(json.dumps({'episodes':[{'task':'fixture','episode':0,'path':'episode0.hdf5','sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}]}))
            builder.build(root,root/'one',SOURCE);builder.build(root,root/'two',SOURCE)
            self.assertEqual((root/'one/manifest.json').read_bytes(),(root/'two/manifest.json').read_bytes())
            self.assertGreater(len(json.loads((root/'one/manifest.json').read_text())['items']),0)

    def test_all_2500(self):
        index=aug.seen_index(os.environ['LINGBOT_ROOT'])
        self.assertEqual(len(index),2500)
        self.assertEqual(len({x[2] for x in index.values()}),50)


if __name__=='__main__':unittest.main(verbosity=2)
