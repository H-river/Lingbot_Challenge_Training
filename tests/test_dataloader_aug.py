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
