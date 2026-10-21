import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

p = Path(__file__).resolve().parents[1] / 'reproduction/step2000/replay500.py'
spec = importlib.util.spec_from_file_location('replay', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ReplayCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.history = self.root / 'history'
        for setting in m.SETTINGS:
            for task in m.TASKS:
                for i in range(5):
                    d = dict(setting=setting,task=task,index=i,seed=i,instruction='test',initial_hash='x',setup_hash='y',simulation_rng={},policy_seed=42,instruction_type='seen' if setting=='demo_clean' else 'unseen')
                    p = self.history / 'cases' / setting / task / f'{i:02d}.json'
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
                    p = self.history / 'bf16' / setting / task / f'{i:02d}.json'
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(d,status='complete')))
    def tearDown(self):
        self.tmp.cleanup()
    def test_roundtrip_and_corruption(self):
        bundle=self.root/'bundle';m.export_bundle(self.history,bundle)
        self.assertEqual(len(m.verify_bundle(bundle)),500)
        p=bundle/'cases/demo_clean/lift_pot/00.json';d=json.loads(p.read_text());d['seed']=99;p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'checksum'):m.verify_bundle(bundle)
    def test_incomplete_record_refused(self):
        p=self.history/'bf16/demo_clean/lift_pot/00.json';p.write_text('{"status":"started"}')
        with self.assertRaisesRegex(ValueError,'incomplete'):m.export_bundle(self.history,self.root/'bundle')
    def test_identity_mismatch_refused(self):
        p=self.history/'cases/demo_clean/lift_pot/00.json';d=json.loads(p.read_text());d['index']=1;p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'identity'):m.read_cases(self.history/'cases')
    def test_existing_bundle_not_overwritten(self):
        bundle=self.root/'bundle';m.export_bundle(self.history,bundle)
        with self.assertRaises(FileExistsError):m.export_bundle(self.history,bundle)

if __name__=='__main__': unittest.main()
