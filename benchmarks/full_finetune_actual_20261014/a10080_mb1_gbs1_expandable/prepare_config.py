#!/usr/bin/env python3
"""Generate an isolated full-training smoke config from the pinned official recipe."""
import argparse, ast, dataclasses, datetime, hashlib, json, os, pathlib, types
import yaml

p = argparse.ArgumentParser()
p.add_argument('--project', default=os.environ.get('LINGBOT_ROOT', '/root/lingbot_challenge'))
p.add_argument('--checkpoint-root')
p.add_argument('--world-size',type=int,choices=[1],default=1)
a = p.parse_args()
r = pathlib.Path(a.project).resolve()
dest = pathlib.Path(__file__).resolve().parent
repo = r / 'repos/lingbot-vla-v2'
source = repo / 'configs/vla/robotwin/robotwin.yaml'
c = yaml.safe_load(source.read_text())
c['model'].update(model_path=str(r/'models/lingbot-vla-v2-6b'), tokenizer_path=str(r/'models/Qwen3-VL-4B-Instruct-config'))
c['data'].update(train_path=str(r/'configs/train_clean.txt'), robot_config_root=str(r/'configs/robot_configs'), norm_stats_file=str(r/'configs/clean_norm/robotwin.json'), num_workers=2)
cp = pathlib.Path(a.checkpoint_root or str(r/'checkpoints')).resolve()
assert cp.is_absolute() and str(cp) != '/', 'Specify a dedicated checkpoint directory'
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
c['train'].update(output_dir=str(cp/f'full_clean50_smoke_{stamp}'), train_expert_only=False,
    freeze_vision_encoder=False, freeze_vit=False, enable_gradient_checkpointing=True,
    enable_full_shard=True, enable_mixed_precision=True, enable_fp32=False, use_compile=True,
    micro_batch_size=1, gradient_accumulation_steps=1, global_batch_size=1,
    max_steps=11, save_steps=11, save_epochs=0, save_hf_weights=False,
    enable_resume=False, async_save_hf_weights=False, seed=42)
c['train'].pop('load_checkpoint_path', None)
align = c['train']['align_params']
align['depth'].update(moge_path=str(r/'models/MoGe-2-vitb-normal/model.pt'), morgbd_path=str(r/'models/lingbot-vla-v2-6b/depth/model.pt'))
align['video'].update(ckpt_path=str(r/'models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth'), config_path=str(r/'models/lingbot-vla-v2-6b/dino_video/config.yaml'))
c['eval'] = {'enabled': False, 'command': None}
path = dest/'full_clean50_smoke.yaml'
assert not path.exists(), 'Existing config retained; use a new preparation directory'
path.write_text(yaml.safe_dump(c, sort_keys=False))

# Extract exact dataclasses in memory. Do not import the GPU-dependent trainer or model.
trainer = repo/'tasks/vla/train_lingbotvla.py'
text = trainer.read_text()
nodes = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name in {'MyTrainingArguments','MyDataArguments','Arguments'}]
assert len(nodes) == 3
header = 'from dataclasses import dataclass, field\nfrom typing import Any, Dict, List, Optional, Literal\nfrom lingbotvla.utils.arguments import TrainingArguments, DataArguments, ModelArguments, EvalArguments\n'
module = types.ModuleType('smoke_argument_validation')
import sys
sys.modules[module.__name__] = module
exec(header+'\n\n'.join('@dataclass\n'+ast.get_source_segment(text,n) for n in nodes), module.__dict__)
from lingbotvla.utils.arguments import parse_args
os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE=str(a.world_size))
sys.argv = ['cpu_parse', str(path)]
parsed = parse_args(module.Arguments)
assert parsed.train.global_batch_size == parsed.train.micro_batch_size*parsed.train.gradient_accumulation_steps*a.world_size
assert not parsed.train.freeze_vit and not parsed.train.freeze_vision_encoder and not parsed.train.train_expert_only
assert parsed.train.enable_gradient_checkpointing and parsed.train.enable_full_shard
assert parsed.train.align_params and parsed.data.use_future_image
assert not parsed.eval.enabled
assert not parsed.train.load_checkpoint_path and not parsed.train.enable_resume
(dest/'parsed_config_cpu.json').write_text(json.dumps(dataclasses.asdict(parsed), indent=2, default=str))
original = yaml.safe_load(source.read_text())
changes = {section:{key:{'official':original.get(section,{}).get(key),'smoke':value} for key,value in vals.items() if original.get(section,{}).get(key)!=value} for section,vals in c.items()}
(dest/'config_provenance.json').write_text(json.dumps({'source_config':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'trainer_sha256':hashlib.sha256(trainer.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'world_size':a.world_size,'GPU_tested':False,'changes':changes},indent=2))
print(json.dumps({'config':str(path),'checkpoint_root':str(cp),'CPU_official_dataclass_parse':'PASS','GPU_tested':False},indent=2))
