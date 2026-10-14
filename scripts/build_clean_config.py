#!/usr/bin/env python3
import os,pathlib,json,yaml,hashlib
r=pathlib.Path(os.environ['LINGBOT_ROOT']);src=r/'repos/lingbot-vla-v2';m=json.loads((r/'configs/clean_archives.manifest.json').read_text())
assert len(m['archives'])==50 and len({e['task'] for e in m['archives']})==50
c=yaml.safe_load((src/'configs/vla/robotwin/robotwin.yaml').read_text())
c['model']['model_path']=str(r/'models/lingbot-vla-v2-6b');c['model']['tokenizer_path']=str(r/'models/Qwen3-VL-4B-Instruct-config')
c['data']['train_path']=str(r/'configs/train_clean.txt');c['data']['robot_config_root']=str(r/'configs/robot_configs');c['data']['norm_stats_file']=str(r/'configs/clean_norm/robotwin.json')
c['train']['output_dir']=str(r/'checkpoints')
# Explicit single-process reference; GPU launch must recompute from actual data parallel size.
c['train']['gradient_accumulation_steps']=32
a=c['train']['align_params'];a['depth']['moge_path']=str(r/'models/MoGe-2-vitb-normal/model.pt');a['depth']['morgbd_path']=str(r/'models/lingbot-vla-v2-6b/depth/model.pt');a['video']['ckpt_path']=str(r/'models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth');a['video']['config_path']=str(r/'models/lingbot-vla-v2-6b/dino_video/config.yaml')
(r/'configs/competition_clean.template.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
rc=yaml.safe_load((src/'configs/robot_configs/robotwin.yaml').read_text());rc['norm_stats']=str(r/'configs/clean_norm/robotwin.json');d=r/'configs/robot_configs';d.mkdir(exist_ok=True);(d/'robotwin.yaml').write_text(yaml.safe_dump(rc,sort_keys=False))
(r/'configs/train_clean.txt').write_text(''.join(f'robotwin {r}/data/clean_processed/lerobot/{e["task"]}\n' for e in m['archives']))
print('Created clean-only TEMPLATE. Original hyperparameters retained; not a hardware-tuned training launch config. Missing datasets/stats must block training.')
