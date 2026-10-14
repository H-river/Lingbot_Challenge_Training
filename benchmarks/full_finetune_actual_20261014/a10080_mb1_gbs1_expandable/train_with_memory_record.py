#!/usr/bin/env python3
"""Execute the untouched official trainer; add only per-rank final memory accounting."""
import json,os,pathlib,runpy,sys,time
import torch,yaml
cfg=pathlib.Path(sys.argv[1]).resolve();c=yaml.safe_load(cfg.read_text())
rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank);torch.cuda.reset_peak_memory_stats(local_rank)
entry=pathlib.Path(os.environ['LINGBOT_ROOT'])/'repos/lingbot-vla-v2/tasks/vla/train_lingbotvla.py'
started=time.monotonic();completed=False
try:
    sys.argv=[str(entry),str(cfg)]
    runpy.run_path(str(entry),run_name='__main__')
    completed=True
finally:
    out=pathlib.Path(c['train']['output_dir'])
    out.mkdir(parents=True,exist_ok=True)
    report={'rank':rank,'completed':completed,'total_process_wall_s_including_startup_and_saves':time.monotonic()-started,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated(local_rank)/1024**3,
        'peak_reserved_GiB':torch.cuda.max_memory_reserved(local_rank)/1024**3,
        'entry':str(entry),'training_logic_modified':False}
    (out/f'memory_rank_{rank}.json').write_text(json.dumps(report,indent=2))
    print('SMOKE_MEMORY',json.dumps(report),flush=True)
