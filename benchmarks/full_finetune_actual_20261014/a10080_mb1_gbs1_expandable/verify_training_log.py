#!/usr/bin/env python3
import json,math,pathlib,re,statistics,sys,yaml
c=yaml.safe_load(pathlib.Path(sys.argv[1]).read_text());text=pathlib.Path(sys.argv[2]).read_text(errors='replace');stage=sys.argv[3]
fields=['Loss','VLA_Loss','Depth_Loss','Future_Depth_Loss','FutureVideo_Loss','GradNorm','StepTime']
records=[]
for line in text.splitlines():
    hit=re.search(r'Step (\d+)/.*Loss ',line)
    if not hit:continue
    row={'step':int(hit.group(1))}
    for key in fields:
        match=re.search(r'(?<![\w])'+key+r'\s+([^,\s]+)',line)
        assert match,f'Missing {key} in step line'
        val=float(match.group(1).removesuffix('s'));assert math.isfinite(val),f'Non-finite {key}'
        row[key]=val
    records.append(row)
expected=list(range(1,12)) if stage=='initial' else [12]
assert [x['step'] for x in records]==expected,[x['step'] for x in records]
assert all(x['GradNorm']>0 for x in records),'No nonzero global training gradients'
assert not re.search(r'(?:Loss|GradNorm)\s+(?:nan|inf)',text,re.IGNORECASE)
steady=[x['StepTime'] for x in records if x['step']>=2]
out=pathlib.Path(c['train']['output_dir'])
memory=[json.loads((out/f'memory_rank_{i}.json').read_text()) for i in range(c['train']['global_batch_size']//(c['train']['micro_batch_size']*c['train']['gradient_accumulation_steps']))]
assert all(x['completed'] for x in memory)
report={'state':'FINITE_TRAINING_METRICS_VERIFIED','steps':records,'timing_source':'official trainer StepTime; not separately synchronized compute-only timing',
    'steps_2_11_mean_s':statistics.mean(steady) if stage=='initial' else None,
    'steps_2_11_sample_std_s':statistics.stdev(steady) if stage=='initial' else None,'memory':memory}
(out/'training_metrics_verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
