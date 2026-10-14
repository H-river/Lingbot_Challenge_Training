#!/usr/bin/env python3
import datetime,json,pathlib,subprocess,time
p=pathlib.Path(__file__).resolve().parent
status=p/'status.json';start=time.monotonic()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
status.write_text(json.dumps({'state':'RUNNING','started_at':stamp(),'training_scope':'single A10080, full VLM+expert, mb1/GA1/gbs1, 11 steps, DCP only'},indent=2))
with (p/'supervisor.log').open('w') as log,(p/'resource_samples.jsonl').open('w') as resource:
 proc=subprocess.Popen(['bash',str(p/'run_initial.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 while proc.poll() is None:
  gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True)
  mem={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
  resource.write(json.dumps({'at':stamp(),'gpu_query_exit':gpu.returncode,'gpu_memory_MiB_and_utilization_percent':gpu.stdout.strip(),'host_available_GiB':mem['MemAvailable']/1024**3})+'\n');resource.flush()
  time.sleep(5)
 rc=proc.returncode
status.write_text(json.dumps({'state':'COMPLETE' if rc==0 else 'FAILED','exit_code':rc,'ended_at':stamp(),'elapsed_wall_s':time.monotonic()-start,'HF_export':'PENDING','resume':'PENDING','simulation_started':False},indent=2))
raise SystemExit(rc)
