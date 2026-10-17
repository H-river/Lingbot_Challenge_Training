import os,json,time,subprocess,socket,fcntl,traceback,threading,queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
R=Path('/root/lingbot_challenge');O=R/'results/step2000_bf16';lock=open(O/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
TASKS='lift_pot hanging_mug stack_bowls_three scan_object handover_block click_bell put_object_cabinet open_microwave stack_blocks_three place_shoe adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock dump_bin_bigbin grab_roller handover_mic move_can_pot move_pillbottle_pad move_playingcard_away place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand move_stapler_pad open_laptop pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket press_stapler rotate_qrcode shake_bottle_horizontally shake_bottle stack_blocks_two stack_bowls_two stamp_seal turn_switch put_bottles_dustbin'.split()
SETTINGS=['demo_clean','demo_randomized'];active={};errors=[];mutex=threading.Lock();started=time.time();servers=[]
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def status(state):
 with mutex:save(O/'status.json',{'state':state,'updated_at':time.time(),'started_at':started,'active':active.copy(),'errors':errors.copy()})
def ready(port):
 try:
  with socket.create_connection(('127.0.0.1',port),1):return True
 except OSError:return False
def wait_server(port):
 for _ in range(180):
  if ready(port):return
  time.sleep(2)
 raise RuntimeError('Server startup timeout '+str(port))
def run(task,setting,slot,limit=5):
 d=O/'bf16'/setting/task;offset=0
 while offset<limit:
  p=d/f'{offset:02d}.json'
  if not p.exists():break
  if json.loads(p.read_text()).get('status')!='complete':raise RuntimeError('Interrupted record needs inspection '+str(p))
  offset+=1
 if offset==limit:return
 env=os.environ.copy();env.update(BASELINE_PRECISION='bf16',BASELINE_CASE_OFFSET=str(offset),BASELINE_EPISODES_PER_TASK=str(limit-offset),VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',PYTHONUNBUFFERED='1',SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0')
 cmd=[str(R/'envs/robotwin/bin/python'),'-u','script/eval_policy_client_step2000.py','--config','policy/ACT/deploy_policy.yml','--overrides','--task_name',task,'--task_config',setting,'--instruction_type','seen' if setting=='demo_clean' else 'unseen','--train_config_name','0','--seed','0','--policy_name','ACT','--port',str(9430+slot),'--robo_name','robotwin_base500','--eval_video_log','False','--output_dir',str(O/'client_results'/setting)]
 with mutex:active[str(slot)]={'task':task,'setting':setting,'offset':offset,'count':limit-offset}
 status('EVALUATING')
 log=O/f'{setting}_{task}_{offset}_{time.time_ns()}.log'
 with log.open('x') as f:p=subprocess.run(cmd,cwd=R/'repos/RoboTwin',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=21600)
 if p.returncode:raise RuntimeError(f'Client exit {p.returncode}: {log}')
 for i in range(offset,limit):assert json.loads((d/f'{i:02d}.json').read_text())['status']=='complete'
 with mutex:active.pop(str(slot),None)
try:
 assert (O/'export_verified.json').exists()
 status('SMOKE_TEST');wait_server(9430);run('lift_pot','demo_clean',0,1)
 save(O/'smoke_verified.json',{'state':'PASS','case':'bf16/demo_clean/lift_pot/00.json','counted_in_500':True,'note':'Full real observation rollout, policy success not required for pipeline readiness.'})
 if ready(9431):raise RuntimeError('Unowned server port9431')
 f=(O/'server_slot1.log').open('a');p=subprocess.Popen(['bash',str(O/'start_server.sh'),'9431'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);servers.append(p);(O/'server_slot1.pid').write_text(str(p.pid));wait_server(9431)
 jobs=queue.Queue()
 for setting in SETTINGS:
  for task in TASKS:jobs.put((task,setting))
 def worker(slot):
  while True:
   try:task,setting=jobs.get_nowait()
   except queue.Empty:return
   try:run(task,setting,slot)
   except Exception as e:
    with mutex:errors.append({'task':task,'setting':setting,'error':str(e)});active.pop(str(slot),None)
    status('EVALUATING_WITH_ERRORS')
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,[0,1]))
 records=[]
 for s in SETTINGS:
  for task in TASKS:
   for i in range(5):
    p=O/'bf16'/s/task/f'{i:02d}.json'
    if p.exists():
     d=json.loads(p.read_text())
     if d.get('status')=='complete':records.append(d)
 save(O/'summary.json',{'planned':500,'completed':len(records),'successes':sum(d['success'] for d in records),'complete':len(records)==500,'errors':errors,'elapsed_seconds':time.time()-started,'formal_submission':False})
 status('COMPLETE' if len(records)==500 else 'NEEDS_INSPECTION')
except Exception as e:
 errors.append({'error':str(e),'traceback':traceback.format_exc()});status('BLOCKED');raise
finally:
 for p in servers:
  p.terminate()
 # initial server is this workflow's PID, started before the runner.
 p=O/'server.pid'
 if p.exists():
  try:os.kill(int(p.read_text()),15)
  except ProcessLookupError:pass
