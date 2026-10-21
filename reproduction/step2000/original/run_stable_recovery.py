import os,json,time,subprocess,socket,fcntl,traceback,threading,queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
R=Path('/root/lingbot_challenge');O=R/'results/step2000_bf16';lock=open(O/'stable_recovery.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
TASKS='lift_pot hanging_mug stack_bowls_three scan_object handover_block click_bell put_object_cabinet open_microwave stack_blocks_three place_shoe adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock dump_bin_bigbin grab_roller handover_mic move_can_pot move_pillbottle_pad move_playingcard_away place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand move_stapler_pad open_laptop pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket press_stapler rotate_qrcode shake_bottle_horizontally shake_bottle stack_blocks_two stack_bowls_two stamp_seal turn_switch put_bottles_dustbin'.split()
SETTINGS=['demo_clean','demo_randomized'];active={};errors=[];mutex=threading.Lock();started=time.time();servers=[]
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def status(state):
 with mutex:save(O/'stable_recovery_status.json',{'state':state,'updated_at':time.time(),'started_at':started,'active':active.copy(),'errors':errors.copy()})
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
 cmd=[str(R/'envs/robotwin/bin/python'),'-u','script/eval_policy_client_step2000_stable.py','--config','policy/ACT/deploy_policy.yml','--overrides','--task_name',task,'--task_config',setting,'--instruction_type','seen' if setting=='demo_clean' else 'unseen','--train_config_name','0','--seed','0','--policy_name','ACT','--port',str(9430+slot),'--robo_name','robotwin_base500','--eval_video_log','False','--output_dir',str(O/'client_results'/setting)]
 with mutex:active[str(slot)]={'task':task,'setting':setting,'offset':offset,'count':limit-offset}
 status('EVALUATING')
 log=O/f'{setting}_{task}_{offset}_{time.time_ns()}.log'
 with log.open('x') as f:p=subprocess.run(cmd,cwd=R/'repos/RoboTwin',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=21600)
 if p.returncode:raise RuntimeError(f'Client exit {p.returncode}: {log}')
 for i in range(offset,limit):assert json.loads((d/f'{i:02d}.json').read_text())['status']=='complete'
 with mutex:active.pop(str(slot),None)
try:
 if ready(9430):raise RuntimeError('Unexpected occupied recovery port')
 f=(O/'stable_recovery_server.log').open('a')
 p=subprocess.Popen(['bash',str(O/'start_server.sh'),'9430'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 servers.append(p);(O/'stable_recovery_server.pid').write_text(str(p.pid));wait_server(9430)
 for task in ['dump_bin_bigbin','place_phone_stand']:
  try:run(task,'demo_randomized',0)
  except Exception as e:
   errors.append({'task':task,'error':str(e)});active.pop('0',None);status('RECOVERY_WITH_ERRORS')
 status('RECOVERY_DONE')
finally:
 for p in servers:p.terminate()
