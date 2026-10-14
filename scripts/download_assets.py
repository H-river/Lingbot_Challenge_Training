#!/usr/bin/env python3
"""Pinned, resumable exact-file downloads. Never downloads a whole dataset."""
import argparse,hashlib,json,os,pathlib,shutil,urllib.request,time,fcntl,concurrent.futures,sys
root=pathlib.Path(os.environ['LINGBOT_ROOT'])
p=argparse.ArgumentParser();p.add_argument('scope',choices=['base','clean','simulation','qwen','moge']);p.add_argument('--task');p.add_argument('--workers',type=int,default=1);p.add_argument('--attempts',type=int,default=4);p.add_argument('--min-rate-kib',type=int,default=256);args=p.parse_args()
assert 1<=args.attempts<=20 and 0<=args.min_rate_kib<=1024
assert 1<=args.workers<=4, 'workers must be 1..4'
locks=root/'cache/locks';locks.mkdir(exist_ok=True)
lock=(locks/(args.scope+'.lock')).open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Same-scope download is already running')
if args.scope=='clean':
 m=json.loads((root/'configs/clean_archives.manifest.json').read_text());repo=m['repository'];rev=m['revision'];kind='datasets/'
 files=[{'rfilename':f['path'],'size':f['size'],'lfs':{'sha256':f['sha256']}} for f in m['archives'] if args.task is None or f['task']==args.task]
 if not files:raise SystemExit('Unknown task; nothing downloaded')
 dest=root/'data/clean_raw/archives'
else:
 m=json.loads((root/'configs/assets.manifest.json').read_text())[{'base':'models','simulation':'simulation_assets','qwen':'qwen','moge':'moge'}[args.scope]];repo=m['repo'];rev=m['revision'];files=m['files'];kind='datasets/' if args.scope=='simulation' else ''
 dest=root/{'base':'models/lingbot-vla-v2-6b','simulation':'assets/archives','qwen':'models/Qwen3-VL-4B-Instruct-config','moge':'models/MoGe-2-vitb-normal'}[args.scope]
assert repo in ['robbyant/lingbot-vla-v2-6b','TianxingChen/RoboTwin2.0','Qwen/Qwen3-VL-4B-Instruct','Ruicheng/moge-2-vitb-normal']
reserve=20*1024**3
def download_one(f):
 rel=pathlib.PurePosixPath(f['rfilename'])
 assert not rel.is_absolute() and '..' not in rel.parts
 target=dest.joinpath(*rel.parts); target.parent.mkdir(parents=True,exist_ok=True)
 expected=f['size'];sha=f.get('lfs',{}).get('sha256')
 def valid(path):
  if not path.exists() or path.stat().st_size!=expected:return False
  h=hashlib.sha256() if sha else hashlib.sha1()
  if not sha:h.update(f'blob {expected}\0'.encode())
  with path.open('rb') as inp:
   for chunk in iter(lambda:inp.read(8*1024**2),b''):h.update(chunk)
  return h.hexdigest()==(sha or f['blobId'])
 if valid(target):print('VERIFIED',target,flush=True);return
 if target.exists():raise SystemExit(f'Existing file failed integrity check: {target}')
 partial=target.with_name(target.name+'.partial');offset=partial.stat().st_size if partial.exists() else 0
 if shutil.disk_usage(target.parent).free < expected-offset+reserve:raise SystemExit(f'SPACE_PENDING: need {expected-offset+reserve} bytes available')
 url=f.get('mirror_url') or f'https://huggingface.co/{kind}{repo}/resolve/{rev}/{f["rfilename"]}'
 print('DOWNLOAD',url,'bytes',expected,'resume',offset,flush=True)
 for attempt in range(args.attempts):
  try:
   offset=partial.stat().st_size if partial.exists() else 0
   if offset>expected:raise RuntimeError('Partial exceeds declared size; manual integrity investigation required')
   if offset==expected:break
   # Fresh official resolve request prevents reuse of a cached signed range response.
   request_url=url+('&' if '?' in url else '?')+'download=true&resume_request='+str(time.time_ns())
   req=urllib.request.Request(request_url,headers={'Range':f'bytes={offset}-'} if offset else {})
   with urllib.request.urlopen(req,timeout=45) as response:
    if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise RuntimeError('Server did not honor resume offset')
    started=time.monotonic(); received=0;last_progress=started
    with partial.open('ab' if offset else 'wb') as out:
     while chunk:=response.read(64*1024):
      if out.tell()+len(chunk)>expected:raise RuntimeError('Response exceeds declared file size')
      if shutil.disk_usage(target.parent).free<reserve+len(chunk):raise RuntimeError('20 GiB reserve reached')
      out.write(chunk); received+=len(chunk)
      now=time.monotonic();elapsed=now-started
      if now-last_progress>=30:
       print('PROGRESS',target,out.tell(),expected,'rate_KiB_s',round(received/max(elapsed,0.001)/1024,1),flush=True);last_progress=now
      if elapsed>60:
       if received/elapsed<args.min_rate_kib*1024:raise TimeoutError(f'Sustained transfer below {args.min_rate_kib} KiB/s; reconnecting with resume')
       started=time.monotonic();received=0
    if partial.stat().st_size!=expected:raise EOFError(f'Incomplete HTTP response: {partial.stat().st_size}/{expected}; resuming')
   break
  except Exception as e:
   print('RETRY',target,attempt+1,type(e).__name__,str(e),flush=True)
   if attempt==args.attempts-1:raise
   if 'reserve reached' in str(e) or 'exceeds declared' in str(e) or 'manual integrity' in str(e):raise
   time.sleep(min(30,2**attempt))
 if not valid(partial):raise SystemExit(f'Checksum/size failure: {partial}')
 partial.rename(target)
 receipt={'repository':repo,'revision':rev,'url':url,'mirror_revision':f.get('mirror_revision'),'size':expected,'sha256':sha,'local_path':str(target)}
 receipts=root/'docs/download_receipts';receipts.mkdir(exist_ok=True)
 (receipts/(hashlib.sha256(str(target).encode()).hexdigest()+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
 print('VERIFIED',target,flush=True)
errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
 futures={pool.submit(download_one,f):f['rfilename'] for f in files}
 for future in concurrent.futures.as_completed(futures):
  try:future.result()
  except (Exception,SystemExit) as e:
   errors.append((futures[future],type(e).__name__,str(e)));print('FILE_FAILED',errors[-1],flush=True)
if errors:
 print('DOWNLOAD_SCOPE_FAILED',args.scope,json.dumps(errors),flush=True);sys.exit(1)
print('DOWNLOAD_SCOPE_VERIFIED',args.scope,flush=True)
