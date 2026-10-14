#!/usr/bin/env python3
import io,urllib.request,zipfile,json,pathlib,os,concurrent.futures
root=pathlib.Path(os.environ['LINGBOT_ROOT']);m=json.loads((root/'configs/clean_archives.manifest.json').read_text());am=json.loads((root/'configs/assets.manifest.json').read_text())['simulation_assets']
class RangeFile(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if self.pos<0:raise ValueError('Negative seek')
  return self.pos
 def read(self,n=-1):
  n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
  if n<=0:return b''
  if n>8*1024**2:raise ValueError('Central directory exceeds 8 MiB probe budget')
  start=self.pos
  req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{start+n-1}'})
  with urllib.request.urlopen(req,timeout=25) as r:
   if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):raise ValueError('Range not honored')
   b=r.read(n)
  self.pos+=len(b);return b
previous_file=root/'results/space_estimate.json'
previous={e['name']:e for e in json.loads(previous_file.read_text())} if previous_file.exists() else {}
items=[(x['task'],x['path'],x['size'],m['revision'],'clean') for x in m['archives']]+[(x['rfilename'],x['rfilename'],x['size'],am['revision'],'asset') for x in am['files']]
def check(x):
 name,path,size,rev,kind=x
 if previous.get(name,{}).get('status')=='OK':return previous[name]
 try:
  url=f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{rev}/{path}'
  local=root/('data/clean_raw/archives' if kind=='clean' else 'assets/archives')/path
  with zipfile.ZipFile(local if local.is_file() else RangeFile(url,size)) as z:
   inf=z.infolist();selected=[i for i in inf if i.filename.startswith('aloha-agilex_clean_50/data/') or i.filename.startswith('aloha-agilex_clean_50/instructions/')] if kind=='clean' else inf
   result={'name':name,'kind':kind,'compressed':size,'expanded_all':sum(i.file_size for i in inf),'expanded_selected':sum(i.file_size for i in selected),'status':'OK'}
 except Exception as e:result={'name':name,'kind':kind,'compressed':size,'status':'UNKNOWN','error':str(e)}
 print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(check,items))
(root/'results/space_estimate.json').write_text(json.dumps(results,indent=2)+'\n')
print('KNOWN TOTAL',sum(x.get('expanded_selected',0) for x in results),'UNKNOWN',sum(x['status']!='OK' for x in results))
