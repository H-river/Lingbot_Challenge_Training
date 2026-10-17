from pathlib import Path
import json,time
from torch.distributed.checkpoint import FileSystemReader
root=Path('/root/deliverables/trained_ckpt/global_step_2000')
out={'checked_at':time.time(),'state':'WAITING_TRANSFER','directories':{}}
for name in ('model','optimizer'):
 reader=FileSystemReader(root/name);m=reader.read_metadata();expected={}
 for v in m.storage_data.values():expected[v.relative_path]=max(expected.get(v.relative_path,0),v.offset+v.length)
 bad=[]
 for f,size in expected.items():
  p=root/name/f;actual=p.stat().st_size if p.exists() else 0
  if actual<size:bad.append({'file':f,'minimum_bytes':size,'actual_bytes':actual})
 out['directories'][name]={'expected_files':len(expected),'missing_or_short':bad,'tensor_entries':len(m.state_dict_metadata),'minimum_bytes':sum(expected.values())}
 print(name,'expected_files',len(expected),'missing_or_short',len(bad),'minimum_bytes',sum(expected.values()))
 if name=='model':
  print('model metadata sample',[(k,str(v.size),str(v.properties.dtype)) for k,v in list(m.state_dict_metadata.items())[:3]])
Path('/root/lingbot_challenge/results/step2000_bf16/transfer_check.json').write_text(json.dumps(out,indent=2))
