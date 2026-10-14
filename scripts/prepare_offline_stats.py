#!/usr/bin/env python3
"""Copy official argument dataclasses and remove unused GPU imports from statistics only.
The original source, statistics equations, data loader and model code are unchanged.
"""
import ast,difflib,os,pathlib,hashlib,json
r=pathlib.Path(os.environ['LINGBOT_ROOT']);repo=r/'repos/lingbot-vla-v2'
source=(repo/'tasks/vla/train_lingbotvla.py').read_text();tree=ast.parse(source)
names={'MyTrainingArguments','MyDataArguments','Arguments'}
classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
assert len(classes)==3 and all(len(n.decorator_list)==1 and ast.unparse(n.decorator_list[0])=='dataclass' for n in classes)
header='# Extracted verbatim from pinned LingBot argument dataclasses; no training entrypoint imported.\nfrom dataclasses import dataclass, field\nfrom typing import Any, Dict, List, Optional, Literal\nfrom lingbotvla.utils.arguments import TrainingArguments, DataArguments, ModelArguments, EvalArguments\n\n'
argument_text=header+'\n\n'.join('@dataclass\n'+ast.get_source_segment(source,n) for n in classes)+'\n'
(r/'scripts/generated_offline_arguments.py').write_text(argument_text)
p=repo/'scripts/compute_norm_stats.py';old=p.read_text()
assert old.count('from lingbotvla.models import build_processor')==1
assert sum(isinstance(n,ast.Name) and n.id=='build_processor' for n in ast.walk(ast.parse(old)))==0
new=old.replace('from lingbotvla.models import build_processor','# Unused model import omitted for CPU-only statistics.')
new=new.replace('from tasks.vla.train_lingbotvla import MyTrainingArguments, MyDataArguments','from generated_offline_arguments import MyTrainingArguments, MyDataArguments')
(r/'scripts/generated_offline_norm_stats.py').write_text(new)
d=r/'docs/patches';d.mkdir(exist_ok=True)
(d/'cpu-norm-imports.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='upstream/scripts/compute_norm_stats.py',tofile='generated_offline_norm_stats.py')))
(d/'cpu-norm-provenance.json').write_text(json.dumps({'source_commit':'be969b8fd117fb70550c5d4bf4bc328211b5b1b6','official_stats_sha256':hashlib.sha256(old.encode()).hexdigest(),'adapter_sha256':hashlib.sha256(new.encode()).hexdigest(),'argument_dataclasses_sha256':hashlib.sha256(argument_text.encode()).hexdigest(),'changes':'Only unused model import and source of verbatim argument dataclasses; no statistics algorithm changes'},indent=2)+'\n')
print('Prepared official CPU statistics adapter and exact import-only diff')
