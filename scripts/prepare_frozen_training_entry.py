import os,pathlib,hashlib,ast,difflib,json
r=pathlib.Path(os.environ['LINGBOT_ROOT']);repo=r/'repos/lingbot-vla-v2';src=repo/'tasks/vla/train_lingbotvla.py';original=src.read_text();benchmark=(r/'benchmarks/frozen_vlm_20261013/train_benchmark.py').read_text();start=benchmark.index('    # Benchmark policy:');end=benchmark.index('    use_depth_align =',start);block=benchmark[start:end]
assert hashlib.sha256(block.encode()).hexdigest()=='f3fcdfafa4bccb52b5ac8f9964834c8561515e96f5b4c89e9b4a99d1441d0094'
anchor='    use_depth_align = True if args.train.align_params != {} else False\n';assert original.count(anchor)==1
new=original.replace(anchor,'    assert args.train.train_expert_only, "Dedicated frozen entry requires train_expert_only=true"\n'+block+anchor,1);ast.parse(new)
dest=r/'scripts/train_frozen_clean.py'
if dest.exists():assert dest.read_text()==new,'Existing frozen entry differs; preserve and investigate'
else:dest.write_text(new)
# Dedicated copy retains the official checkpoint/finalization flow; benchmark early return is never copied.
assert 'return  # Timing-only run:' not in new
assert new.count('Checkpointer.save(')==original.count('Checkpointer.save(') and new.count('hf_saver.wait_all_across_ranks()')==original.count('hf_saver.wait_all_across_ranks()')
(r/'docs/patches').mkdir(parents=True,exist_ok=True);(r/'docs/patches/train_frozen_clean.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile=str(src),tofile=str(dest))))
(r/'results/frozen_entry_provenance.json').write_text(json.dumps({'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'generated_sha256':hashlib.sha256(new.encode()).hexdigest(),'freeze_policy_sha256':hashlib.sha256(block.encode()).hexdigest(),'only_change':'assert frozen config + exact benchmark whitelist inserted before parallel wrapping; official save/finalization untouched','GPU_tested':False},indent=2));print(dest)
