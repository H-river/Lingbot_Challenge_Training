#!/usr/bin/env python3
"""Read-only preflight: verify clean seen provenance and numeric episode mapping.
Does not import the simulator, open evaluation assets, or decode video.
"""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    import pandas as pd
    manifests = sorted((root / 'configs').glob('clean_episodes_*.manifest.json'))
    checked, tasks, failures, provenance_warnings = 0, [], [], []
    for manifest in manifests:
        data = json.loads(manifest.read_text())
        episodes = data['episodes']
        task = data['tasks'][0]
        if len(episodes) != 50 or {e['episode'] for e in episodes} != set(range(50)):
            raise ValueError(f'Incomplete clean manifest: {manifest}')
        dataset = root / 'data/clean_processed/lerobot' / task
        provenance = json.loads((dataset / 'conversion_provenance.json').read_text())
        if provenance['input_manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
            combined = root / 'configs/clean_episodes.manifest.json'
            if not combined.is_file() or hashlib.sha256(combined.read_bytes()).hexdigest() != provenance['input_manifest_sha256']:
                raise ValueError(f'Conversion provenance mismatch: {task}')
            original = [e for e in json.loads(combined.read_text())['episodes'] if e['task'] == task]
            if original != episodes:
                raise ValueError(f'Combined manifest entries mismatch: {task}')
        if provenance.get('instruction_field') != 'seen' or provenance.get('sample_only'):
            raise ValueError(f'Not full seen conversion: {task}')
        files = sorted((dataset / 'meta/episodes').rglob('*.parquet'))
        meta = pd.concat([pd.read_parquet(f, columns=['episode_index', 'tasks']) for f in files])
        if len(meta) != 50 or set(meta.episode_index) != set(range(50)):
            raise ValueError(f'Incomplete LeRobot metadata: {task}')
        meta = meta.set_index('episode_index')
        for e in episodes:
            path = (root / e['instructions_path']).resolve()
            if 'unseen' in path.parts:
                raise ValueError('Forbidden instruction path')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != e['instructions_sha256']:
                raise ValueError(f'Seen SHA mismatch: {task}/{e["episode"]}')
            obj = json.loads(raw)
            if set(obj) != {'seen'} or not isinstance(obj['seen'], list) or not obj['seen']:
                raise ValueError(f'Not seen-only: {task}/{e["episode"]}')
            prompts = list(meta.loc[e['episode'], 'tasks'])
            if len(prompts) != 1 or prompts[0] not in obj['seen']:
                failures.append({'task': task, 'episode': e['episode']})
            checked += 1
        tasks.append(task)
    if len(tasks) != 50 or checked != 2500:
        raise ValueError(f'Expected 50/2500, got {len(tasks)}/{checked}')
    return {'tasks': len(tasks), 'episodes_checked': checked, 'mapping_failures': failures,
            'mapping_basis': 'numeric episode order inserted by scripts/convert_clean.py; checked against all existing prompts',
            'provenance_manifest_mismatch_tasks': provenance_warnings, 'passed': not failures}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)
