#!/usr/bin/env python3
"""Bitwise regression: two augment.py versions on the same configs, samples and frames.

CPU only, no simulator. Loads data.aug from each YAML config, redirects its file paths to a
temporary fixture (seen textures, distractor library built with build_distractor_library's
own candidate filter from the given clean frames plus synthetic fixture blobs, optional ORACLE unseen dir), then runs
SampleAugment(...).apply_pair for every sample under both versions and compares every
student/teacher/future tensor with torch.equal. Exit code 1 on any difference.

Frames: a directory of <task>__<camera>.png clean frames (camera_top, camera_wrist_left,
camera_wrist_right). episode_colors() is replaced by estimate_colors() on the given frame,
the same function the real path applies to each episode's first frame.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

CAMERAS = ('camera_top', 'camera_wrist_left', 'camera_wrist_right')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root, frames, aug, builder):
    seen = root / 'background_texture' / 'seen'
    unseen = root / 'background_texture' / 'unseen'
    seen.mkdir(parents=True); unseen.mkdir(parents=True)
    for i in range(6):
        for d, offset in ((seen, 0), (unseen, 100)):
            tile = aug.procedural_texture(aug.sample_rng(7, 0, i + offset, 'fixture'))
            Image.fromarray(tile).save(d / f'{i}.png')
    library = root / 'distractors'
    library.mkdir()
    items = []
    for task, cams in frames.items():
        for k, (_, patch_rgba) in enumerate(builder.candidates(cams['camera_top'], aug)):
            name = f'{task}_{k:02d}.png'
            Image.fromarray(patch_rgba).save(library / name)
            items.append({'file': name, 'task': task, 'sha256': hashlib.sha256((library / name).read_bytes()).hexdigest()})
    # Re-compressed preview frames rarely pass the conservative cutout filter, so the fixture
    # always adds a few synthetic RGBA blobs owned by two fixture tasks. The library code path
    # (entry draw, read_cutout, resize, flip, paste) is identical whatever the pixels are.
    for k, (colour, axes) in enumerate((((180, 40, 30), (9, 6)), ((40, 90, 190), (6, 10)), ((60, 160, 70), (8, 8)), ((200, 170, 40), (11, 5)))):
        cut = np.zeros((24, 24, 4), np.uint8)
        cv2.ellipse(cut, (12, 12), axes, 0, 0, 360, (*colour, 255), -1)
        name = f'fixture_{k}.png'
        Image.fromarray(cut).save(library / name)
        items.append({'file': name, 'task': f'fixture_task_{k % 2}', 'sha256': hashlib.sha256((library / name).read_bytes()).hexdigest()})
    (library / 'manifest.json').write_text(json.dumps({'source': 'release_clean_first_head_frames', 'items': items}))
    return {'seen_dir': str(seen), 'unseen_dir': str(unseen), 'library_dir': str(library), 'cutouts': len(items)}


def redirect(aug_config, paths):
    c = copy.deepcopy(aug_config)
    if 'texture' in c:
        c['texture']['seen_dir'] = paths['seen_dir']
        if 'unseen_dir' in c['texture']:
            c['texture']['unseen_dir'] = paths['unseen_dir']
    if 'distractor' in c and 'library_dir' in c['distractor']:
        c['distractor']['library_dir'] = paths['library_dir']
    return c


def run(aug, config, frames, colors, samples, placed=None):
    outputs = []
    planners = {}
    for name in ('plan_distractors', 'plan_distractors_v2b'):
        original = getattr(aug, name)
        def counted(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            if placed is not None:
                placed[0] += len(result)
            return result
        planners[name] = counted
    with patch.object(aug, 'episode_colors', side_effect=lambda raw: colors[raw]), \
            patch.object(aug, 'plan_distractors', planners['plan_distractors']), \
            patch.object(aug, 'plan_distractors_v2b', planners['plan_distractors_v2b']):
        for index in range(samples):
            task = sorted(frames)[index % len(frames)]
            current = {c: torch.from_numpy(frames[task][c].transpose(2, 0, 1).copy()) for c in CAMERAS}
            # future frame: a small shift of the current one so current/future differ
            future = {c: torch.roll(t, shifts=(0, 2), dims=(1, 2)) for c, t in current.items()}
            sample = aug.SampleAugment(config, 42, 1, index, task, task)
            # exercise both idle-arm branches of V2b plus "no idle arm"
            sample.idle_arm = (None, 'left', 'right')[index % 3] if hasattr(sample, 'idle_arm') else None
            outputs.append(sample.apply_pair(current, future))
    return outputs


def main(a):
    old = load('aug_old', a.old)
    new = load('aug_new', a.new)
    builder = load('library_builder', Path(__file__).resolve().parent / 'build_distractor_library.py')
    cv2.setNumThreads(1); torch.set_num_threads(1)
    frames = {}
    for p in sorted(a.frames.glob('*__camera_*.png')):
        task, camera = p.stem.split('__')
        frames.setdefault(task, {})[camera] = np.asarray(Image.open(p).convert('RGB'))
    frames = {t: c for t, c in frames.items() if set(c) == set(CAMERAS)}
    if not frames:
        raise SystemExit('no complete <task>__<camera>.png frame sets')
    colors = {t: {c: new.estimate_colors(im) for c, im in cams.items()} for t, cams in frames.items()}
    report = {'old': str(a.old), 'old_sha256': hashlib.sha256(a.old.read_bytes()).hexdigest(),
              'new': str(a.new), 'new_sha256': hashlib.sha256(a.new.read_bytes()).hexdigest(),
              'tasks': sorted(frames), 'samples_per_config': a.samples, 'configs': {}}
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        paths = fixture(Path(tmp), frames, new, builder)
        report['fixture_cutouts'] = paths['cutouts']
        for cfg_path in a.configs:
            aug_config = redirect(yaml.safe_load(cfg_path.read_text())['data']['aug'], paths)
            env = {}
            if aug_config.get('texture', {}).get('source') == new.ORACLE_SOURCE:
                env = {new.ORACLE_ENV: new.ORACLE_TOKEN}
            with patch.dict(os.environ, env):
                for m in (old, new):
                    for fn in ('oracle_unseen_files', 'texture_files', 'distractor_library'):
                        getattr(m, fn).cache_clear()
                before = run(old, aug_config, frames, colors, a.samples)
                placed = [0]
                after = run(new, aug_config, frames, colors, a.samples, placed)
            tensors = equal = changed_pixels = 0
            for x, y in zip(before, after):
                for dx, dy in zip(x, y):
                    if set(dx) != set(dy):
                        failed = True
                    for cam in dx:
                        tensors += 1
                        same = torch.equal(dx[cam], dy[cam])
                        equal += same
                        changed_pixels += int((dx[cam] != dy[cam]).any(0).sum())
            # the run must actually exercise the augmentations it claims to compare
            touched = sum(int((s[c] != torch.from_numpy(frames[sorted(frames)[i % len(frames)]][c].transpose(2, 0, 1).copy())).any())
                          for i, (s, _, _, _) in enumerate(after) for c in s)
            row = {'tensors': tensors, 'bitwise_equal': equal, 'differing_pixels': changed_pixels,
                   'student_views_changed_by_augmentation': touched, 'distractors_placed': placed[0],
                   'texture': aug_config.get('texture', {}).get('mode', 'mask') + '/' + str(aug_config.get('texture', {}).get('source')),
                   'distractor': aug_config.get('distractor', {}).get('placement', 'v1') + '/' + aug_config.get('distractor', {}).get('source', 'library')}
            row['pass'] = tensors > 0 and equal == tensors
            failed |= not row['pass']
            report['configs'][cfg_path.name] = row
            print(cfg_path.name, json.dumps(row), flush=True)
    report['pass'] = not failed
    if a.output:
        a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True, help='previous augment.py')
    p.add_argument('--new', type=Path, required=True, help='candidate augment.py')
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--configs', type=Path, nargs='+', required=True)
    p.add_argument('--samples', type=int, default=60)
    p.add_argument('--output', type=Path)
    main(p.parse_args())
