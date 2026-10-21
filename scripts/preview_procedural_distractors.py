#!/usr/bin/env python3
"""Preview sheets for procedural distractors (V4 flat / V5 shaded). CPU only, no simulator.

Input frames come either from --frames (a directory of <task>__<camera>.png clean frames) or
from --root (release clean HDF5 via the clean_episodes_*.manifest.json files; frame --frame of
episode --episode). One or more --config files (e.g. V4 and V5); per task one sheet with rows =
head / left wrist / right wrist and columns:
  original | V2b (optional, cropped from an earlier V2b sheet via --before) |
  <cfg> normal ... | <cfg> idle=LEFT arm ... | <cfg> idle=RIGHT arm ...
with the configs side by side inside each group. All configs use the SAME sample keys, so V4 and
V5 columns show the same shapes, colours, sizes and positions. data.aug is used unchanged except:
idle columns force idle_focus_p=1 and set the idle arm by hand (in training the arm comes from the
release action chunk and the probability from the config); --texture-p overrides texture.p for
the preview only (default: keep the config's training probability, so most columns have no
texture). "normal" uses the first sample index whose distractor decision is on (search starts at
--start + 10 x task position). Also writes gallery_<cfg>.jpg (every shape x colour class) and
placement_<cfg>.json (placement success per camera, timing, idle-side check). With --frames,
table/wall colours are estimated on the given frame (training uses episode_colors on frame 0).
"""
import argparse
import copy
import importlib.util
import json
import re
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

CAMERAS = ('camera_top', 'camera_wrist_left', 'camera_wrist_right')
H5 = {'camera_top': 'head_camera', 'camera_wrist_left': 'left_camera', 'camera_wrist_right': 'right_camera'}


def load_augment(source):
    spec = importlib.util.spec_from_file_location('aug', Path(source) / 'lingbotvla/data/vla_data/augment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frames_from_dir(directory):
    out = {}
    for p in sorted(Path(directory).glob('*__camera_*.png')):
        task, camera = p.stem.split('__')
        out.setdefault(task, {})[camera] = np.asarray(Image.open(p).convert('RGB'))
    return {t: c for t, c in out.items() if set(c) == set(CAMERAS)}


def frames_from_root(root, aug, episode, frame, tasks):
    import h5py
    out, raw_paths = {}, {}
    for manifest in sorted(Path(root, 'configs').glob('clean_episodes_*.manifest.json')):
        data = json.loads(manifest.read_text())
        task = data['tasks'][0]
        if tasks and task not in tasks:
            continue
        entry = next(e for e in data['episodes'] if e['episode'] == episode)
        raw = aug.safe_path(Path(root) / entry['path'])
        with h5py.File(raw, 'r') as h:
            out[task] = {}
            for cam, name in H5.items():
                buf = h[f'observation/{name}/rgb'][min(frame, len(h[f'observation/{name}/rgb']) - 1)]
                im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                out[task][cam] = cv2.cvtColor(cv2.resize(im, (320, 240)), cv2.COLOR_BGR2RGB)
        raw_paths[task] = str(raw)
    return out, raw_paths


def before_column(path, rows=3):
    """Second column (index 1) of an earlier 4-column V2b sheet: 320x240 cells, 24 px header,
    4 px gaps (layout of evidence/augmentation_preview/revision_20261019_v2b/*.jpg)."""
    im = Image.open(path).convert('RGB')
    return [np.asarray(im.crop((324, 24 + r * 244, 644, 264 + r * 244))) for r in range(rows)]


def run(aug, cfg, cams, colors, index, idle):
    t = {c: torch.from_numpy(cams[c].transpose(2, 0, 1).copy()) for c in CAMERAS}
    with patch.object(aug, 'episode_colors', return_value=colors):
        sample = aug.SampleAugment(cfg, 42, 0, index, 'preview', 'preview')
        sample.idle_arm = idle
        student = sample.apply_pair(t)[0]
    return {c: aug.uint8_hwc(student[c]) for c in CAMERAS}, sample.distractor_on


def gallery(aug, path, style='shaded'):
    shapes, colours = aug.PROCEDURAL_SHAPES, tuple(aug.PROCEDURAL_COLORS)
    cell = 120
    canvas = np.full((cell * len(shapes) + 24, cell * len(colours), 3), 232, np.uint8)
    for i, s in enumerate(shapes):
        for j, c in enumerate(colours):
            p = aug.procedural_distractor(aug.sample_rng(0, i, j, 'gallery'), {'shapes': [s], 'colors': [c], 'style': style}, 56)
            h, w = p.shape[:2]
            scale = min(1.0, (cell - 8) / w)
            if scale < 1:
                p = cv2.resize(p, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
                h, w = p.shape[:2]
            x, y = j * cell + (cell - w) // 2, 24 + i * cell + (cell - h) // 2
            a = p[..., 3:].astype(np.float32) / 255
            canvas[y:y + h, x:x + w] = (canvas[y:y + h, x:x + w] * (1 - a) + p[..., :3] * a).astype(np.uint8)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    for j, c in enumerate(colours):
        d.text((j * cell + 6, 6), c, fill='black')
    im.save(path, quality=92)


def placement_stats(aug, dc, frames, n=30):
    """Per camera: share of plans with >=1 object, mean count, plan+paste time; head idle-side check."""
    import time
    out = {}
    for cam in CAMERAS:
        placed, times = [], []
        for task, cams in sorted(frames.items()):
            im = cams[cam]
            table = aug.region_mask(im, aug.estimate_colors(im)['table'])
            for i in range(n):
                t0 = time.perf_counter()
                plan = aug.plan_distractors_v2b(dc, task, cam, [table], im, aug.sample_rng(42, 0, i, f'distractor:{cam}'))
                aug.paste_distractors_v2b(im, plan)
                times.append(time.perf_counter() - t0)
                placed.append(len(plan))
        out[cam] = {'at_least_one': round(float(np.mean(np.array(placed) > 0)), 3), 'mean_count': round(float(np.mean(placed)), 2),
                    'plan_paste_ms_median': round(1000 * float(np.median(times)), 1),
                    'plan_paste_ms_p90': round(1000 * float(np.percentile(times, 90)), 1)}
    ok = total = 0
    for task, cams in sorted(frames.items()):
        im = cams['camera_top']
        table = aug.region_mask(im, aug.estimate_colors(im)['table'])
        for i in range(n):
            for side in ('left', 'right'):
                for x, _, p, _ in aug.plan_distractors_v2b(dc, task, 'camera_top', [table], im, aug.sample_rng(42, 0, i, f'idle:{side}'), focus=side):
                    total += 1
                    ok += (x + p.shape[1] / 2 < im.shape[1] / 2) == (side == 'left')
    out['head_idle_side_ok'] = f'{ok}/{total}'
    out['samples_per_task_and_camera'] = n
    return out


def main(a):
    aug = load_augment(a.source)
    cv2.setNumThreads(1); torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    raw_paths = {}
    if a.frames:
        frames = frames_from_dir(a.frames)
    else:
        frames, raw_paths = frames_from_root(a.root, aug, a.episode, a.frame, a.tasks)
    if a.tasks:
        frames = {t: v for t, v in frames.items() if t in a.tasks}
    if not frames:
        raise SystemExit('no frames')
    configs = []
    for path in a.config:
        cfg = yaml.safe_load(path.read_text())['data']['aug']
        if cfg.get('texture', {}).get('source') != 'procedural':
            raise SystemExit('preview expects texture.source procedural (no texture assets needed)')
        if a.texture_p is not None:
            cfg['texture']['p'] = a.texture_p
        idle = copy.deepcopy(cfg); idle['distractor']['idle_focus_p'] = 1.0
        label = re.sub(r'^train_8xA100_aug_', '', path.stem).upper()
        configs.append((label, cfg, idle))
    first = configs[0][1]
    records = []
    for k, (task, cams) in enumerate(sorted(frames.items())):
        colors = (aug.episode_colors(raw_paths[task]) if task in raw_paths
                  else {c: aug.estimate_colors(im) for c, im in cams.items()})
        index = next(i for i in range(a.start + 10 * k, a.start + 10 * k + 1000)
                     if aug.SampleAugment(first, 42, 0, i, 'preview', 'preview').distractor_on)
        columns = [('original', cams)]
        before = Path(a.before, f'{task}.jpg') if a.before else None
        if before and before.is_file():
            columns.append(('V2b (earlier sheet)', dict(zip(CAMERAS, before_column(before)))))
        for title, offset, arm, which in (('normal', 0, None, 1), ('idle=LEFT arm', 1, 'left', 2), ('idle=RIGHT arm', 2, 'right', 2)):
            for label, cfg, idle in configs:
                columns.append((f'{label} {title}', run(aug, cfg if which == 1 else idle, cams, colors, index + offset, arm)[0]))
        h, w = cams['camera_top'].shape[:2]
        sheet = Image.new('RGB', (len(columns) * (w + 4) - 4, 24 + 3 * (h + 4) - 4), 'white')
        draw = ImageDraw.Draw(sheet)
        for ci, (title, images) in enumerate(columns):
            draw.text((ci * (w + 4) + 4, 6), f'{task}  {title}' if ci == 0 else title, fill='black')
            for ri, cam in enumerate(CAMERAS):
                sheet.paste(Image.fromarray(np.ascontiguousarray(images[cam])), (ci * (w + 4), 24 + ri * (h + 4)))
        sheet.save(a.output / f'{task}.jpg', quality=90)
        records.append({'task': task, 'normal_index': index, 'idle_left_index': index + 1, 'idle_right_index': index + 2,
                        'columns': [t for t, _ in columns]})
    for label, cfg, _ in configs:
        gallery(aug, a.output / f'gallery_{label.lower()}.jpg', cfg['distractor'].get('procedural', {}).get('style', 'shaded'))
        stats = placement_stats(aug, cfg['distractor'], frames)
        (a.output / f'placement_{label.lower()}.json').write_text(json.dumps(stats, indent=2) + '\n')
    (a.output / 'preview.json').write_text(json.dumps({'configs': [str(p) for p in a.config], 'seed': 42, 'epoch': 0,
                                                        'texture_p_override': a.texture_p,
                                                        'frames': str(a.frames or a.root), 'sheets': records}, indent=2) + '\n')
    print(json.dumps(records))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--frames', type=Path)
    src.add_argument('--root', type=Path)
    p.add_argument('--source', type=Path, required=True, help='patched repos/lingbot-vla-v2')
    p.add_argument('--config', type=Path, nargs='+', required=True, help='one or more configs, e.g. V4 and V5')
    p.add_argument('--texture-p', type=float, help='preview-only override of texture.p (default: training value)')
    p.add_argument('--output', type=Path, required=True, help='new directory')
    p.add_argument('--before', type=Path, help='directory of earlier V2b sheets named <task>.jpg')
    p.add_argument('--tasks', nargs='*', default=[])
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--start', type=int, default=0, help='first sample index to consider')
    main(p.parse_args())
