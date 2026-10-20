#!/usr/bin/env python3
"""Offline probe for the "object in front of the gripper => move" shortcut.

Uses ONLY release clean data, no simulator. For frames where exactly one arm is
still in the demonstration's next chunk (the "idle" arm), it builds a pair of
observations that differ only by distractor objects pasted in front of the idle
arm (its wrist view + its half of the head view). The model is queried on both
with identical sampling noise. A shortcut shows up as the idle arm being
predicted to move only when the distractors are present.

Metrics (radians, absolute joint targets vs current state, idle arm only):
  idle_motion_clean / idle_motion_distractor   max_t |a_t - s| over the idle arm's 6 joints
  triggered   clean motion <= --move-thr but distractor motion > --move-thr

  --dry-run  : no model; only selects frames and writes the paired images (CPU check).
  otherwise  : needs the GPU inference env (QWEN3VL_PATH etc., like eval) and
               --model pointing at an exported hf_ckpt. NOT run in the authoring
               environment (no GPU) -- validate on the server before trusting numbers.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

CAMS = {'camera_top': 'head_camera', 'camera_wrist_left': 'left_camera', 'camera_wrist_right': 'right_camera'}
OBS_KEYS = {'camera_top': 'observation.images.cam_high', 'camera_wrist_left': 'observation.images.cam_left_wrist',
            'camera_wrist_right': 'observation.images.cam_right_wrist'}
LEFT, RIGHT = slice(0, 6), slice(7, 13)          # RoboTwin aloha-agilex joint vector: 6+gripper, 6+gripper
GRIP = {'left': 6, 'right': 13}


def load_augment(source):
    spec = importlib.util.spec_from_file_location('aug', Path(source) / 'lingbotvla/data/vla_data/augment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_idle_frames(joint, chunk=50, idle_thr=0.01, move_thr=0.05, stride=10):
    """Frames t where exactly one arm (6 joints + gripper) stays within idle_thr over
    [t, t+chunk) while the other arm's joints move more than move_thr."""
    out = []
    for t in range(0, len(joint) - chunk, stride):
        seg = joint[t:t + chunk]
        rng = {s: np.abs(seg[:, sl] - seg[0, sl]).max() for s, sl in (('left', LEFT), ('right', RIGHT))}
        grip = {s: np.abs(seg[:, GRIP[s]] - seg[0, GRIP[s]]).max() for s in GRIP}
        for idle, active in (('left', 'right'), ('right', 'left')):
            if rng[idle] < idle_thr and grip[idle] < idle_thr and rng[active] > move_thr:
                out.append((t, idle))
    return out


def read_joints(h):
    """(T, 14) = left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1), the
    same layout as the eval client's observation.state (joint_action/vector)."""
    if 'joint_action/vector' in h:
        return np.asarray(h['joint_action/vector'], np.float32)
    col = lambda k: np.asarray(h[f'joint_action/{k}'], np.float32).reshape(len(h[f'joint_action/{k}']), -1)
    return np.concatenate([col('left_arm'), col('left_gripper'), col('right_arm'), col('right_gripper')], 1)


def decode(buf):
    frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_pairs(args, aug):
    root = Path(args.root)
    rng = np.random.default_rng(args.seed)
    pairs = []
    for manifest in sorted((root / 'configs').glob('clean_episodes_*.manifest.json')):
        data = json.loads(manifest.read_text())
        task = data['tasks'][0]
        if args.tasks and task not in args.tasks:
            continue
        frames = []
        for entry in sorted(data['episodes'], key=lambda e: e['episode']):
            raw = aug.safe_path(root / entry['path'])
            with h5py.File(raw, 'r') as h:
                joint = read_joints(h)
            frames += [(entry, raw, t, idle) for t, idle in find_idle_frames(joint, args.chunk, args.idle_thr, args.move_thr)]
        if not frames:
            continue
        for k in rng.choice(len(frames), size=min(args.frames_per_task, len(frames)), replace=False):
            entry, raw, t, idle = frames[int(k)]
            with h5py.File(raw, 'r') as h:
                images = {c: decode(h[f'observation/{n}/rgb'][t]) for c, n in CAMS.items()}
                state = read_joints(h)[t]
            seen = json.loads(aug.safe_path(root / entry['instructions_path']).read_text())['seen']
            colors = aug.episode_colors(str(raw))
            pasted = dict(images)
            dc = {'library_dir': args.library, 'max_count': args.max_count, 'near_object_p': 0.0, 'shadow': True}
            if args.distractor_source == 'procedural':
                flat = args.procedural_style == 'flat'
                dc = {'source': 'procedural', 'max_count': args.max_count, 'near_object_p': 0.0, 'shadow': not flat,
                      'procedural': {'style': args.procedural_style}}
            n_obj = 0
            for cam in ('camera_top', aug.WRIST_CAMERAS[idle]):
                table = aug.region_mask(images[cam], colors[cam]['table'])
                key = hashlib.sha256(f'{task}:{entry["episode"]}:{t}:{cam}'.encode()).digest()
                place = aug.plan_distractors_v2b(dc, task, cam, [table], images[cam],
                                                 np.random.default_rng(int.from_bytes(key[:8], 'little')), focus=idle)
                n_obj += len(place)
                pasted[cam] = aug.paste_distractors_v2b(images[cam], place)
            if n_obj == 0:
                continue
            pairs.append(dict(task=task, episode=entry['episode'], t=t, idle=idle, instruction=seen[0],
                              state=state, clean=images, distractor=pasted, n_objects=n_obj))
    return pairs


def to_obs(images, state, instruction):
    obs = {OBS_KEYS[c]: img for c, img in images.items()}
    obs.update({'observation.state': state, 'task': instruction})
    return obs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True, help='LINGBOT_ROOT with configs/ manifests and raw clean data')
    p.add_argument('--source', required=True, help='patched repos/lingbot-vla-v2 (for augment.py and deploy/)')
    p.add_argument('--library', help='distractor library built by build_distractor_library.py (source library)')
    p.add_argument('--distractor-source', choices=['library', 'procedural'], default='library',
                   help='procedural = V4/V5 generated objects; a V4/V5-trained model has seen this object distribution')
    p.add_argument('--procedural-style', choices=['shaded', 'flat'], default='shaded',
                   help='with --distractor-source procedural: shaded (V5, contact shadow) or flat (V4, no shadow)')
    p.add_argument('--output', required=True, help='new output directory')
    p.add_argument('--model', help='exported hf_ckpt; required unless --dry-run')
    p.add_argument('--tasks', nargs='*', default=[])
    p.add_argument('--frames-per-task', type=int, default=4)
    p.add_argument('--chunk', type=int, default=50)
    p.add_argument('--idle-thr', type=float, default=0.01)
    p.add_argument('--move-thr', type=float, default=0.05)
    p.add_argument('--max-count', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fp32', action='store_true', help='FP32 inference (default BF16, as in the eval runs)')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.distractor_source == 'library' and not a.library:
        p.error('--library is required with --distractor-source library')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    aug = load_augment(a.source)
    pairs = build_pairs(a, aug)
    print(json.dumps({'pairs': len(pairs), 'tasks': len({x['task'] for x in pairs})}))
    for i, x in enumerate(pairs[:20]):
        top = np.concatenate([x['clean'][c] for c in CAMS], 1)
        bottom = np.concatenate([x['distractor'][c] for c in CAMS], 1)
        cv2.imwrite(str(out / f'pair_{i:02d}_{x["task"]}_{x["idle"]}.jpg'), cv2.cvtColor(np.concatenate([top, bottom], 0), cv2.COLOR_RGB2BGR))
    if a.dry_run:
        return
    if not a.model:
        raise SystemExit('--model is required unless --dry-run')
    import torch
    sys.path.insert(0, str(Path(a.source)))
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    server = LingbotVLAv2Server(a.model, use_length=a.chunk, chunk_ret=True, use_bf16=not a.fp32, use_fp32=a.fp32)
    server.infer(dict(reset=True, robo_name='robotwin'))
    rows = []
    for i, x in enumerate(pairs):
        motion = {}
        for kind in ('clean', 'distractor'):
            torch.manual_seed(a.seed + i)                      # identical sampling noise for both inputs
            torch.cuda.manual_seed_all(a.seed + i)
            action = server.infer(to_obs(x[kind], x['state'], x['instruction']))
            chunk = np.asarray(next(iter(v for k, v in action.items() if not k.startswith('_'))), np.float32)
            sl = LEFT if x['idle'] == 'left' else RIGHT
            motion[kind] = float(np.abs(chunk[:, sl] - x['state'][sl]).max())
        rows.append(dict(task=x['task'], episode=x['episode'], t=x['t'], idle=x['idle'], n_objects=x['n_objects'],
                         idle_motion_clean=motion['clean'], idle_motion_distractor=motion['distractor'],
                         triggered=motion['clean'] <= a.move_thr < motion['distractor']))
    with open(out / 'probe.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    clean = np.array([r['idle_motion_clean'] for r in rows]); dist = np.array([r['idle_motion_distractor'] for r in rows])
    summary = dict(model=a.model, pairs=len(rows), move_thr=a.move_thr,
                   idle_motion_clean_median=float(np.median(clean)), idle_motion_distractor_median=float(np.median(dist)),
                   mean_increase=float(np.mean(dist - clean)), triggered_rate=float(np.mean([r['triggered'] for r in rows])))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
