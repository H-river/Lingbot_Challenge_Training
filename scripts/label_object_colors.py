#!/usr/bin/env python3
"""Count the colour of the manipulated object(s) in every release clean episode.

Read-only. Uses only the release clean archives already extracted on the server:
  1. <task dir>/scene_info.json  (RoboTwin writes {"episode_k": {"info": {"{A}": "001_bottle/base13", ...}}})
  2. fallback: the episode's SEEN instructions, matched against the seen object descriptions
     encoded in docs/object_variant_colors.json (procedural pads/blocks: colour word in the instruction).
The 'unseen' instruction field is never read.

Outputs (new directory):
  episodes.csv       task, episode, key, variant, colour, source
  task_summary.json  per task: colour-bin counts of the primary object, and suggested in-task weights
                     w ∝ 1/sqrt(count of the episode's colour bin), capped at --cap x the task mean, mean 1.
"""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

# Which scene_info key is the object the policy has to grasp/move first (from RoboTwin envs/*.py).
PRIMARY_KEY = collections.defaultdict(lambda: '{A}', {
    'move_can_pot': '{B}', 'place_bread_basket': '{B}', 'place_bread_skillet': '{B}',
    'place_container_plate': '{B}', 'place_object_scale': '{B}',
})
PROCEDURAL = {  # tasks whose primary colour is a fixed/procedural colour, not a mesh variant
    'beat_block_hammer': 'fixed', 'blocks_ranking_rgb': 'fixed', 'blocks_ranking_size': 'random_rgb',
    'handover_block': 'fixed', 'stack_blocks_three': 'fixed', 'stack_blocks_two': 'fixed',
}
TASK_CANDIDATES = {  # primary-object variant pool per task (RoboTwin envs/*.py); bare model name = all its variants
    'adjust_bottle': ['001_bottle/base13', '001_bottle/base16'],
    'beat_block_hammer': ['020_hammer/base0'],
    'click_alarmclock': ['046_alarm-clock/base1', '046_alarm-clock/base3'],
    'click_bell': ['050_bell/base0', '050_bell/base1'],
    'dump_bin_bigbin': ['063_tabletrashbin/base0', '063_tabletrashbin/base3', '063_tabletrashbin/base7', '063_tabletrashbin/base8', '063_tabletrashbin/base9', '063_tabletrashbin/base10'],
    'grab_roller': ['102_roller/base0', '102_roller/base2'],
    'handover_mic': ['018_microphone/base0', '018_microphone/base4', '018_microphone/base5'],
    'hanging_mug': ['039_mug/base0', '039_mug/base1', '039_mug/base2', '039_mug/base3', '039_mug/base4', '039_mug/base5', '039_mug/base6', '039_mug/base7', '039_mug/base8', '039_mug/base9'],
    'lift_pot': ['060_kitchenpot/base0', '060_kitchenpot/base1'],
    'move_can_pot': ['105_sauce-can/base0', '105_sauce-can/base2', '105_sauce-can/base4', '105_sauce-can/base5', '105_sauce-can/base6'],
    'move_pillbottle_pad': ['080_pillbottle/base1', '080_pillbottle/base2', '080_pillbottle/base3', '080_pillbottle/base4', '080_pillbottle/base5'],
    'move_playingcard_away': ['081_playingcards/base0', '081_playingcards/base1', '081_playingcards/base2'],
    'move_stapler_pad': ['048_stapler/base0', '048_stapler/base1', '048_stapler/base2', '048_stapler/base3', '048_stapler/base4', '048_stapler/base5', '048_stapler/base6'],
    'open_laptop': ['015_laptop/base0', '015_laptop/base1', '015_laptop/base2', '015_laptop/base3', '015_laptop/base4', '015_laptop/base5', '015_laptop/base6', '015_laptop/base7', '015_laptop/base8', '015_laptop/base9', '015_laptop/base10'],
    'open_microwave': ['044_microwave/base0', '044_microwave/base1'],
    'pick_diverse_bottles': ['001_bottle/base0', '001_bottle/base1', '001_bottle/base2', '001_bottle/base3', '001_bottle/base4', '001_bottle/base5', '001_bottle/base6', '001_bottle/base7', '001_bottle/base8', '001_bottle/base9', '001_bottle/base10', '001_bottle/base11', '001_bottle/base12', '001_bottle/base13', '001_bottle/base14', '001_bottle/base15', '001_bottle/base16', '001_bottle/base17', '001_bottle/base18', '001_bottle/base19'],
    'pick_dual_bottles': ['001_bottle/base13', '001_bottle/base16'],
    'place_a2b_left': ['047_mouse', '048_stapler', '050_bell', '057_toycar', '073_rubikscube', '075_bread', '077_phone', '081_playingcards', '086_woodenblock', '112_tea-box', '113_coffee-box', '107_soap'],
    'place_a2b_right': ['047_mouse', '048_stapler', '050_bell', '057_toycar', '073_rubikscube', '075_bread', '077_phone', '081_playingcards', '086_woodenblock', '112_tea-box', '113_coffee-box', '107_soap'],
    'place_bread_basket': ['075_bread/base0', '075_bread/base1', '075_bread/base3', '075_bread/base5', '075_bread/base6'],
    'place_bread_skillet': ['075_bread/base0', '075_bread/base1', '075_bread/base3', '075_bread/base5', '075_bread/base6'],
    'place_burger_fries': ['006_hamburg/base0', '006_hamburg/base1', '006_hamburg/base2', '006_hamburg/base3', '006_hamburg/base4', '006_hamburg/base5'],
    'place_can_basket': ['071_can/base0', '071_can/base1', '071_can/base2', '071_can/base3', '071_can/base5', '071_can/base6'],
    'place_cans_plasticbox': ['071_can/base0', '071_can/base1', '071_can/base2', '071_can/base3', '071_can/base5', '071_can/base6'],
    'place_container_plate': ['002_bowl/base1', '002_bowl/base2', '002_bowl/base3', '002_bowl/base5', '021_cup/base1', '021_cup/base2', '021_cup/base3', '021_cup/base4', '021_cup/base5', '021_cup/base6', '021_cup/base7'],
    'place_dual_shoes': ['041_shoe/base0', '041_shoe/base1', '041_shoe/base2', '041_shoe/base3', '041_shoe/base4', '041_shoe/base5', '041_shoe/base6', '041_shoe/base7', '041_shoe/base8', '041_shoe/base9'],
    'place_empty_cup': ['021_cup/base0'],
    'place_fan': ['099_fan/base4', '099_fan/base5'],
    'place_mouse_pad': ['047_mouse/base0', '047_mouse/base1', '047_mouse/base2'],
    'place_object_basket': ['081_playingcards/base0', '081_playingcards/base1', '081_playingcards/base2', '057_toycar/base0', '057_toycar/base1', '057_toycar/base2', '057_toycar/base3', '057_toycar/base4', '057_toycar/base5'],
    'place_object_scale': ['047_mouse', '048_stapler', '050_bell'],
    'place_object_stand': ['047_mouse', '048_stapler', '050_bell', '073_rubikscube', '057_toycar', '079_remotecontrol'],
    'place_phone_stand': ['077_phone/base0', '077_phone/base1', '077_phone/base2', '077_phone/base4'],
    'place_shoe': ['041_shoe/base0', '041_shoe/base1', '041_shoe/base2', '041_shoe/base3', '041_shoe/base4', '041_shoe/base5', '041_shoe/base6', '041_shoe/base7', '041_shoe/base8', '041_shoe/base9'],
    'press_stapler': ['048_stapler/base0', '048_stapler/base1', '048_stapler/base2', '048_stapler/base3', '048_stapler/base4', '048_stapler/base5', '048_stapler/base6'],
    'put_bottles_dustbin': ['114_bottle/base1', '114_bottle/base2', '114_bottle/base3'],
    'put_object_cabinet': ['047_mouse', '048_stapler', '057_toycar', '073_rubikscube', '075_bread', '077_phone', '081_playingcards', '112_tea-box', '113_coffee-box', '107_soap'],
    'rotate_qrcode': ['070_paymentsign/base0', '070_paymentsign/base1', '070_paymentsign/base2', '070_paymentsign/base3'],
    'scan_object': ['112_tea-box/base0', '112_tea-box/base1', '112_tea-box/base2', '112_tea-box/base3', '112_tea-box/base4', '112_tea-box/base5'],
    'shake_bottle': ['001_bottle/base0', '001_bottle/base1', '001_bottle/base2', '001_bottle/base3', '001_bottle/base4', '001_bottle/base5', '001_bottle/base6', '001_bottle/base7', '001_bottle/base8', '001_bottle/base9', '001_bottle/base10', '001_bottle/base11', '001_bottle/base12', '001_bottle/base13', '001_bottle/base14', '001_bottle/base15', '001_bottle/base16', '001_bottle/base17', '001_bottle/base18', '001_bottle/base19'],
    'shake_bottle_horizontally': ['001_bottle/base0', '001_bottle/base1', '001_bottle/base2', '001_bottle/base3', '001_bottle/base4', '001_bottle/base5', '001_bottle/base6', '001_bottle/base7', '001_bottle/base8', '001_bottle/base9', '001_bottle/base10', '001_bottle/base11', '001_bottle/base12', '001_bottle/base13', '001_bottle/base14', '001_bottle/base15', '001_bottle/base16', '001_bottle/base17', '001_bottle/base18', '001_bottle/base19'],
    'stack_bowls_three': ['002_bowl/base3'],
    'stack_bowls_two': ['002_bowl/base3'],
    'stamp_seal': ['100_seal/base0', '100_seal/base2', '100_seal/base3', '100_seal/base4', '100_seal/base6'],
    'turn_switch': ['056_switch/base0', '056_switch/base1', '056_switch/base2', '056_switch/base3', '056_switch/base4', '056_switch/base5', '056_switch/base6', '056_switch/base7'],
}
COLOUR_WORDS = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black', 'gray', 'orange', 'purple', 'brown']


def load_table(repo):
    return json.loads((repo / 'docs/object_variant_colors.json').read_text())['variants']


def seen_descriptions(desc_dir, variant):
    f = desc_dir / variant.split('/')[0] / f'{variant.split("/")[1]}.json'
    return json.loads(f.read_text())['seen'] if f.is_file() else []


def from_instructions(seen_instr, candidates, desc_dir):
    """Pick the candidate variant whose seen descriptions occur most often in the seen instructions."""
    text = ' || '.join(s.lower() for s in seen_instr)
    best, score = None, 0
    for v in candidates:
        n = sum(text.count(d.lower()) for d in seen_descriptions(desc_dir, v))
        if n > score:
            best, score = v, n
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True, help='LINGBOT_ROOT (configs/clean_episodes_<task>.manifest.json, extracted clean data)')
    p.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--descriptions', help='RoboTwin description/objects_description (only for the instruction fallback)')
    p.add_argument('--output', required=True)
    p.add_argument('--cap', type=float, default=2.0)
    a = p.parse_args()
    root, repo, out = Path(a.root), Path(a.repo), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    table = load_table(repo)
    desc_dir = Path(a.descriptions) if a.descriptions else None
    rows, summary = [], {}
    for manifest in sorted((root / 'configs').glob('clean_episodes_*.manifest.json')):
        data = json.loads(manifest.read_text())
        task = data['tasks'][0]
        episodes = sorted(data['episodes'], key=lambda e: e['episode'])
        task_dir = (root / episodes[0]['path']).parents[1]
        scene = task_dir / 'scene_info.json'
        scene = json.loads(scene.read_text()) if scene.is_file() else None
        model_variants = collections.defaultdict(set)
        for v in table:
            model_variants[v.split('/')[0]].add(v)
        primary = []
        for e in episodes:
            info, source = {}, 'scene_info'
            if scene is not None:
                info = (scene.get(f'episode_{e["episode"]}') or {}).get('info', {})
            seen = json.loads((root / e['instructions_path']).read_text())['seen']   # never 'unseen'
            if not info:
                source = 'instructions'
            for key, value in sorted(info.items()):
                if not key[1:-1].isupper():          # {a}/{b} are arm tags
                    continue
                colour = table.get(value) or (value.lower() if value.lower() in COLOUR_WORDS else 'unknown')
                rows.append(dict(task=task, episode=e['episode'], key=key, variant=value, colour=colour, source=source))
            value = info.get(PRIMARY_KEY[task])
            if task in PROCEDURAL:
                colour = PROCEDURAL[task]
            elif value:
                colour = table.get(value, 'unknown')
            elif desc_dir is not None:                  # instruction fallback: needs the candidate model name
                cands = [c for x in TASK_CANDIDATES.get(task, []) for c in (sorted(model_variants[x]) if '/' not in x else [x])]
                value = from_instructions(seen, cands, desc_dir)
                colour = table.get(value, 'unknown') if value else 'unknown'
                rows.append(dict(task=task, episode=e['episode'], key=PRIMARY_KEY[task], variant=value, colour=colour, source=source))
            else:
                colour = 'unknown'
            primary.append(colour)
        counts = collections.Counter(c.split('/')[0] for c in primary)
        raw = [counts[c.split('/')[0]] ** -0.5 for c in primary]
        mean = sum(raw) / len(raw)
        w = [min(x / mean, a.cap) for x in raw]
        w = [x * len(w) / sum(w) for x in w]
        summary[task] = dict(primary_key=PRIMARY_KEY[task], source='scene_info' if scene is not None else 'instructions',
                             colour_counts=dict(counts.most_common()), n_bins=len(counts),
                             in_task_weight={str(e['episode']): round(x, 4) for e, x in zip(episodes, w)})
    with open(out / 'episodes.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'episode', 'key', 'variant', 'colour', 'source'])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'task_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    overall = collections.Counter()
    for s in summary.values():
        overall.update(s['colour_counts'])
    print(json.dumps({'tasks': len(summary), 'scene_info_tasks': sum(s['source'] == 'scene_info' for s in summary.values()),
                      'unknown_primary': overall.get('unknown', 0), 'overall_primary_colours': dict(overall.most_common())}, indent=1))


if __name__ == '__main__':
    main()
