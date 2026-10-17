#!/usr/bin/env python3
"""Export private scene metadata, or strictly replay the historical 500 cases.
No expert calls, scene replacement, or automatic retry of started policy cases.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROBOTWIN_COMMIT = '13c3c47ff4312dd62484bcd51be034af55c062d1'
TASKS = 'lift_pot hanging_mug stack_bowls_three scan_object handover_block click_bell put_object_cabinet open_microwave stack_blocks_three place_shoe adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock dump_bin_bigbin grab_roller handover_mic move_can_pot move_pillbottle_pad move_playingcard_away place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand move_stapler_pad open_laptop pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket press_stapler rotate_qrcode shake_bottle_horizontally shake_bottle stack_blocks_two stack_bowls_two stamp_seal turn_switch put_bottles_dustbin'.split()
SETTINGS = ['demo_clean', 'demo_randomized']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def read_cases(directory):
    paths = []
    for setting in SETTINGS:
        for task in TASKS:
            for i in range(5):
                p = directory / setting / task / f'{i:02d}.json'
                d = json.loads(p.read_text())
                if (d['setting'], d['task'], d['index']) != (setting, task, i):
                    raise ValueError(f'Case identity mismatch: {p}')
                for key in ['seed', 'instruction', 'initial_hash', 'setup_hash', 'simulation_rng']:
                    if key not in d:
                        raise ValueError(f'Missing {key}: {p}')
                if d['policy_seed'] != 42 or d['instruction_type'] != ('seen' if setting == 'demo_clean' else 'unseen'):
                    raise ValueError(f'Unexpected seed/instruction protocol: {p}')
                paths.append(p)
    if set(paths) != set(directory.glob('*/*/*.json')):
        raise ValueError('Extra or duplicate scene files')
    return paths


def export_bundle(history, destination):
    paths = read_cases(history / 'cases')
    # Require a completed record for every historical scene. Never package training data.
    for p in paths:
        d = json.loads((history / 'bf16' / p.relative_to(history / 'cases')).read_text())
        scene = json.loads(p.read_text())
        if d['status'] != 'complete' or any(d.get(k) != scene[k] for k in ['task', 'setting', 'index', 'seed', 'instruction', 'initial_hash', 'setup_hash', 'policy_seed']):
            raise ValueError(f'Historical case incomplete: {p}')
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for p in paths:
        name = p.relative_to(history / 'cases')
        q = destination / 'cases' / name
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        manifest[str(name)] = digest(q)
    save(destination / 'manifest.json', {'version': 1, 'cases': manifest, 'private_external_asset': True})
    print(f'Exported 500 scene records to {destination}; keep outside Git.')


def verify_bundle(bundle):
    paths = read_cases(bundle / 'cases')
    expected = json.loads((bundle / 'manifest.json').read_text())['cases']
    actual = {str(p.relative_to(bundle / 'cases')): digest(p) for p in paths}
    if actual != expected:
        raise ValueError('Scene bundle checksum mismatch')
    return actual


def install_same(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f'Refusing to overwrite different file: {target}')
    else:
        shutil.copyfile(source, target)


def ready(port):
    try:
        with socket.create_connection(('127.0.0.1', port), 1):
            return True
    except OSError:
        return False


def run(a):
    if ROOT != Path('/root/lingbot_challenge'):
        raise ValueError('Historical setup hashes require checkout at /root/lingbot_challenge')
    if not a.execute:
        raise ValueError('Pass --execute to run local evaluation; no competition submission is performed')
    cases = verify_bundle(a.bundle)
    rt = ROOT / 'repos/RoboTwin'
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=rt, text=True).strip()
    if rev != ROBOTWIN_COMMIT or subprocess.check_output(['git', 'diff', 'HEAD', '--'], cwd=rt):
        raise ValueError('RoboTwin revision or tracked files differ')
    if a.weights.resolve() == a.output.resolve() or a.output.resolve() in a.weights.resolve().parents:
        raise ValueError('Output must not contain weights')
    weights = sorted(a.weights.glob('*.safetensors'))
    if not weights:
        raise ValueError('Missing safetensors weights')
    env_python = [Path(os.environ[k]) / 'bin/python' for k in ['LINGBOT_ENV', 'ROBOTWIN_ENV']]
    if not all(p.is_file() for p in env_python):
        raise ValueError('Both inference and RoboTwin environments are required')
    fingerprint = {'cases': cases, 'weights': {p.name: digest(p) for p in weights},
                   'inference': digest(HERE / 'inference.yaml'),
                   'normalization': digest(ROOT / 'configs/clean_norm/robotwin.json'),
                   'robotwin': rev, 'precision': 'bf16', 'chunk': 50, 'compile': False,
                   'code': {str(p.relative_to(HERE)): digest(p) for p in [Path(__file__), HERE / 'robotwin/eval_policy_client_replay500.py', HERE / 'policy/lingbot_vla_v2_policy_base.py']}}
    historical = json.loads((HERE / 'evidence/export_verified.json').read_text())
    expected_weights = {x['file']: x['sha256'] for x in historical['shards']}
    if fingerprint['weights'] != expected_weights:
        raise ValueError('Weights differ from verified step2000 export; this entrypoint is historical replay only')
    provenance = json.loads((HERE / 'evidence/inference_config_provenance.json').read_text())
    if fingerprint['normalization'] != provenance['normalization_sha256']:
        raise ValueError('Normalization differs from historical evaluation')
    if a.output.exists() and not a.resume:
        raise ValueError('Output exists. Use a fresh output or explicit --resume.')
    a.output.mkdir(parents=True, exist_ok=a.resume)
    lock = (a.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = a.output / 'inputs.json'
    if a.resume:
        if json.loads(prior.read_text()) != fingerprint:
            raise ValueError('Resume inputs changed')
    else:
        save(prior, fingerprint)
        shutil.copytree(a.bundle / 'cases', a.output / 'cases')
    # Recheck the working copies on resume too.
    if {str(p.relative_to(a.output / 'cases')): digest(p) for p in read_cases(a.output / 'cases')} != cases:
        raise ValueError('Working scene records changed')
    install_same(HERE / 'robotwin/eval_policy_client_replay500.py', rt / 'script/eval_policy_client_replay500.py')
    install_same(HERE / 'robotwin/episode_recording.py', rt / 'script/episode_recording.py')
    for p in (HERE / 'robotwin/deploy').glob('*.py'):
        install_same(p, rt / 'script/deploy' / p.name)
    policy_root = ROOT / 'repos/lingbot-vla-v2'
    install_same(HERE / 'policy/lingbot_vla_v2_policy_base.py', policy_root / 'deploy/lingbot_vla_v2_policy_base.py')
    install_same(HERE / 'robotwin_base500.yaml', policy_root / 'configs/robot_configs/robotwin_base500.yaml')
    layout = a.output / 'model_layout'
    model = layout / 'checkpoints/global_step_2000/hf_ckpt'
    if not a.resume:
        model.parent.mkdir(parents=True)
        model.symlink_to(a.weights.resolve(), target_is_directory=True)
        shutil.copyfile(HERE / 'inference.yaml', layout / 'lingbotvla_cli.yaml')
    if model.resolve() != a.weights.resolve() or digest(layout / 'lingbotvla_cli.yaml') != fingerprint['inference']:
        raise ValueError('Model layout changed')
    ports = [a.port, a.port + 1]
    if any(ready(p) for p in ports):
        raise ValueError('Evaluation port occupied; refusing to touch existing servers')
    env = os.environ.copy()
    env.update(LINGBOT_EVAL_OUTPUT=str(a.output), BASELINE_PRECISION='bf16',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='42',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0',
               VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
               QWEN3VL_PATH=str(ROOT / 'models/Qwen3-VL-4B-Instruct-config'))
    servers, errors = [], []
    started = time.time()

    def evaluate(task, setting, slot, limit=5):
        directory = a.output / 'bf16' / setting / task
        offset = 0
        while offset < limit and (directory / f'{offset:02d}.json').exists():
            if json.loads((directory / f'{offset:02d}.json').read_text())['status'] != 'complete':
                raise RuntimeError('Interrupted policy case requires manual inspection; never auto-retry')
            offset += 1
        if offset == limit:
            return
        e = dict(env, BASELINE_CASE_OFFSET=str(offset), BASELINE_EPISODES_PER_TASK=str(limit-offset))
        command = [str(env_python[1]), '-u', 'script/eval_policy_client_replay500.py',
                   '--config', 'policy/ACT/deploy_policy.yml', '--overrides', '--task_name', task,
                   '--task_config', setting, '--instruction_type', 'seen' if setting == 'demo_clean' else 'unseen',
                   '--train_config_name', '0', '--seed', '0', '--policy_name', 'ACT',
                   '--port', str(ports[slot]), '--robo_name', 'robotwin_base500',
                   '--eval_video_log', 'False', '--output_dir', str(a.output / 'client_results' / setting)]
        log = a.output / f'{setting}_{task}_{offset}_{time.time_ns()}.log'
        with log.open('x') as f:
            subprocess.run(command, cwd=rt, env=e, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=21600)
        for i in range(offset, limit):
            if json.loads((directory / f'{i:02d}.json').read_text())['status'] != 'complete':
                raise RuntimeError(f'Client did not finish case {i}')

    try:
        save(a.output / 'status.json', {'state': 'STARTING', 'started_at': started})
        for slot, port in enumerate(ports):
            command = [str(env_python[0]), '-u', '-m', 'deploy.lingbot_vla_v2_policy_base',
                       '--model_path', str(model), '--use_length', '50', '--chunk_ret', 'True',
                       '--port', str(port), '--use_bf16', 'True', '--use_fp32', 'False', '--use_compile', 'False']
            with (a.output / f'server{slot}.log').open('a') as f:
                server = subprocess.Popen(command, cwd=policy_root, env=env, stdout=f, stderr=subprocess.STDOUT)
            servers.append(server)
            for _ in range(180):
                if server.poll() is not None:
                    raise RuntimeError('Policy server exited; inspect its log')
                if ready(port):
                    break
                time.sleep(2)
            else:
                raise TimeoutError('Policy server startup timeout')
        evaluate('lift_pot', 'demo_clean', 0, 1)  # Full rollout smoke, included in 500.
        save(a.output / 'status.json', {'state': 'EVALUATING', 'started_at': started})
        jobs = queue.Queue()
        for setting in SETTINGS:
            for task in TASKS:
                jobs.put((task, setting))

        def worker(slot):
            while True:
                try:
                    task, setting = jobs.get_nowait()
                except queue.Empty:
                    return
                try:
                    evaluate(task, setting, slot)
                except Exception as e:
                    errors.append({'task': task, 'setting': setting, 'error': str(e)})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, range(2)))
        records = [json.loads(p.read_text()) for p in (a.output / 'bf16').glob('*/*/*.json')]
        completed = [r for r in records if r['status'] == 'complete']
        summary = {'completed': len(completed), 'planned': 500, 'successes': sum(r['success'] for r in completed),
                   'elapsed_seconds': time.time()-started, 'errors': errors,
                   'settings': {s: {'completed': sum(r['setting'] == s for r in completed),
                                    'successes': sum(r['success'] for r in completed if r['setting'] == s)} for s in SETTINGS}}
        save(a.output / 'summary.json', summary)
        save(a.output / 'status.json', {'state': 'COMPLETE' if len(completed) == 500 else 'NEEDS_INSPECTION', **summary})
        if len(completed) != 500:
            raise RuntimeError('Incomplete replay; inspect infrastructure errors without resampling')
    except BaseException as e:
        save(a.output / 'failure.json', {'error': str(e), 'time': time.time()})
        raise
    finally:
        for server in servers:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export-bundle', help='Copy private historical scene/RNG metadata outside Git')
    e.add_argument('--history', type=Path, required=True)
    e.add_argument('--destination', type=Path, required=True)
    c = sub.add_parser('check-bundle', help='Validate all 500 records/checksums, no simulator')
    c.add_argument('--bundle', type=Path, required=True)
    q = sub.add_parser('run')
    q.add_argument('--bundle', type=Path, required=True)
    q.add_argument('--weights', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--port', type=int, default=9430)
    q.add_argument('--resume', action='store_true')
    q.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.command == 'export-bundle':
        export_bundle(a.history.resolve(), a.destination.resolve())
    elif a.command == 'check-bundle':
        print(f'PASS {len(verify_bundle(a.bundle.resolve()))} historical cases')
    else:
        a.bundle, a.weights, a.output = a.bundle.resolve(), a.weights.resolve(), a.output.resolve()
        run(a)


if __name__ == '__main__':
    main()
