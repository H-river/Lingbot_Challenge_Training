import sys
# Explicit user authorization is distinct from an organizer quota exemption.
import os, json, hashlib, time, random
import torch
from pathlib import Path
# Explicit CLI invocation authorizes local replay; no organizer exemptions are asserted.

import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None
    video_fps = str(usr_args.get("video_fps", 10))

    # Inference is served by the single LingBot websocket model; no ACT model is instantiated.

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if usr_args.get("output_dir"):
        save_dir = Path(usr_args["output_dir"]) / task_name
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 命令行 --eval_video_log 优先于 YAML 配置
    if "eval_video_log" in usr_args:
        args["eval_video_log"] = usr_args["eval_video_log"]

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(os.environ["BASELINE_EPISODES_PER_TASK"])
    assert 1 <= test_num <= 5
    topk = 1

    # model = get_model(usr_args)
    # from IPython import embed;embed()
    from script.deploy.websocket_client_policy import WebsocketClientPolicy
    model = WebsocketClientPolicy(port=usr_args['port'])

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   video_fps=video_fps,
                                   instruction_type=instruction_type,
                                   usr_args=usr_args)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                video_fps="10",
                instruction_type=None,
                usr_args = None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    offset = int(os.environ.get('BASELINE_CASE_OFFSET', '0'))
    precision = os.environ['BASELINE_PRECISION']
    root = Path(os.environ['LINGBOT_EVAL_OUTPUT'])
    case_dir = root/'cases'/args['task_config']/task_name
    result_dir = root/precision/args['task_config']/task_name
    case_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    def write_json(path, obj):
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
        tmp.replace(path)
    def initial_hash(env):
        # Exact initial observations, joints, poses and configuration are compared on replay.
        obs = env.get_obs()
        h = hashlib.sha256()
        for key in ('head_camera', 'left_camera', 'right_camera'):
            a = np.ascontiguousarray(obs['observation'][key]['rgb'])
            h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        h.update(np.ascontiguousarray(obs['joint_action']['vector']).tobytes())
        for entity in env.scene.entities:
            h.update(str(entity.name).encode())
            h.update(np.asarray(entity.pose.p).tobytes())
            h.update(np.asarray(entity.pose.q).tobytes())
        return h.hexdigest()
    setup_hash = hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    # eval_func = eval_function_decorator(policy_name, "eval")
    # reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    if offset:
        previous = json.loads((case_dir/f'{offset-1:02d}.json').read_text())
        now_seed = previous['seed'] + 1
    attempts_file = case_dir/'expert_attempts.jsonl'
    if attempts_file.exists():
        for line in attempts_file.read_text().splitlines():
            attempt = json.loads(line)
            if attempt.get('case_index') == offset and attempt.get('status') == 'expert_infeasible':
                now_seed = max(now_seed, attempt['seed'] + 1)
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        case_index = offset + now_id
        case_file = case_dir/f'{case_index:02d}.json'
        result_file = result_dir/f'{case_index:02d}.json'
        if result_file.exists():
            raise RuntimeError(f'Refusing policy case replay: {result_file}')
        saved_case = json.loads(case_file.read_text())  # Required: never sample a new scene.
        if saved_case is not None:
            now_seed = saved_case['seed']
        expert_check = False  # Strict replay never invokes an expert.
        case_started = time.time()
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            attempts = [json.loads(x) for x in attempts_file.read_text().splitlines()] if attempts_file.exists() else []
            used = sum(x.get('status') == 'started' and x.get('case_index') == case_index for x in attempts)
            if used >= 5:
                raise RuntimeError(f'EXPERT_PRECHECK_LIMIT: {task_name} {case_index}')
            with (case_dir/'expert_attempts.jsonl').open('a') as f:
                f.write(json.dumps(dict(seed=now_seed, case_index=case_index, status='started', time=time.time()))+'\n')
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                with (case_dir/'expert_attempts.jsonl').open('a') as f:
                    f.write(json.dumps(dict(seed=now_seed, status='complete', plan_success=bool(TASK_ENV.plan_success), time=time.time()))+'\n')
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except AssertionError as e:
                if str(e) != 'target_pose cannot be None for move action.':
                    TASK_ENV.close_env()
                    raise
                with (case_dir/'expert_attempts.jsonl').open('a') as f:
                    f.write(json.dumps(dict(seed=now_seed,case_index=case_index,status='expert_infeasible',reason=str(e),time=time.time()))+'\n')
                TASK_ENV.close_env()
                now_seed += 1
                args['render_freq'] = render_freq
                continue
            except TypeError as e:
                frames = traceback.extract_tb(e.__traceback__)
                known = (task_name == 'click_alarmclock' and str(e) == "'NoneType' object is not subscriptable" and any(f.name == 'play_once' and 'get_grasp_pose' in (f.line or '') for f in frames))
                if not known:
                    TASK_ENV.close_env()
                    raise
                with (case_dir/'expert_attempts.jsonl').open('a') as f:
                    f.write(json.dumps(dict(seed=now_seed,case_index=case_index,status='expert_infeasible',reason='click_alarmclock get_grasp_pose returned None',time=time.time()))+'\n')
                TASK_ENV.close_env()
                now_seed += 1
                args['render_freq'] = render_freq
                continue
            except IndexError as e:
                frames = traceback.extract_tb(e.__traceback__)
                known = (task_name == 'put_bottles_dustbin' and str(e) == 'list index out of range' and any(f.name == 'play_once' and '.target_pose[2]' in (f.line or '') and ('left_action' in (f.line or '') or 'right_action' in (f.line or '')) for f in frames))
                if not known:
                    TASK_ENV.close_env()
                    raise
                with (case_dir/'expert_attempts.jsonl').open('a') as f:
                    f.write(json.dumps(dict(seed=now_seed,case_index=case_index,status='expert_infeasible',reason='put_bottles_dustbin grasp action list empty',time=time.time()))+'\n')
                TASK_ENV.close_env()
                now_seed += 1
                args['render_freq'] = render_freq
                continue
            except Exception:
                TASK_ENV.close_env()
                raise  # Infrastructure failures are not silently resampled scenes.


        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        if saved_case is None:
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = str(np.random.choice(results[0][instruction_type]))
        else:
            instruction = saved_case['instruction']

        # Restore evaluator-only fields normally initialized before expert movement.
        # Saved-case replay skips expert execution; use identical initial-scene formulas.
        from envs.utils import ArmTag, get_face_prod
        if saved_case is not None:
            if task_name in ('put_object_cabinet', 'place_object_scale'):
                TASK_ENV.arm_tag = ArmTag('right' if TASK_ENV.object.get_pose().p[0] > 0 else 'left')
                if task_name == 'put_object_cabinet':
                    TASK_ENV.origin_z = TASK_ENV.object.get_pose().p[2]
            elif task_name == 'open_laptop':
                face_prod = get_face_prod(TASK_ENV.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
                TASK_ENV.arm_tag = ArmTag('left' if face_prod > 0 else 'right')
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction
        observation_hash = initial_hash(TASK_ENV)
        def tuple_tree(x):
            return tuple(tuple_tree(v) for v in x) if isinstance(x, list) else x
        def rng_snapshot():
            n = np.random.get_state()
            return dict(python=random.getstate(), numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
                        torch=torch.get_rng_state().tolist(),
                        cuda=[x.tolist() for x in torch.cuda.get_rng_state_all()])
        def rng_restore(r):
            random.setstate(tuple_tree(r['python']))
            n=r['numpy']; np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
            torch.set_rng_state(torch.tensor(r['torch'],dtype=torch.uint8))
            torch.cuda.set_rng_state_all([torch.tensor(x,dtype=torch.uint8) for x in r['cuda']])
        if saved_case is not None:
            assert saved_case['initial_hash'] == observation_hash, 'INITIAL_SCENE_MISMATCH'
            assert saved_case['setup_hash'] == setup_hash, 'SETUP_MISMATCH'
            rng_restore(saved_case['simulation_rng'])
        else:
            saved_case = dict(task=task_name, setting=args['task_config'], index=case_index,
                              seed=now_seed, instruction=instruction, instruction_type=instruction_type,
                              initial_hash=observation_hash, setup_hash=setup_hash, policy_seed=42,
                              expert_solvable=True, simulation_rng=rng_snapshot())
            write_json(case_file, saved_case)
        record = dict({k:v for k,v in saved_case.items() if k != 'simulation_rng'}, precision=precision, status='started', started_at=time.time())
        write_json(result_file, record)


        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    video_fps,
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        from episode_recording import EpisodeRecorder
        recorder = EpisodeRecorder(root, precision, args['task_config'], task_name, case_index)
        recorder.enabled = False
        recorder.capture(TASK_ENV)
        succ = False
        path_to_pi_model = None
        if usr_args is not None and "new_ckpt_path" in usr_args:
            path_to_pi_model = usr_args['new_ckpt_path']
        ret = model.infer(dict(reset = True, robo_name=usr_args['robo_name'], path_to_pi_model=path_to_pi_model, policy_seed=saved_case['policy_seed']))
        
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim and not succ:
            observation = TASK_ENV.get_obs()
            # from IPython import embed;embed()
            
            formatted_observation = {
                "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"], # H,W,3
                "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
                "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
                
                "observation.state": observation["joint_action"]["vector"],
                "task": TASK_ENV.get_instruction(),
            }

            # print(formatted_observation["observation.images.cam_high"].shape)
            # ========= debug image =========
            # import torch, torchvision
            # from PIL import Image
            # imgs_np = []
            # for k in ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
            #     arr = formatted_observation['image'][k].squeeze(0)  # (H,W,3), uint8
            #     print(arr.max(), arr.min())
            #     imgs_np.append(arr)

            # concat = np.concatenate(imgs_np, axis=1)
            # if TASK_ENV.take_action_cnt % 25 ==0:
            #     Image.fromarray(concat).save(f'combined_{TASK_ENV.take_action_cnt}.png')
            # ========= debug image =========

            # from IPython import embed;embed()
            ret = model.infer(formatted_observation) #(TASK_ENV, model, observation)
            action, latency = ret['action'], ret['server_timing']
            if len(action.shape) == 2:
                initial_obs = False
                for act in action:
                    if initial_obs: # ensure the video is correct, but slow down simulation
                        # observation = TASK_ENV.get_obs()
                        pass
                    else:
                        initial_obs = True
                    TASK_ENV.take_action(act)
                    recorder.enabled = False
                    recorder.capture(TASK_ENV)
                    if TASK_ENV.eval_success:
                        succ = True
                        break
            else:
                TASK_ENV.take_action(action)
                recorder.enabled = False
                recorder.capture(TASK_ENV)
                if TASK_ENV.eval_success:
                    succ = True
            
            print(f"infer time {latency}")


        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        result_tag = "success" if succ else "failure"
        recorder.capture(TASK_ENV, force=True)
        record['recording'] = recorder.finish()
        record.update(status='complete', success=bool(succ), steps=int(TASK_ENV.take_action_cnt),
                      elapsed_seconds=time.time()-case_started, finished_at=time.time())
        write_json(result_file, record)
        print('CASE_COMPLETE', json.dumps(record), flush=True)
        
        old_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4"
        new_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}_{result_tag}.mp4"

        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"Video saved: {new_name}")


        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    # from test_render import Sapien_TEST
    # Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
