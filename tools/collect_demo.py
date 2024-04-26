from rlbench.backend.task import Task
from rlbench.backend.scene import DemoError
from rlbench.observation_config import ObservationConfig
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from rlbench.backend.const import TTT_FILE
from rlbench.backend.scene import Scene
from rlbench.backend.utils import task_file_to_task_class
from rlbench.backend.task import TASKS_PATH
from rlbench.backend.robot import Robot
from rlbench.demo import Demo
from typing import List, Tuple
import numpy as np
import os
import json
import yaml
import pickle
import argparse

####################################
## Utils
####################################

def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path

def write_yaml(path, data):
    with open(path, 'w+') as f:
        yaml.dump(data, f, default_flow_style=False)

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def write_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)

def data_numpy_to_list(data: dict) -> dict:
    '''
    Convert numpy arrays in nested dict or list to list
    data: nested dict or list, containing numpy arrays
    '''
    for k, v in data.items():
        if type(v) == np.ndarray:
            data[k] = v.tolist()
        elif type(v) == dict:
            data[k] = data_numpy_to_list(v)
        elif type(v) == list:
            data[k] = [data_numpy_to_list(e) for e in v]
    return data

def data_numpy_float32(data: dict) -> dict:
    '''
    Make sure all numpy arrays in nested dict or list have dtype float32
    data: nested dict or list, containing numpy arrays
    '''
    for k, v in data.items():
        if type(v) == np.ndarray:
            data[k] = v.astype(np.float32)
        elif type(v) == dict:
            data[k] = data_numpy_float32(v)
        elif type(v) == list:
            data[k] = [data_numpy_float32(e) for e in v]
    return data

####################################
## Functions for collecting demos
####################################
def get_env_setup_data(scene: Scene):
    init_arm_q = scene.robot.arm.get_joint_positions()
    init_gripper_q = scene.robot.gripper.get_joint_positions()
    init_q = np.concatenate([init_arm_q, init_gripper_q])
    init_robot_pos = scene.robot.arm.get_position()
    init_robot_quat = scene.robot.arm.get_quaternion()
    
    data_env_setup = {
        'init_q': init_q,
        'init_robot_pos': init_robot_pos,
        'init_robot_quat': init_robot_quat,
        'init_cube_pos': init_robot_pos,    # XXX
        'init_cube_quat': init_robot_quat,  # XXX
        'init_target_pos': init_robot_pos + np.array([20,20,20]),    # XXX
        'init_target_quat': init_robot_quat,  # XXX
    }
    
    return data_env_setup

def get_demo_one_try(scene: Scene, variation_index) -> Tuple[Demo, dict]:
    scene.reset()
    desc = scene.init_episode(variation_index, max_attempts=10)
    data_env_setup = get_env_setup_data(scene)
    demo = scene.get_demo(record=True)
    return demo, data_env_setup

def get_demo(scene: Scene, variation_index) -> Tuple[Demo, dict]:
    attempts = 10
    while attempts > 0:
        try:
            demo, data_env_setup = get_demo_one_try(scene, variation_index)
        except Exception as e:
            attempts -= 1
            print(f'[DEBUG] Failed to get task {scene.task.get_name()} (variation: {variation_index}). Rest attempts {attempts}. Retrying...')
            continue
        break
    
    if attempts > 0:
        print(f'[INFO] Successfully got task {scene.task.get_name()} (variation: {variation_index})')
        return demo, data_env_setup
    else:
        raise Exception(f'[ERROR] Failed to get task {scene.task.get_name()} (variation: {variation_index})')

def get_demos(task: Task, scene: Scene, variation_num: int) -> List[Demo]:
    task_name = task.get_name()
    save_dir = mkdir(os.path.join('outputs', f'{task_name}'))
    
    scene.load(task)
    scene.init_task()
    
    data_demos = []
    for variation_index in range(variation_num):
        demo, data_env_setup = get_demo(scene, variation_index)
        data_demo = get_demo_data(demo, data_env_setup, task_name, variation_index)    
        data_demos.append(data_demo)
    
    save_path = os.path.join(save_dir, f'{task_name}.pkl')
    save_demos(data_demos, save_path)

def get_demo_data(demo: Demo, data_env_setup: dict, task_name: str, variation_index: int):
    data = {}
    
    ## Export all keys
    # for obs in demo:
    #     for key in dir(obs):
    #         if key.startswith("__"):
    #             continue

    #         attr = getattr(obs, key)
    #         if callable(attr):
    #             continue
    #         if type(attr) == np.ndarray:
    #             attr = attr.tolist()
    #         data[key] = attr

    ## Process trajectory
    qs = []
    ee_acts = []
    for obs in demo:
        # collecting joint positions for q [L x q_len]
        q = obs.joint_positions.tolist() + obs.gripper_joint_positions.tolist()
        # collecting gripper open for ee_act [L x 1]
        ee_act = [obs.gripper_open]
        
        qs.append(q)
        ee_acts.append(ee_act)
    
    data_traj = {
        'q': np.array(qs),
        'ee_act': np.array(ee_acts),
        'success': np.ones(len(qs), dtype=bool)
    }
    
    ## All demo data
    data_demo = {
        'name': f'{task_name}_traj_{variation_index}',
        'asset_name': 'default_asset',
        'episode_len': len(qs),
        'env_setup': data_env_setup,
        'robot_traj': data_traj
    }
    
    return data_demo

def save_demos(data_demos: List[dict], save_path: str):
    max_episode_len = max([data_demo['episode_len'] for data_demo in data_demos])
    
    full_data = {
        'max_episode_len': max_episode_len,
        'demos': {
            'franka': data_demos
        }
    }
    
    if save_path.endswith('.pkl'):
        full_data = data_numpy_float32(full_data)
        write_pickle(save_path, full_data)
    elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
        full_data = data_numpy_to_list(full_data)
        write_yaml(save_path, full_data)
    elif save_path.endswith('.json'):
        full_data = data_numpy_to_list(full_data)
        write_json(save_path, full_data)
    else:
        raise Exception(f'[ERROR] Unsupported file format: {save_path}')

####################################
## Main
####################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="The task file to test.")
    parser.add_argument("--headless", action='store_true')
    parser.add_argument("--variation_num", type=int, default=1)
    args = parser.parse_args()

    ## Task
    python_file = os.path.join(TASKS_PATH, args.task)
    if not os.path.isfile(python_file):
        raise RuntimeError('Could not find the task file: %s' % python_file)
    task_class = task_file_to_task_class(args.task)
    
    ## CoppeliaSim
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    sim = PyRep()
    ttt_file = os.path.join(
        DIR_PATH, '..', 'rlbench', TTT_FILE)
    sim.launch(ttt_file, headless=args.headless)
    sim.step_ui()
    sim.set_simulation_timestep(0.005)
    sim.step_ui()
    sim.start()
    
    ## Setup
    robot = Robot(Panda(), PandaGripper())
    active_task = task_class(sim, robot)
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    scene = Scene(sim, robot, obs_config)
    
    ## Run task
    demos = get_demos(active_task, scene, args.variation_num)
    