import argparse
import json
import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import quaternion
import yaml
from PIL import Image
from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.sawyer import Sawyer
from pyrep.robots.arms.ur5 import UR5
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.robots.end_effectors.robotiq85_gripper import Robotiq85Gripper

from rlbench.backend.const import TTT_FILE
from rlbench.backend.myscene import MyScene as Scene
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.task import TASKS_PATH, Task
from rlbench.demo import Demo
from rlbench.observation_config import CameraConfig, ObservationConfig
from tools.utils import *

####################################
## Consts
####################################
SUPPORTED_ROBOTS = {
    'panda': (Panda, PandaGripper, 7),
    'sawyer': (Sawyer, BaxterGripper, 7),
    'ur5': (UR5, Robotiq85Gripper, 6),
}

####################################
## Demo writer
####################################
class DemoWriter:
    def __init__(self, cfg: dict, file_path: str):
        self.cfg = cfg
        self.save_path = file_path
        self.demo_entries : List[dict] = []
    
    def capture_env_setup_data(self, scene: Scene):
        data_env_setup = {}
        data_env_setup |= self._capture_panda_data(scene)
        data_env_setup |= self._capture_object_data()
        data_env_setup |= self._capture_joint_data()
        
        self.data_env_setup = data_env_setup
    
    def _capture_panda_data(self, scene: Scene):
        init_arm_q = scene.robot.arm.get_joint_positions()
        init_gripper_q = scene.robot.gripper.get_joint_positions()
        init_q = np.concatenate([init_arm_q, init_gripper_q])
        init_robot_pos = scene.robot.arm.get_position()
        init_robot_quat = scene.robot.arm.get_quaternion()
        init_robot_quat = xyzw_to_wxyz(init_robot_quat)
        
        # robot_name = 'Panda'
        # robot = Object.get_object(robot_name)
        # init_panda_pos = robot.get_position()
        # init_panda_quat = robot.get_quaternion()
        # init_panda_quat = xyzw_to_wxyz(init_panda_quat)
        
        # assert np.allclose(init_robot_pos, init_panda_pos)
        # assert np.allclose(init_robot_quat, init_panda_quat)
        
        # t = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])  # first rotate around Y axis by 90 degree
        # t = quat_multiply_numpy(np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0, 0]), t)  # then rotate around X axis by 180 degree
        # init_robot_quat = quat_multiply_numpy(init_robot_quat, t)  # t is the transformation from RLBench to RoboVerse
        
        print(f'[DEBUG] Initial panda q: {float_array_to_str(init_q)}')
        print(f'[DEBUG] Initial panda position: {float_array_to_str(init_robot_pos)}')
        print(f'[DEBUG] Initial panda orientation: {float_array_to_str(init_robot_quat)}')
        
        panda_data = {
            'init_q': init_q,
            'init_robot_pos': init_robot_pos,
            'init_robot_quat': init_robot_quat,
            'init_target_pos': init_robot_pos + np.array([20,20,20]),    # XXX
            'init_target_quat': init_robot_quat,  # XXX
        }
        
        return panda_data
    
    def _capture_object_data(self):
        object_names = self.cfg['objects']
        object_data = {}
        for object_name in object_names:
            object = Object.get_object(object_name)
            init_object_pos = object.get_position()
            init_object_quat = object.get_quaternion()
            init_object_quat = xyzw_to_wxyz(init_object_quat)
            object_data |= {
                f'init_{object_name}_pos': init_object_pos,
                f'init_{object_name}_quat': init_object_quat,
            }
            print(f'[DEBUG] Initial {object_name} position: {float_array_to_str(init_object_pos)}')
            print(f'[DEBUG] Initial {object_name} orientation: {float_array_to_str(init_object_quat)} or {float_array_to_str(quat_to_euler(init_object_quat))}')
        
        return object_data
    
    def _capture_joint_data(self):
        joint_names = self.cfg['joints']
        joint_data = {}
        for joint_name in joint_names:
            joint = Joint(joint_name)
            init_joint_q = joint.get_joint_position()
            init_joint_q = np.array([init_joint_q])  # float to array
            joint_data |= {
                f'init_{joint_name}_q': init_joint_q,
            }
            print(f'[DEBUG] Initial {joint_name} q: {float_array_to_str(init_joint_q)}')
        
        return joint_data
    
    def _process_traj(self, demo: Demo):
        if demo is None:
            return None
        
        qs = []
        ee_acts = []
        ee_poses = []
        ee_quats = []
        for obs in demo:
            obs: Observation
            # collecting joint positions for q [L x q_len]
            q = obs.joint_positions.tolist() + obs.gripper_joint_positions.tolist()
            # collecting gripper open for ee_act [L x 1]
            ee_act = [obs.gripper_open]
            # collecting gripper pose for ee_pos [L x 3] and 
            ee_pos = obs.gripper_pose[:3] - np.array([0, 0, 0.75])
            ee_quat = xyzw_to_wxyz(obs.gripper_pose[3:])
            
            qs += [q]
            ee_acts += [ee_act]
            ee_poses += [ee_pos]
            ee_quats += [ee_quat]
        
        data_traj = {
            'q': np.array(qs),
            'ee_act': np.array(ee_acts),
            'ee_pos': np.array(ee_poses),
            'ee_quat': np.array(ee_quats),
            'success': np.ones(len(qs), dtype=bool)
        }
        return data_traj
    
    def save_demo_data(self, demo: Demo, task_name: str, episode_index: int, object_states:list=None):
        data_traj = self._process_traj(demo)
        
        ## All demo data
        demo_entry = {
            'name': f'{task_name}_traj_{episode_index}',
            'asset_name': 'default_asset',
            'episode_len': len(demo) if demo else 0,
            'env_setup': self.data_env_setup,
            'robot_traj': data_traj,
            'object_states': object_states.copy() if object_states else None,
        }
        self.demo_entries.append(demo_entry)
    
    def write(self):
        max_episode_len = max([data_demo['episode_len'] for data_demo in self.demo_entries], default=0)
        print(f'[INFO] Saving demos to {self.save_path}, max_episode_len: {max_episode_len}')
        
        full_data = {
            'max_episode_len': max_episode_len,
            'demos': {
                'franka': self.demo_entries,
                'franka_rlbench': self.demo_entries
            }
        }
        
        if self.save_path.endswith('.pkl'):
            full_data = ensure_numpy_float32(full_data)
            write_pickle(self.save_path, full_data)
        elif self.save_path.endswith('.yaml') or self.save_path.endswith('.yml'):
            full_data = ensure_numpy_as_list(full_data)
            write_yaml(self.save_path, full_data)
        elif self.save_path.endswith('.json'):
            full_data = ensure_numpy_as_list(full_data)
            write_json(self.save_path, full_data)
        else:
            raise Exception(f'[ERROR] Unsupported file format: {self.save_path}')

####################################
## Functions for collecting demos
####################################
def get_callable_when_reach_waypoint(scene: Scene):
    sensor_left = VisionSensor('cam_over_shoulder_left')
    sensor_right = VisionSensor('cam_over_shoulder_right')
    sensor_overhead = VisionSensor('cam_overhead')
    sensors = {
        'left': sensor_left, 
        'right': sensor_right, 
        'overhead': sensor_overhead,
    }
    snapshot_save_dir = mkdir(os.path.join(save_dir, 'snapshots'))
    
    def func(waypoint_index: int):
        ## Capture robot joint positions
        arm_q = scene.robot.arm.get_joint_positions()
        gripper_q = scene.robot.gripper.get_joint_positions()
        q = np.concatenate([arm_q, gripper_q])
        print(f'[INFO] Reached waypoint {waypoint_index}, q: {float_array_to_str(q)}')
        write_yaml(os.path.join(snapshot_save_dir, f'waypoint{waypoint_index}.yaml'), {'q': q.tolist()})

        ## Capture snapshot
        for sensor_name, sensor in sensors.items():
            sensor.handle_explicitly()
            rgb = sensor.capture_rgb()
            rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
            rgb = Image.fromarray(rgb)
            rgb.save(os.path.join(snapshot_save_dir, f'waypoint{waypoint_index}_{sensor_name}.png'))
    return func

def get_callable_each_step(cfg: dict, object_states: list):
    object_names = cfg['objects']
    joint_names = cfg['joints']
    
    def record_objects(observation: Observation):
        object_state_this_step = {}
        for object_name in object_names:
            object = Object.get_object(object_name)
            object_pos = object.get_position()
            object_quat = object.get_quaternion()
            # object_vel = object.get_velocity()
            object_quat = xyzw_to_wxyz(object_quat)
            object_state_this_step |= {
                f'{object_name}_pos': object_pos,
                f'{object_name}_quat': object_quat,
                # f'{object_name}_vel': object_vel,
            }
            print(f'[DEBUG] {object_name} position: {float_array_to_str(object_pos)}')
            print(f'[DEBUG] {object_name} orientation: {float_array_to_str(object_quat)} or {float_array_to_str(quat_to_euler(object_quat))}')
        for joint_name in joint_names:
            joint = Joint(joint_name)
            joint_q = joint.get_joint_position()
            joint_v = joint.get_joint_velocity()
            object_state_this_step |= {
                f'{joint_name}_q': joint_q,
                f'{joint_name}_v': joint_v,
            }
            
            print(f'[DEBUG] {joint_name} q: {joint_q}')
        object_states.append(object_state_this_step)
    
    return record_objects

class DemoGetter:
    def __init__(self, args, cfg: dict, writer: DemoWriter):
        self.args = args
        self.cfg = cfg
        self.writer = writer
        self.object_states = [] if self.args.record_object_states else None
        
        self._launch_sim()
        self._setup_robot()
        self._create_scene()
    
    def _launch_sim(self):
        DIR_PATH = os.path.dirname(os.path.abspath(__file__))
        self.sim = PyRep()
        ttt_file = os.path.join(DIR_PATH, '..', 'rlbench', TTT_FILE)
        self.sim.launch(ttt_file, headless=self.args.headless)
        self.sim.step_ui()
        self.sim.set_simulation_timestep(1/60)  # Control frequency 60Hz
        self.sim.step_ui()
        self.sim.start()
        
    def _setup_robot(self):
        DIR_PATH = os.path.dirname(os.path.abspath(__file__))
        arm_class, gripper_class, _ = SUPPORTED_ROBOTS[self.args.robot]
        # Default robot is Panda
        if self.args.robot != 'panda':
            panda_arm = Panda()
            panda_pos = panda_arm.get_position()
            panda_arm.remove()
            arm_path = os.path.join(DIR_PATH, '..', 'rlbench', 'robot_ttms', f'{self.args.robot}.ttm')
            self.sim.import_model(arm_path)
            arm, gripper = arm_class(), gripper_class()
            arm.set_position(panda_pos)
        else:
            arm, gripper = arm_class(), gripper_class()
        self.robot = Robot(arm, gripper)
        
    def _create_scene(self):
        obs_config = ObservationConfig()
        obs_config.set_all(False)  # Comment this to slow down the simulation
        if self.args.snapshot:
            obs_config.overhead_camera = CameraConfig(image_size=(256, 256))
            obs_config.left_shoulder_camera = CameraConfig(image_size=(256, 256))
            obs_config.right_shoulder_camera = CameraConfig(image_size=(256, 256))
        obs_config.joint_positions = True
        obs_config.gripper_joint_positions = True
        obs_config.gripper_open = True
        obs_config.gripper_pose = True
        self.scene = Scene(self.sim, self.robot, obs_config)
    
    def load_task(self, task_name):
        def task_name_to_task_class(name):
            import importlib
            class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
            mod = importlib.import_module("rlbench.tasks.%s" % name)
            mod = importlib.reload(mod)
            task_class = getattr(mod, class_name)
            return task_class
        
        task_class = task_name_to_task_class(task_name)
        self.task : Task = task_class(self.sim, self.robot)
        self.scene.unload()
        self.scene.load(self.task)
        self.scene.init_task()
    
    def _try_get_demo(self, variation_index=0, only_setup=False):
        self.scene.reset()
        self.scene.init_episode(variation_index, max_attempts=10)
        self.writer.capture_env_setup_data(self.scene)
        if only_setup:
            demo = None
        else:
            demo = self.scene.get_demo(
                record=True,
                callable_each_step=get_callable_each_step(self.cfg, self.object_states) if self.args.record_object_states else None,
                callable_when_reach_waypoint=get_callable_when_reach_waypoint(self.scene) if self.args.snapshot else None
            )
        return demo

    def get_demo(self, episode_index, variation_index=0, attempts=10, raise_error=True, only_setup=False):
        error = None
        while attempts > 0:
            try:
                demo = self._try_get_demo(variation_index, only_setup=only_setup)
            except Exception as e:
                attempts -= 1
                print(f'[DEBUG] Failed to get task {self.scene.task.get_name()} (episode: {episode_index}). Rest attempts {attempts}. Retrying...')
                error = e  # record the error
                # raise e  # for debug
                continue
            break
        
        if attempts > 0:
            print(f'[INFO] Successfully got task {self.scene.task.get_name()} (episode: {episode_index}), length: {len(demo) if demo else 0}')
            return demo
        else:
            print(f'[ERROR] Failed to get task {self.scene.task.get_name()} (episode: {episode_index})')
            if raise_error:
                raise error
            else:
                print(f'[ERROR] Error: {error}')
    
    def _get_one_demo(self, episode_index: int, only_setup=False):
        # np.random.seed(0)  # Make all the demo the same
        task_name = self.task.get_name()
        demo = self.get_demo(episode_index, variation_index=0, only_setup=only_setup)  # Always get the first variation
        self.writer.save_demo_data(demo, task_name, episode_index, object_states=self.object_states)
        if self.args.record_object_states:
            self.object_states.clear()
    
    def get_demos(self, num_episodes: int, only_setup=False):
        start_time = time.time()
        
        for episode_index in range(num_episodes):
            self._get_one_demo(episode_index, only_setup=only_setup)
            
            elapsed_time = time.time() - start_time
            average_time = elapsed_time / (episode_index + 1)
            remaining_time = (num_episodes - episode_index - 1) * average_time
            print(f"Episode {episode_index+1}/{num_episodes} - Elapsed Time: {elapsed_time:.2f}s ({average_time:.2f}s per episode) - Estimated Remaining Time: {remaining_time:.2f}s")

        self.writer.write()

####################################
## Main
####################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--headless", action='store_true')
    parser.add_argument("--episode_num", type=int, default=1)
    parser.add_argument("--conf", "-c", default="data/cfg/rlbench_objects.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot", default='panda', choices=['panda', 'sawyer', 'ur5'])
    parser.add_argument("--snapshot", action='store_true')
    parser.add_argument("--only_setup", action='store_true')
    parser.add_argument("--record_object_states", action='store_true')
    args = parser.parse_args()
    cfg = read_yaml(args.conf).get(args.task, {'objects': [], 'joints': []})
    cfg['objects'] += ['diningTable']

    ## Task
    python_file = os.path.join(TASKS_PATH, f'{args.task}.py')
    if not os.path.isfile(python_file):
        raise RuntimeError('Could not find the task file: %s' % python_file)
    
    ## Run task
    np.random.seed(args.seed)
    TaskName = ''.join([x.capitalize() for x in args.task.split('_')])
    # save_dir = mkdir(f'/home/fs/cod/UniRobo/IsaacSimInfra/omniisaacgymenvs/data/demos/rlbench/{TaskName}-v{args.episode_num}')
    save_dir = mkdir(f'/home/fs/cod/UniRobo/IsaacSimInfra/omniisaacgymenvs/data/demos/rlbench/{TaskName}-v0')
    if args.record_object_states:
        pkl_filename = 'trajectory-unified_with_object_states.pkl'
    elif args.only_setup:
        pkl_filename = 'trajectory-unified_no_demo.pkl'
    else:
        pkl_filename = 'trajectory-unified.pkl'
    writer = DemoWriter(cfg, os.path.join(save_dir, pkl_filename))
    getter = DemoGetter(args, cfg, writer)
    getter.load_task(args.task)
    getter.get_demos(args.episode_num, only_setup=args.only_setup)