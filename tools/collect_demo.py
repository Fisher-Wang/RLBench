from rlbench.backend.task import Task
from rlbench.backend.scene import DemoError
from rlbench.observation_config import ObservationConfig
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.robots.arms.sawyer import Sawyer
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.robots.arms.ur5 import UR5
from pyrep.robots.end_effectors.robotiq85_gripper import Robotiq85Gripper
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.const import TTT_FILE
from rlbench.backend.scene import Scene
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
import quaternion

####################################
## Consts
####################################
SUPPORTED_ROBOTS = {
    'panda': (Panda, PandaGripper, 7),
    'sawyer': (Sawyer, BaxterGripper, 7),
    'ur5': (UR5, Robotiq85Gripper, 6),
}

####################################
## Utils
####################################

def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path

def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def write_yaml(path, data):
    with open(path, 'w+') as f:
        yaml.dump(data, f, default_flow_style=False)

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def write_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)

def ensure_numpy_as_list(data: dict) -> dict:
    '''
    Convert numpy arrays in nested dict or list to list
    data: nested dict or list, containing numpy arrays
    '''
    for k, v in data.items():
        if type(v) == np.ndarray:
            data[k] = v.tolist()
        elif type(v) == dict:
            data[k] = ensure_numpy_as_list(v)
        elif type(v) == list:
            data[k] = [ensure_numpy_as_list(e) for e in v]
    return data

def ensure_numpy_float32(data: dict) -> dict:
    '''
    Make sure all numpy arrays in nested dict or list have dtype float32
    data: nested dict or list, containing numpy arrays
    '''
    for k, v in data.items():
        if type(v) == np.ndarray:
            data[k] = v.astype(np.float32)
        elif type(v) == dict:
            data[k] = ensure_numpy_float32(v)
        elif type(v) == list:
            data[k] = [ensure_numpy_float32(e) for e in v]
    return data

def xyzw_to_wxyz(quat: np.ndarray):
    '''
    Convert a quaternion array from [x, y, z, w] to [w, x, y, z]
    '''
    assert quat.shape[-1] == 4
    return np.stack([quat[..., 3], quat[..., 0], quat[..., 1], quat[..., 2]], axis=-1)

def float_array_to_str(arr: np.ndarray):
    '''
    Convert a list or 1D array of float to a string with 2 decimal places
    '''
    assert type(arr) == list or (type(arr) == np.ndarray and len(arr.shape) == 1)
    return '[' + ', '.join([f'{e:.4f}' for e in arr]) + ']'

def quat_multiply_numpy(q1, q2):
    """
    Multiply two quaternions or arrays of quaternions.

    Args:
        q1 (ndarray): The first quaternion or array of quaternions of shape (..., 4).
        q2 (ndarray): The second quaternion or array of quaternions of shape (..., 4).

    Returns:
        ndarray: The result of quaternion multiplication, with shape (..., 4).

    Notes:
        The quaternions should be in the format [w, x, y, z].
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2],
        axis=-1
    )

def quat_to_euler(q):
    """
    Converts quaternion (w in first place) to euler (roll, pitch, yaw)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)

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
    
    def save_demo_data(self, demo: Demo, task_name: str, episode_index: int):
        data = {}

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
        demo_entry = {
            'name': f'{task_name}_traj_{episode_index}',
            'asset_name': 'default_asset',
            'episode_len': len(qs),
            'env_setup': self.data_env_setup,
            'robot_traj': data_traj
        }
    
        self.demo_entries.append(demo_entry)
    
    def write(self):
        max_episode_len = max([data_demo['episode_len'] for data_demo in self.demo_entries])
        print(f'[INFO] Saving demos to {self.save_path}, max_episode_len: {max_episode_len}')
        
        full_data = {
            'max_episode_len': max_episode_len,
            'demos': {
                'franka': self.demo_entries
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
class DemoGetter:
    def __init__(self, args, writer: DemoWriter):
        self.args = args
        self.writer = writer
        
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
        obs_config.joint_positions = True
        obs_config.gripper_joint_positions = True
        obs_config.gripper_open = True
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
    
    def _try_get_demo(self, variation_index=0):
        self.scene.reset()
        desc = self.scene.init_episode(variation_index, max_attempts=10)
        self.writer.capture_env_setup_data(self.scene)
        demo = self.scene.get_demo(record=True)
        return demo

    def get_demo(self, episode_index, variation_index=0, attempts=10, raise_error=True):
        error = None
        while attempts > 0:
            try:
                demo = self._try_get_demo(variation_index)
            except Exception as e:
                attempts -= 1
                print(f'[DEBUG] Failed to get task {self.scene.task.get_name()} (episode: {episode_index}). Rest attempts {attempts}. Retrying...')
                error = e  # record the error
                continue
            break
        
        if attempts > 0:
            print(f'[INFO] Successfully got task {self.scene.task.get_name()} (episode: {episode_index}), length: {len(demo)}')
            return demo
        else:
            print(f'[ERROR] Failed to get task {self.scene.task.get_name()} (episode: {episode_index})')
            if raise_error:
                raise error
            else:
                print(f'[ERROR] Error: {error}')

    def get_demos(self, num_episodes: int):
        task_name = self.task.get_name()
        
        for episode_index in range(num_episodes):
            demo = self.get_demo(episode_index, variation_index=0)  # Always get the first variation
            self.writer.save_demo_data(demo, task_name, episode_index)    
        
        self.writer.write()

####################################
## Main
####################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="The task name to test.")
    parser.add_argument("--headless", action='store_true')
    parser.add_argument("--episode_num", type=int, default=1)
    parser.add_argument("--conf", "-c", default="data/cfg/rlbench_objects.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot", default='panda', choices=['panda', 'sawyer', 'ur5'])
    args = parser.parse_args()
    cfg = read_yaml(args.conf).get(args.task, {'objects': [], 'joints': []})

    ## Task
    python_file = os.path.join(TASKS_PATH, f'{args.task}.py')
    if not os.path.isfile(python_file):
        raise RuntimeError('Could not find the task file: %s' % python_file)
    
    ## Run task
    np.random.seed(args.seed)
    save_dir = mkdir(os.path.join('outputs', args.task))
    writer = DemoWriter(cfg, os.path.join(save_dir, f'{args.task}.pkl'))
    getter = DemoGetter(args, writer)
    getter.load_task(args.task)
    getter.get_demos(args.episode_num)