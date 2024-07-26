import argparse
import os
import pickle
import time

import numpy as np
import yaml
from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

from rlbench.backend.const import TTT_FILE
from rlbench.backend.robot import Robot
from tools.collect_demo import float_array_to_str, quat_to_euler, read_yaml


####################################
## Utils
####################################
def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def wxyz_to_xyzw(quat: np.ndarray):
    '''
    Convert a quaternion array from [w, x, y, z] to [x, y, z, w]
    '''
    assert quat.shape[-1] == 4
    return np.stack([quat[..., 1], quat[..., 2], quat[..., 3], quat[..., 0]], axis=-1)

####################################
## Main
####################################

## Parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True)
args = parser.parse_args()
cfg = read_yaml('data/cfg/rlbench_objects.yaml')[args.task]

## Launch PyRep
sim = PyRep()
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ttt_file = os.path.join(DIR_PATH, '../rlbench', TTT_FILE)
sim.launch(ttt_file, headless=False)
sim.set_simulation_timestep(1/60)  # Control frequency 60Hz

## Load task
ttm_file = os.path.join(DIR_PATH, '../rlbench/task_ttms', f'{args.task}.ttm')
base_object = sim.import_model(ttm_file)

## Load demo
task_pascalcase = ''.join([word.title() for word in args.task.split('_')])
demo_path = os.path.join(DIR_PATH, "../trajectories", f'{task_pascalcase}-v0', "trajectory-unified_with_object_states.pkl")
demo_data = read_pickle(demo_path)
demo = demo_data['demos']['franka'][0]
env_setup = demo['env_setup']
traj = demo['robot_traj']

## Environment setup
arm, gripper = Panda(), PandaGripper()
robot = Robot(arm, gripper)
arm.set_joint_positions(env_setup['init_q'][:7])
gripper.set_joint_positions(env_setup['init_q'][7:])
gripper.set_control_loop_enabled(True)

for object_name in cfg['objects']:
    pos = env_setup[f'init_{object_name}_pos']
    quat = env_setup[f'init_{object_name}_quat']
    object = Object.get_object(object_name)
    object.set_position(pos)
    object.set_orientation(quat_to_euler(quat))
    print('Set object', object_name, 'to', float_array_to_str(pos), float_array_to_str(quat_to_euler(quat)))

for joint_name in cfg['joints']:
    q = env_setup[f'init_{joint_name}_q']
    joint = Joint(joint_name)
    joint.set_joint_position(q)
    print('Set joint', joint_name, 'to', float_array_to_str(q))

## Replay demo
sim.start()
sim.step()

L = traj['q'].shape[0]
for i in range(L):
    q = traj['q'][i]
    
    ## Borrow from JointPositionActionMode
    arm.set_joint_target_positions(q[:7])
    gripper.set_joint_target_positions(q[7:])
    sim.step()
    arm.set_joint_target_positions(arm.get_joint_positions())
    gripper.set_joint_target_positions(gripper.get_joint_positions())
    
sim.stop()
sim.shutdown()