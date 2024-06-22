import argparse
import os
import pickle
import time

import numpy as np
import yaml
from pyrep import PyRep
from pyrep.backend import simConst
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

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
parser.add_argument('--gravity', type=bool, default=False)
args = parser.parse_args()
cfg = read_yaml('data/cfg/rlbench_objects.yaml')[args.task]

## Launch PyRep
sim = PyRep()
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ttt_file = os.path.join(DIR_PATH, '../rlbench', 'task_design_wo_franka.ttt')
sim.launch(ttt_file, headless=False)
if not args.gravity:
    sim.script_call('setGravity@PyRep', simConst.sim_scripttype_addonscript, floats=[0, 0, 0])
sim.set_simulation_timestep(1/60)  # Control frequency 60Hz

## Load task
ttm_file = os.path.join(DIR_PATH, '../rlbench/task_ttms', f'{args.task}.ttm')
base_object = sim.import_model(ttm_file)

## Load demo
TaskName = ''.join([x.capitalize() for x in args.task.split('_')])
demo_path = f'/home/fs/cod/UniRobo/IsaacSimInfra/omniisaacgymenvs/data/demos/rlbench/{TaskName}-v0/trajectory-unified_with_object_states.pkl'
demo_data = read_pickle(demo_path)
demo = demo_data['demos']['franka'][0]
env_setup = demo['env_setup']
object_states = demo['object_states']

## Environment setup
for object_name in cfg['objects']:
    pos = env_setup[f'init_{object_name}_pos']
    quat = env_setup[f'init_{object_name}_quat']
    object = Object.get_object(object_name)
    object.set_position(pos)
    object.set_orientation(quat_to_euler(quat))
    print('Initially set object', object_name, 'to', float_array_to_str(pos), float_array_to_str(quat_to_euler(quat)))

for joint_name in cfg['joints']:
    q = env_setup[f'init_{joint_name}_q']
    joint = Joint(joint_name)
    joint.set_joint_position(q)
    print('Initially set joint', joint_name, 'to', float_array_to_str(q))

## Replay
sim.start()
sim.step()

for i, object_state in enumerate(object_states):
    for object_name in cfg['objects']:
        object = Object.get_object(object_name)
        object_pos = object_state[f'{object_name}_pos']
        object_quat = object_state[f'{object_name}_quat']
        object.set_position(object_pos)
        object.set_quaternion(wxyz_to_xyzw(object_quat))
        
        print('[DEBUG] Set object', object_name, 'to', float_array_to_str(object_pos), float_array_to_str(quat_to_euler(object_quat)))
    
    for joint_name in cfg['joints']:
        joint = Joint(joint_name)
        joint.set_joint_position(object_state[f'{joint_name}_q'])
        # joint.set_joint_target_velocity(object_state[f'{joint_name}_v'])
        
        print('[DEBUG] Set joint', joint_name, 'to', object_state[f'{joint_name}_q'])
    
    sim.step()
    
sim.stop()
sim.shutdown()