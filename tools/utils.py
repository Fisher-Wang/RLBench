import json
import os
import pickle

import numpy as np
import yaml


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

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

def wxyz_to_xyzw(quat: np.ndarray):
    """
    Convert a quaternion array from [w, x, y, z] to [x, y, z, w]
    """
    assert quat.shape[-1] == 4
    return np.stack([quat[..., 1], quat[..., 2], quat[..., 3], quat[..., 0]], axis=-1)

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