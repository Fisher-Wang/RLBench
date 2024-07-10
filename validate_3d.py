import open3d as o3d
import cv2
import os
import json
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import pickle as pkl

def plot_point_cloud(pts, **kwargs):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        **kwargs
    )

task_name = "phone_on_base"
traj_path = "trajectories/PhoneOnBase-v0/trajectory-unified_with_object_states.pkl"
demo_id = 0
load_dir = f"outputs/{task_name}/norobot_replay/demo_{str(demo_id).zfill(4)}"
metadata = json.load(open(f"outputs/{task_name}/norobot_replay/metadata.json", 'r'))

traj_data = pkl.load(open(traj_path, 'rb'))

cam_intr, cam_extr = {}, {}
rgb_in, depth_in = {}, {}
depth_min, depth_max = {}, {}

width, height = 512, 512

for cam_name in ["cam_front", "cam_over_shoulder_left", "cam_over_shoulder_left", "cam_overhead"]:
    cam_intr_mat = np.array(metadata[f"cam_intr_{cam_name}"])        
    cam_intr[cam_name] = o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=cam_intr_mat[0, 0],
        fy=cam_intr_mat[1, 1],
        cx=cam_intr_mat[0, 2],
        cy=cam_intr_mat[1, 2],
    )
    
    # Fixed camera
    cam_extr[cam_name] = np.linalg.inv(np.array(metadata[f"cam_extr_{cam_name}"]))
    
    rgb_in[cam_name] = cv2.VideoCapture(os.path.join(load_dir, f"{cam_name}_rgb.mp4"))
    depth_in[cam_name] = cv2.VideoCapture(os.path.join(load_dir, f"{cam_name}_depth.mkv"))
    depth_in[cam_name].set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
    depth_in[cam_name].set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
    
    depth_min[cam_name] = metadata[f"depth_min_{cam_name}"]
    depth_max[cam_name] = metadata[f"depth_max_{cam_name}"]

while rgb_in['cam_front'].isOpened:
    to_plot = []
    for cam_name in ["cam_front", "cam_over_shoulder_left", "cam_over_shoulder_left"]:
        (rgb_ret, rgb_frame), (depth_ret, depth_frame) = rgb_in[cam_name].read(), depth_in[cam_name].read()
        
        rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_frame[:, :, ::-1]).astype(np.uint8))
        depth_o3d = o3d.geometry.Image((np.ascontiguousarray(depth_frame / 65535 * (depth_max[cam_name] - depth_min[cam_name])).astype(np.float32) + depth_min[cam_name]))
        rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False)
        
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, cam_intr[cam_name], cam_extr[cam_name])
        
        to_plot.append(plot_point_cloud(np.array(point_cloud.points)[::4], marker={ 'color': np.array(point_cloud.colors)[::4], 'size': 3 }, name=cam_name))
    
    
    tcp_pos = np.array(traj_data['demos']['franka'][demo_id]['robot_traj']['ee_pos'])
    tcp_quat = np.array(traj_data['demos']['franka'][demo_id]['robot_traj']['ee_quat'])
    tcp_quat = R.from_quat(tcp_quat).as_matrix()
    tcp_act = np.array(traj_data['demos']['franka'][demo_id]['robot_traj']['ee_act'])
    
    tcp_color = np.array(tcp_act < 0.01, dtype=float)[:, None].repeat(3, 1)
    
    tcp_pos[:, 2] += 0.75 # Table
    to_plot.append(plot_point_cloud(tcp_pos, marker={ 'color': tcp_color, 'size': 5}, name='tcp'))
    
    go.Figure(to_plot).show()
    input("[ENTER] Next frame")
    
    for pc in to_plot:
        del pc
    del to_plot
        