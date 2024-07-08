import open3d as o3d
import cv2
import os
import json
import numpy as np

# Load rgb and depth video from 

task_name = "basketball_in_hoop"
demo_id = 0
load_dir = f"outputs/{task_name}/norobot_replay/{str(demo_id).zfill(4)}"
metadata = json.load(open(f"outputs/{task_name}/norobot_replay/metadata.json", 'r'))

cam_name = "front" # "over_shoulder_left" "over_shoulder_left"

cam_intr_mat = np.array(metadata[f"cam_intr_{cam_name}"])
cam_extr_mat = np.array(metadata[f"cam_extr_{cam_name}"])
rgb_data = cv2.VideoCapture(os.path.join(load_dir, f"{cam_name}_rgb.mp4"))
depth_data = cv2.VideoCapture(os.path.join(load_dir, f"{cam_name}_depth.mkv"))

while rgb_data.isOpened:
    (rgb_ret, rgb_frame), (depth_ret, depth_frame) = rgb_data.read(), depth_data.read()
    width, height = rgb_frame.shape
    
    rgbd_frame = o3d.geometry.create_rgbd_image_from_color_and_depth(rgb_frame, depth_frame, depth_scale=1.0)
    intr = o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=cam_intr_mat[0, 0],
        fy=cam_intr_mat[1, 1],
        cx=cam_intr_mat[0, 2],
        cy=cam_intr_mat[1, 2],
    )
    point_cloud = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_frame, intr, cam_extr_mat)
    o3d.visualization.draw([point_cloud])
    