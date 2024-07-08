import open3d as o3d
import cv2
import os
import json
import numpy as np

# Load rgb and depth video from 

task_name = "basketball_in_hoop"
demo_id = 0
load_dir = f"outputs/{task_name}/norobot_replay/demo_{str(demo_id).zfill(4)}"
metadata = json.load(open(f"outputs/{task_name}/norobot_replay/metadata.json", 'r'))

cam_intr = {}
rgb_in, depth_in = {}, {}

width, height = 512, 512

for cam_name in ["cam_front", "cam_over_shoulder_left", "cam_over_shoulder_left"]:
    cam_intr_mat = np.array(metadata[f"cam_intr_{cam_name}"])        
    cam_intr[cam_name] = o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=cam_intr_mat[0, 0],
        fy=cam_intr_mat[1, 1],
        cx=cam_intr_mat[0, 2],
        cy=cam_intr_mat[1, 2],
    )

    rgb_in[cam_name] = cv2.VideoCapture(os.path.join(load_dir, f"{cam_name}_rgb.mp4"))
    depth_in[cam_name] = cv2.VideoCapture(os.path.join(load_dir, f"{cam_name}_depth.mkv"))
    depth_in[cam_name].set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
    depth_in[cam_name].set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

while rgb_in['cam_front'].isOpened:
    pc_o3d_frame = []
    for cam_name in ["cam_front", "cam_over_shoulder_left", "cam_over_shoulder_left"]:
        (rgb_ret, rgb_frame), (depth_ret, depth_frame) = rgb_in[cam_name].read(), depth_in[cam_name].read()
        
        rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_frame).astype(np.uint8))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_frame / 65535).astype(np.float32))
        rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=1.0)
        
        # TODO: Fix this: the extrinsics matrix changes over time
        cam_extr = np.array(metadata[f"cam_extr_{cam_name}"])
        
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, cam_intr[cam_name], cam_extr)
        pc_o3d_frame.append(point_cloud)
        
    o3d.visualization.draw(pc_o3d_frame)
    
    for pc in pc_o3d_frame:
        del pc
    del pc_o3d_frame
        