import argparse
import os
import time
from os.path import join as pjoin

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from pyrep import PyRep
from pyrep.backend import simConst
from pyrep.const import RenderMode
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from utils import (
    ensure_numpy_as_list,
    mkdir,
    read_pickle,
    read_yaml,
    write_json,
    write_pickle,
    wxyz_to_xyzw,
)

from rlbench.noise_model import Identity, NoiseModel
from tools.collect_demo import float_array_to_str, quat_to_euler, read_yaml

####################################
## Consts
####################################
CAMERA_NAMES = [
    "cam_over_shoulder_left",
    "cam_over_shoulder_right",
    "cam_overhead",
    "cam_front",
]


####################################
## Utils
####################################
def write_uint16_monochrome_mkv(
    frames: list[np.ndarray], save_path: str, fps: int = 30
):
    ## ref: https://stackoverflow.com/a/77028617
    assert len(frames[0].shape) == 2
    assert save_path.endswith(".mkv")
    h, w = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(
        filename=save_path,
        apiPreference=cv2.CAP_FFMPEG,
        fourcc=cv2.VideoWriter_fourcc(*"FFV1"),
        fps=fps,
        frameSize=(w, h),
        params=[
            cv2.VIDEOWRITER_PROP_DEPTH,
            cv2.CV_16U,
            cv2.VIDEOWRITER_PROP_IS_COLOR,
            0,  # false
        ],
    )
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def write_uint8_mp4(frames: list[np.ndarray], save_path: str, fps: int = 30):
    assert save_path.endswith(".mp4")
    h, w = frames[0].shape[:2]
    isColor = len(frames[0].shape) == 3 and frames[0].shape[2] > 1
    video = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor
    )
    for frame in frames:
        video.write(frame)
    video.release()


def write_video(frames: list[np.ndarray], save_path: str, fps: int = 30):
    if frames[0].dtype == np.uint8:
        write_uint8_mp4(frames, save_path, fps)
    elif frames[0].dtype == np.uint16:
        if len(frames[0].shape) == 2:
            write_uint16_monochrome_mkv(frames, save_path, fps)
        else:
            raise Exception("You can only write uint16 video with monocolor")
    else:
        raise Exception(f"Unsupported dtype: {frames[0].dtype}")


def write_pointcloud(xyz: np.ndarray, save_path: str):
    """
    xyz: ndarray of shape (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(save_path, pcd)


####################################
## Functions
####################################
def get_mask(sensor: VisionSensor, mask_fn):
    mask = None
    if sensor is not None:
        sensor.handle_explicitly()
        mask = mask_fn(sensor.capture_rgb())
    return mask


def get_rgb_depth(
    sensor: VisionSensor,
    get_rgb: bool,
    get_depth: bool,
    get_pcd: bool,
    rgb_noise: NoiseModel,
    depth_noise: NoiseModel,
    depth_in_meters: bool,
):
    """
    depth: Returned values are in the range of 0-1 (0=closest to sensor (i.e. close clipping plane), 1=farthest from sensor (i.e. far clipping plane)). If depth_in_meters was specified, then individual values are expressed as distances in meters.
    """
    rgb = depth = pcd = None
    if sensor is not None and (get_rgb or get_depth):
        sensor.handle_explicitly()
        if get_rgb:
            rgb = sensor.capture_rgb()
            if rgb_noise is not None:
                rgb = rgb_noise.apply(rgb)
            rgb = np.clip((rgb * 255.0).astype(np.uint8), 0, 255)
            rgb = rgb[:, :, ::-1]  # BGR -> RGB
        if get_depth or get_pcd:
            depth = sensor.capture_depth(depth_in_meters)
            if depth_noise is not None:
                depth = depth_noise.apply(depth)
        if get_pcd:
            depth_m = depth
            if not depth_in_meters:
                near = sensor.get_near_clipping_plane()
                far = sensor.get_far_clipping_plane()
                depth_m = near + depth * (far - near)
            pcd = sensor.pointcloud_from_depth(depth_m)
            if not get_depth:
                depth = None
    return rgb, depth, pcd


def init_cameras():
    cams = [VisionSensor(name) for name in CAMERA_NAMES]
    ## Set camera properties same as RLBench
    for cam in cams:
        cam.set_explicit_handling(1)
        cam.set_resolution((512, 512))  # change to your desired resolution
        cam.set_render_mode(RenderMode.OPENGL3)
    return cams


def get_observations(cams: list[VisionSensor]):
    rst = {}
    for cam in cams:
        rgb, depth, pcd = get_rgb_depth(
            cam,
            get_rgb=True,
            get_depth=True,
            get_pcd=args.get_pcd,
            rgb_noise=Identity(),
            depth_noise=Identity(),
            depth_in_meters=False,
        )
        rst[cam.get_name()] = {"rgb": rgb, "depth": depth, "pcd": pcd}
    return rst


def save_observations(
    observations: list[dict[str, dict]],
    save_dir,
    cams: list[VisionSensor],
    save_frame=False,
):
    data = {
        f"{cam.get_name()}_{vis_type}": []
        for cam in cams
        for vis_type in ["rgb", "depth", "pcd"]
    }

    frame_save_dir = mkdir(pjoin(save_dir, "frames"))
    for frame_idx, obs in enumerate(observations):
        for cam_name, visual_obs in obs.items():
            rgb = visual_obs["rgb"]  # (H, W, 3)
            depth = visual_obs["depth"]  # (H, W)
            pcd = visual_obs["pcd"]  # (H*W, 3)

            if depth is not None:
                depth = np.clip((depth * 65535).astype(np.uint16), 0, 65535)

            data[f"{cam_name}_rgb"].append(rgb)
            data[f"{cam_name}_depth"].append(depth)
            data[f"{cam_name}_pcd"].append(pcd)

            if save_frame:
                if rgb is not None:
                    rgb = Image.fromarray(rgb)
                    rgb.save(
                        pjoin(frame_save_dir, f"{frame_idx:04d}_{cam_name}_rgb.png")
                    )

                if depth is not None:
                    depth = Image.fromarray(depth)
                    depth.save(
                        pjoin(frame_save_dir, f"{frame_idx:04d}_{cam_name}_depth.png")
                    )

                if pcd is not None and args.get_pcd:
                    pcd = pcd.reshape(-1, 3)
                    write_pointcloud(
                        pcd,
                        pjoin(frame_save_dir, f"{frame_idx:04d}_{cam_name}_pcd.pcd"),
                    )

    cam_names = [cam.get_name() for cam in cams]
    for cam_name in cam_names:
        for vis_type, ext in [("rgb", "mp4"), ("depth", "mkv")]:
            write_video(
                data[f"{cam_name}_{vis_type}"],
                pjoin(save_dir, f"{cam_name}_{vis_type}.{ext}"),
            )
        if args.get_pcd:
            write_pickle(
                pjoin(save_dir, f"{cam_name}_pcd.pkl"), data[f"{cam_name}_pcd"]
            )


def save_metadata(cameras: list[VisionSensor], save_dir):
    data = {}
    for cam in cameras:
        data |= {
            f"depth_min_{cam.get_name()}": cam.get_near_clipping_plane(),
            f"depth_max_{cam.get_name()}": cam.get_far_clipping_plane(),
            f"cam_intr_{cam.get_name()}": cam.get_intrinsic_matrix(),
            f"cam_extr_{cam.get_name()}": cam.get_matrix(),
        }
    data = ensure_numpy_as_list(data)
    write_json(pjoin(save_dir, "metadata.json"), data)


####################################
## Replay demo
####################################
def replay_demo(
    cfg: dict, object_states: dict, cams: list[VisionSensor], save_dir: str, traj=None
):
    ## Environment setup
    for object_name in cfg["objects"]:
        pos = env_setup[f"init_{object_name}_pos"]
        quat = env_setup[f"init_{object_name}_quat"]
        object = Object.get_object(object_name)
        object.set_position(pos)
        object.set_orientation(quat_to_euler(quat))
        print(
            "Initially set object",
            object_name,
            "to",
            float_array_to_str(pos),
            float_array_to_str(quat_to_euler(quat)),
        )

    for joint_name in cfg["joints"]:
        q = env_setup[f"init_{joint_name}_q"]
        joint = Joint(joint_name)
        joint.set_joint_position(q)
        print("Initially set joint", joint_name, "to", float_array_to_str(q))

    if args.with_robot:
        arm, gripper = Panda(), PandaGripper()
        arm.set_joint_positions(env_setup["init_q"][:7])
        gripper.set_joint_positions(env_setup["init_q"][7:])
        gripper.set_control_loop_enabled(True)

    ## Replay and capture observations
    sim.start()
    sim.step()

    observations = []
    for i, object_state in enumerate(object_states):
        for object_name in cfg["objects"]:
            object = Object.get_object(object_name)
            object_pos = object_state[f"{object_name}_pos"]
            object_quat = object_state[f"{object_name}_quat"]
            object.set_position(object_pos)
            object.set_quaternion(wxyz_to_xyzw(object_quat))

            # print(
            #     "[DEBUG] Set object",
            #     object_name,
            #     "to",
            #     float_array_to_str(object_pos),
            #     float_array_to_str(quat_to_euler(object_quat)),
            # )

        for joint_name in cfg["joints"]:
            joint = Joint(joint_name)
            joint.set_joint_position(object_state[f"{joint_name}_q"])
            # joint.set_joint_target_velocity(object_state[f'{joint_name}_v'])

            # print(
            #     "[DEBUG] Set joint", joint_name, "to", object_state[f"{joint_name}_q"]
            # )

        if args.with_robot:
            q = traj["q"][i]
            arm.set_joint_positions(q[:7], disable_dynamics=False)
            gripper.set_joint_positions(q[7:], disable_dynamics=False)

        obs = get_observations(cams)
        observations.append(obs)

        sim.step()

    ## Save
    save_observations(observations, save_dir, cams, save_frame=args.save_frame)

    ## Shutdown
    sim.stop()


####################################
## Main
####################################

if __name__ == "__main__":
    ## Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gravity", type=bool, default=False)
    parser.add_argument("--task", required=True)
    parser.add_argument("--get_pcd", action="store_true")
    parser.add_argument("--max_demo", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_frame", action="store_true")
    parser.add_argument("--with_robot", action="store_true")
    args = parser.parse_args()
    cfg = read_yaml("data/cfg/rlbench_objects_workaround.yaml")[args.task]

    ## Launch PyRep
    sim = PyRep()
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    if args.with_robot:
        ttt_file = os.path.join(DIR_PATH, "../rlbench", "task_design.ttt")
    else:
        ttt_file = os.path.join(DIR_PATH, "../rlbench", "task_design_wo_franka.ttt")
    sim.launch(ttt_file, headless=args.headless)
    if not args.gravity:
        sim.script_call(
            "setGravity@PyRep", simConst.sim_scripttype_addonscript, floats=[0, 0, 0]
        )
    sim.set_simulation_timestep(1 / 60)  # Control frequency 60Hz

    ## Load task
    ttm_file = os.path.join(DIR_PATH, "../rlbench/task_ttms", f"{args.task}.ttm")
    base_object = sim.import_model(ttm_file)

    ## Load demo
    TaskName = "".join([x.capitalize() for x in args.task.split("_")])
    demo_path = f"trajectories/{TaskName}-v0/trajectory-unified_with_object_states.pkl"
    demo_data = read_pickle(demo_path)

    ####################################
    ## Main
    ####################################
    base_save_dir = mkdir(pjoin("outputs", args.task, "norobot_replay"))

    ## Init cameras
    cams = init_cameras()
    save_metadata(cams, base_save_dir)

    ## Start replay
    for i, demo in enumerate(demo_data["demos"]["franka"]):
        if i >= args.max_demo:
            break

        env_setup = demo["env_setup"]
        object_states = demo["object_states"]
        traj = demo["robot_traj"]

        ## Replay demo
        cur_save_dir = mkdir(pjoin(base_save_dir, f"demo_{i:04d}"))
        if not len(object_states) == demo["episode_len"]:
            print(
                f"Skipping {cur_save_dir} as inconsistent episode length, recorded {len(object_states)}, expected {demo['episode_len']}"
            )
            continue
        if (
            os.path.exists(os.path.join(cur_save_dir, "cam_front_rgb.mp4"))
            and not args.overwrite
        ):
            print(f"Skipping {cur_save_dir} as the demo files exist")
            continue
        else:
            print(f"Replaying demo -> {cur_save_dir}")
        replay_demo(cfg, object_states, cams, cur_save_dir, traj=traj)

    ## Exit CoppeliaSim
    sim.shutdown()
