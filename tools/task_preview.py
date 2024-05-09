from rlbench.backend.task import Task
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.backend.scene import Scene
from rlbench.backend.task import TASKS_PATH
from rlbench.demo import Demo
from typing import List, Tuple
import numpy as np
import os
import argparse
import cv2

from tools.collect_demo import DemoGetter
from tools.collect_demo import mkdir, read_yaml 

####################################
## Functions
####################################
class DemoPreviewGetter(DemoGetter):
    def __init__(self, args):
        super().__init__(args, writer=None)
    
    def _create_scene(self):
        cam_config = CameraConfig(image_size=(256, 256))
        cam_config.set_all(False)
        cam_config.rgb = True
        obs_config = ObservationConfig()
        obs_config.set_all(False)
        obs_config.front_camera = cam_config
        obs_config.overhead_camera = cam_config
        self.scene = Scene(self.sim, self.robot, obs_config)

    def _try_get_demo(self, variation_index=0):
        self.scene.reset()
        desc = self.scene.init_episode(variation_index, max_attempts=10)
        demo = self.scene.get_demo(record=True)
        return demo

    def save_demo(self, demo: Demo):
        save_dir = mkdir('preview')
        
        h, w, _ = demo[0].front_rgb.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videos = {
            'front_rgb': cv2.VideoWriter(os.path.join(save_dir, f'{self.task.get_name()}_front_rgb.mp4'), fourcc, 30, (w, h)),
            'overhead_rgb': cv2.VideoWriter(os.path.join(save_dir, f'{self.task.get_name()}_overhead_rgb.mp4'), fourcc, 30, (w, h))
        }

        for key, video in videos.items():
            for i, obs in enumerate(demo):
                frame = getattr(obs, key)
                frame = frame[:, :, ::-1]
                video.write(frame)

            video.release()

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
    
    ## Launch getter
    cfg = read_yaml(args.conf)[args.task]
    
    ## Run task
    getter = DemoPreviewGetter(args)
    task_names_ttm = [n.removesuffix('.ttm') for n in os.listdir('rlbench/task_ttms')]
    task_names_py = [n.removesuffix('.py') for n in os.listdir('rlbench/tasks')]
    task_names = sorted(list(set(task_names_ttm) & set(task_names_py)))
    print('Task names:', task_names)
    for task_name in task_names:
        if os.path.exists(os.path.join('preview', f'{task_name}_overhead_rgb.mp4')):
            print(f'Skipping {task_name} as it already exists.')
            continue
        print(f'==== Task: {task_name} ====')
        np.random.seed(args.seed)
        getter.load_task(task_name)
        demo = getter.get_demo(episode_index=0, attempts=3, raise_error=False)
        if demo is not None:
            getter.save_demo(demo) 