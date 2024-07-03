from typing import Callable, List

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.errors import ConfigurationPathError

from rlbench.backend.exceptions import (
    BoundaryError,
    DemoError,
    NoWaypointsError,
    WaypointError,
)
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig


class MyScene(Scene):
    def __init__(
        self,
        pyrep: PyRep,
        robot: Robot,
        obs_config: ObservationConfig = ObservationConfig(),
        robot_setup: str = "panda",
    ):
        super().__init__(pyrep, robot, obs_config, robot_setup)

    def get_demo(self, record: bool = True,
                 callable_each_step: Callable[[Observation], None] = None,
                 callable_when_reach_waypoint: Callable[[int], None] = None,
                 randomly_place: bool = True) -> Demo:
        """Returns a demo (list of observations)"""

        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            self._joint_position_action = None
            gripper_open = 1.0 if self.robot.gripper.get_open_amount()[0] > 0.9 else 0.0
            self._demo_record_step(demo, record, callable_each_step)
        while True:
            success = False
            for i, point in enumerate(waypoints):
                point.start_of_path()
                if point.skip:
                    continue
                grasped_objects = self.robot.gripper.get_grasped_objects()
                colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                    object_type=ObjectType.SHAPE) if s not in grasped_objects
                                    and s not in self._robot_shapes and s.is_collidable()
                                    and self.robot.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                    self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    gripper = self.robot.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                gripper_open = 1.0
                                done = gripper.actuate(gripper_open, 0.04)
                                self.step()
                                self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                gripper_open = 0.0
                                done = gripper.actuate(gripper_open, 0.04)
                                self.step()
                                self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            gripper_open = num
                            done = gripper.actuate(gripper_open, 0.04)
                            self.step()
                            self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)

                if callable_when_reach_waypoint is not None:
                    callable_when_reach_waypoint(i)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.step()
                self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        processed_demo = Demo(demo)
        processed_demo.num_reset_attempts = self._attempts + 1
        return processed_demo