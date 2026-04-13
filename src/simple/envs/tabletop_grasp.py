"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional
import numpy as np

from simple.core.task import Task
from simple.envs.base_dual_env import BaseDualSim
from simple.robots.protocols import Humanoid

import transforms3d as t3d

from simple.envs.base_dual_env import BaseDualSim
from simple.constants import GripperAction
from simple.robots.protocols import Graspable

def get_eef_pose(isaac):  # room should be an instance of IsaacsimEnv
    try:
        robot_world_pos, robot_world_quat = isaac.robot.get_world_pose()
    except Exception as e:
        # If robot is not initialized yet, return default pose
        print(f"Warning: Robot not initialized yet, using default pose. Error: {e}")
        robot_world_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        robot_world_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w, x, y, z
        
    mat_world_robot = np.eye(4)
    mat_world_robot[:3, 3] = robot_world_pos
    mat_world_robot[:3, :3] = t3d.quaternions.quat2mat(robot_world_quat)

    # robot_eef_mat = self.__get_local_pose(isaac_room.robot_eef_xform, mat_world_robot)
    try:
        world_pos, world_ori = isaac.robot_eef_xform.get_world_pose()
    except Exception as e:
        print(f"Warning: Failed to get robot EEF pose, using default. Error: {e}")
        world_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        world_ori = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w, x, y, z
        
    mat_world = np.eye(4)
    mat_world[:3, 3] = world_pos
    mat_world[:3, :3] = t3d.quaternions.quat2mat(world_ori)
    robot_eef_mat = np.linalg.inv(mat_world_robot) @ mat_world
    robot_eef_pos = robot_eef_mat[:3, 3]
    robot_eef_quat = t3d.quaternions.mat2quat(robot_eef_mat[:3, :3])
    # joint_positions = room.robot.get_joint_positions()    
    return np.concatenate([robot_eef_pos, robot_eef_quat]).astype(np.float32)

class TabletopGraspEnv(BaseDualSim):

    _success: bool 
    
    def __init__(
        self, 
        task : str | Task, #
        sim_mode="mujoco_isaac",  # =SIM_MODE.MUJOCO_ISAAC
        headless=True, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(task, sim_mode, headless, *args, **kwargs)
    
    def _get_obs(self):
        qpos = np.asarray(list(self.mujoco.get_robot_qpos().values()), dtype=np.float32)
        # actuators_action= np.asarray(list(self.mujoco.get_actuators_action().values()), dtype=np.float32)
        if self.isaac:
            eef_pose = get_eef_pose(self.isaac)
            isaacsim_joint_indices = []
            for jname, _ in self.mujoco.get_robot_qpos().items():
                isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
                isaacsim_joint_indices.append(self.isaac.robot.get_dof_index(isaac_jname))
            qpos = self.isaac.robot.get_joint_positions(joint_indices=isaacsim_joint_indices)
        else:
            from simple.robots.protocols import HasKinematics
            if isinstance(self.task.robot, HasKinematics):
                p,q = self.task.robot.fk(qpos)
                eef_pose = np.concatenate([p, q]).astype(np.float32)
            else:
                eef_pose = np.zeros((self.task.robot.dof), dtype=np.float32) #np.zeros(self.task.robot.dof, dtype=np.float32)

        return {
            "agent": qpos, 
            "joint_qpos": qpos,
            "eef_pose": eef_pose,
            ** self._render_frame()
        }
    
    def _get_info(self):
        # if hasattr(self, 'target_object_id') and self.target_object_id is not None:
        #     height = self.mujoco.mj_objects[str(self.target_object_id)].xpos[2] - self.initial_target_z
        #     return {"height": height}
        # else:
        #     return {}
        info = {}
        for k, v in self.mujoco.mj_objects.items():
            info[str(k)] = np.concatenate([v.xpos, v.xquat])
        return info

    def reset(
        self, 
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:  # type: ignore
        super().reset(seed=seed, options=options)

        self.task.reset(seed, options)
        self.mujoco.update_layout()
        self.mujoco.step(render=False)
        if self.isaac is not None:
            self.isaac.reset()
            self.isaac.update_layout()
            self.isaac.step(self.mujoco)

        self.step_count = 0
        self.close_cmd_received = False
        obs = self._get_obs()
        info = self._get_info()
        self._success = False
        return obs, info
    
    def step(self, action):
        self.mujoco.apply_action(action)

        self.mujoco.step()
        if self.isaac:
            self.isaac.step(self.mujoco)
        
        self.step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        reward = self.task.compute_reward(info , mujoco_env=self.mujoco)
        terminated = self.task.check_success(info, mujoco_env=self.mujoco)
        
        truncated = False
        self._success = terminated
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self._render_frame()


    def _render_frame(self):
        frame_mujoco = self.mujoco.render()

        if self.isaac:
            frame_isaac = self.isaac.render()
            if "debug" in self.task.metadata and self.task.metadata["debug"]:
                # tile frame_mujoco and frame isaac together
                frame_tiled = {}
                for key, isaac_img in frame_isaac.items():
                    mujoco_img = frame_mujoco[key]
                    width = mujoco_img.shape[1]
                    frame_tiled[key] = np.concatenate([
                        isaac_img[:,:width//2,:],mujoco_img[:,width//2:,:] 
                    ], axis=1)
                return frame_tiled
            else:
                return frame_isaac
        else:
            return frame_mujoco
        #     frame_isaac = { # FIXME dummy
        #         "front_stereo_left": np.zeros((360, 640, 3), dtype=np.uint8),
        #         "front_stereo_right": np.zeros((360, 640, 3), dtype=np.uint8),
        #         "side_left": np.zeros((360, 640, 3), dtype=np.uint8),
        #         "wrist": np.zeros((270, 480, 3), dtype=np.uint8),
        #         "wrist_left": np.zeros((270, 480, 3), dtype=np.uint8) # FIXME
        #     }
        #     if isinstance(self.task.robot, Humanoid):
        #         frame_isaac["head_stereo_left"] = np.zeros((360, 640, 3), dtype=np.uint8)
        #         frame_isaac["head_stereo_right"] = np.zeros((360, 640, 3), dtype=np.uint8)

        # return {
        #     "mujoco": frame_mujoco[self.mujoco.default_camera_name], # FIXME
        #     ** frame_isaac
        # }


    def close(self):
        super().close()