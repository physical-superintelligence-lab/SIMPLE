# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional
import numpy as np

from simple.core.task import Task
from simple.envs.base_dual_env import BaseDualSim
from simple.robots.protocols import Humanoid

# import transforms3d as t3d

from simple.envs.base_dual_env import BaseDualSim
# from simple.constants import GripperAction
# from simple.robots.protocols import Graspable

class LocoManipulationEnv(BaseDualSim):

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
        # False: step MuJoCo only, and fill the observation from MuJoCo (camera images and joint_qpos)
        # instead of syncing and rendering Isaac. Set it per step when only some observations need
        # Isaac images, e.g. a policy that reads an image once per action chunk.
        self.render_isaac = True
    
    def _get_obs(self):
        qpos = np.asarray(list(self.mujoco.get_robot_qpos().values()), dtype=np.float32)
        if self.isaac and self.render_isaac:
            isaacsim_joint_indices = []
            for jname, _ in self.mujoco.get_robot_qpos().items():
                isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
                isaacsim_joint_indices.append(self.isaac.robot.get_dof_index(isaac_jname))
            qpos = self.isaac.robot.get_joint_positions(joint_indices=isaacsim_joint_indices)
        else:
            ...

        return {
            "joint_qpos": qpos,
            ** self._render_frame()
        }
    
    def _get_info(self):
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
        if self.isaac and self.render_isaac:
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


    def refresh_obs(self):
        """Sync Isaac to the current MuJoCo state and return a fresh observation, e.g. after steps
        taken with render_isaac off."""
        if self.isaac and self.render_isaac:
            self.isaac.step(self.mujoco)
        return self._get_obs()

    def _render_frame(self):
        frame_mujoco = self.mujoco.render()

        if self.isaac and self.render_isaac:
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

    def close(self):
        super().close()
