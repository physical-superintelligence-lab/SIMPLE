# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import mujoco
import mujoco.viewer
import numpy as np
import time
from threading import Lock
from typing import Any

from simple.core.task import Task
from simple.envs.base_dual_env import BaseDualSim
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

class SonicLocoManipEnv(BaseDualSim):

    _success: bool 

    def __init__(
        self, 
        task : str | Task, #
        sonic_config: dict , 
        # config: Dict[str, Any],
        sim_mode="mujoco_isaac",  # =SIM_MODE.MUJOCO_ISAAC
        headless=True, 
        enable_image_publish=False,
        
        *args, 
        **kwargs
    ) -> None:
        super().__init__(task, sim_mode, headless, sonic_config=sonic_config, *args, **kwargs)
        self.sonic_config = sonic_config
        
        self.offscreen = "isaac" in sim_mode or headless
        self.onscreen = not self.offscreen

        self.reward_lock = Lock()

        self.last_reward = 0
        # if self.offscreen:
        #     self.init_renderers()

        try:
            if self.sonic_config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.sonic_config["DOMAIN_ID"], self.sonic_config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.sonic_config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.sim_thread = None
        self.viewer = None
        # False: step MuJoCo only, and fill the observation from MuJoCo (camera images and joint_qpos)
        # instead of syncing and rendering Isaac. Set it per step when only some observations need
        # Isaac images, e.g. a policy that reads an image once per action chunk.
        self.render_isaac = True
        self.timing_event_sink = None
        self.timing_context = None

    def set_timing_event_sink(self, sink) -> None:
        """Install an optional evaluation-only timing event sink."""
        self.timing_event_sink = sink

    def set_timing_context(self, context) -> None:
        self.timing_context = context

    def _emit_timing(self, kind: str) -> None:
        if self.timing_event_sink is not None and self.timing_context is not None:
            self.timing_event_sink(
                {
                    "kind": kind,
                    "perf_ns": time.perf_counter_ns(),
                    **self.timing_context,
                }
            )


    def init_publisher(self):
        pass

    # adapted from base_sim.update_viewer
    def update_viewer(self): # FIXME 
        if self.viewer is not None:
            self.viewer.sync() # where .sync is called?
    
    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def _get_obs(self):
        qpos = np.asarray(list(self.mujoco.get_robot_qpos().values()), dtype=np.float32)
        if self.isaac and self.render_isaac:
            isaacsim_joint_indices = []
            for jname, _ in self.mujoco.get_robot_qpos().items():
                isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
                isaacsim_joint_indices.append(self.isaac.robot.get_dof_index(isaac_jname))
            qpos = self.isaac.robot.get_joint_positions(joint_indices=isaacsim_joint_indices)

        return {
            "joint_qpos": qpos,
            ** self._render_frame()
        }
    
    def _get_info(self):
        info = {}
        for k, v in self.mujoco.mj_objects.items():
            info[str(k)] = np.concatenate([v.xpos, v.xquat])

        # get wholebody proprioception info 
        # currently, it is returned as part of info dict, 
        # but we may want to merge into obs in the future
        proprio = self.task.robot.prepare_obs()
        return {**info, "proprio": proprio}
    
    def reset(
        self, 
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:  # type: ignore
        super().reset(seed=seed, options=options)
        # reset task layout in mujoco with domain randomization
        self.task.reset(seed, options)
        # update mujoco layout with new task layout
        self.mujoco.update_layout(sonic_config=self.sonic_config)

        # FIXME
        self.mjSpec = self.mujoco.mjSpec
        self.mjModel = self.mujoco.mjModel
        self.mjData = self.mujoco.mjData
        
        # set up viewer
        if self.onscreen:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = mujoco.viewer.launch_passive(
                self.mjModel,
                self.mjData,
                key_callback=self.task.robot.elastic_band.MujuocoKeyCallback,
                show_left_ui=False,
                show_right_ui=False,
            )
        else:
            mujoco.mj_forward(self.mjModel, self.mjData)
            self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 100
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 3
            self.viewer.cam.lookat = np.array([0, 0, 0.38])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE # mjCAMERA_TRACKING

        self.mujoco.step(render=False)

        if self.isaac is not None:
            self.isaac.reset()
            self.isaac.update_layout()
            self.isaac.step(self.mujoco)

        self.control_decimal = int((1/self.mujoco.physics_dt) / self.task.render_hz)
        self.step_count = 0
        self.close_cmd_received = False
        obs = self._get_obs()
        info = self._get_info()
        self._success = False
        return obs, info
    
    def step(self, action):
        for i in range(self.control_decimal):
            # self.check_fall()
            # with self._telemetry.timer("mujoco.apply_action"):
            self.mujoco.apply_action(action)

            # with self._telemetry.timer("mujoco.step"):
            self.mujoco.step(render=False) # FIXME  # only render on last step
            # Optional per-physics-step hook (unset by default, so nothing
            # changes for existing entry points).  The wbc path uses it to
            # publish robot state at the physics rate, matching gear_sonic's
            # own sim loop. Run it immediately after MuJoCo advances, before
            # any IsaacSim synchronization/rendering work can block the external
            # controller's lowstate heartbeat.
            if getattr(self, "substep_callback", None) is not None:
                self.substep_callback()
            if self.isaac and self.render_isaac and (getattr(self, "sync_isaac_every_substep", True) or i == self.control_decimal - 1):
                self._emit_timing("isaac_step_begin")
                self.isaac.step(self.mujoco)
                self._emit_timing("isaac_step_end")
        
        self.step_count += 1
        # with self._telemetry.timer("env._get_obs"):
        obs = self._get_obs()
        # with self._telemetry.timer("env._get_info"):
        info = self._get_info()

        # with self._telemetry.timer("task.compute_reward"):
        reward = self.task.compute_reward(info , mujoco_env=self.mujoco)
        
        # with self._telemetry.timer("task.check_success"):
        terminated = self.task.check_success(info, mujoco_env=self.mujoco)
        
        truncated = False
        self._success = terminated
        return obs, reward, terminated, truncated, info
    
    """ def check_fall(self):
        self.fall = False
        if self.mjData.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mjData.qpos[2]:.3f} m")

        if self.fall:
            self.__reset() """

    """ def __reset(self): # FIXME
        mujoco.mj_resetData(self.mjModel, self.mjData)
        # Re-arm the elastic band so the robot hangs safely again after a fall.
        if self.task.robot.elastic_band:
            self.task.robot.elastic_band.length = 0.0
            self.task.robot.elastic_band.enable = True """
    
    def render(self):
        return self._render_frame()

    def refresh_obs(self):
        """Sync Isaac to the current MuJoCo state and return a fresh observation, e.g. after steps
        taken with render_isaac off."""
        if self.isaac and self.render_isaac:
            self.isaac.step(self.mujoco)
        return self._get_obs()

    def _render_frame(self):
        self._emit_timing("render_readback_begin")
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
                result = frame_tiled
            else:
                result = frame_isaac
        else:
            result = frame_mujoco
        self._emit_timing("render_readback_end")
        return result

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        super().close()
