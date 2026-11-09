# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Record a g1_wholebody (AMO) rollout, e.g. simple.cli.datagen, directly in the psi0 LeRobot format
(simple.datasets.psi0), so it replays with simple.cli.replay and trains like the MP tasks of
SIMPLE-decoupled-wbc.

Wrap it outside StandStabilizationWrapper: the stand warm-up done inside reset is not recorded, and
frame 0 is the post-warm-up observation, as in eval and replay.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import gymnasium as gym
import numpy as np

from simple.datasets.psi0 import Psi0DatasetWriter

if TYPE_CHECKING:
    from simple.agents.base_agent import BaseAgent as Agent


def wholebody_targets(robot) -> dict[str, float]:
    """Joint targets the g1_wholebody robot applied in its last step, by joint name: AMO/waist PD targets for
    legs and waist, the position targets of the arms, and the hand targets (as LerobotRecorder records them)."""
    targets = {j: float(robot.actuators[j].ctrl.item()) for j in robot.joint_names}
    lower_names = getattr(robot, "output_joint_names", None) or robot.joint_names[:15]
    for name, value in zip(lower_names, robot.pd_target):
        targets[name] = float(value)
    hand = robot.hand_target_qpos()
    for name in robot.hand_names:
        targets[name] = float(hand.get(name, 0.0))
    return targets


def _is_stand(action) -> bool:
    return getattr(action, "type", None) == "loco_command" and action.parameters.get("motion_type") == "stand"


class Psi0Recorder(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Pairs each step's command with the observation it was computed from (the one before the step).

    Saves an episode when it terminates with reward > 0.9 (``auto_save``), like LerobotRecorder.
    ``skip_leading_stand``: do not record the stand commands an episode starts with (the solver's
    warm-up), so frame 0 is the first command of the task.
    """

    def __init__(
        self,
        env: gym.Env,
        root_dir: str = "data/datagen",
        agent: Optional[Agent] = None,
        camera: str = "head_stereo_left",
        dr_level: int | None = None,
        auto_save: bool = True,
        skip_leading_stand: bool = True,
    ):
        gym.utils.RecordConstructorArgs.__init__(self, root_dir=root_dir, camera=camera)
        gym.Wrapper.__init__(self, env)
        from simple.robots.protocols import Wholebody

        self.task = env.unwrapped.task  # type: ignore
        self.robot = self.task.robot
        if not isinstance(self.robot, Wholebody) or not hasattr(self.robot, "amo_policy"):
            raise ValueError(f"psi0 recording needs the g1_wholebody (AMO) robot, got {type(self.robot).__name__}")
        save_dr_level = self.task.dr.level if dr_level is None else dr_level
        self.root_dir = f"{os.path.abspath(root_dir)}/{env.unwrapped.spec.id}/level-{save_dr_level}"  # type: ignore
        self.writer = Psi0DatasetWriter(self.root_dir, fps=self.task.metadata["render_hz"], controller="amo")
        self.agent = agent
        self.camera = camera
        self.auto_save = auto_save
        self.skip_leading_stand = skip_leading_stand
        self.episode_index = self.writer.num_episodes  # next episode to be saved
        self._last_obs = None
        self.env_conf = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.writer.start_episode()
        if self.agent is not None:
            self.agent.reset()
        self.env_conf = self.task.state_dict()
        self._last_obs = obs
        return obs, info

    def clear_episode_buffer(self):
        """Drop the frames of the current episode (e.g. the motion plan failed)."""
        self.writer.discard_episode()

    def step(self, action):
        obs_before = self._last_obs
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        if obs_before is None:
            return obs, reward, terminated, truncated, info
        if self.skip_leading_stand and self.writer.episode_length == 0 and _is_stand(action):
            return obs, reward, terminated, truncated, info

        amo = self.robot.amo_policy
        command = np.asarray(amo.obs_command, dtype=np.float64)  # [vx, vy, _, torso yaw/pitch/roll, height x3]
        self.writer.add_frame(
            image=obs_before[self.camera],
            qpos=dict(zip(self.robot.joint_names, np.asarray(obs_before["joint_qpos"], dtype=np.float64))),
            targets=wholebody_targets(self.robot),
            base_height=float(command[6]),
            vx=float(command[0]),
            vy=float(command[1]),
            vyaw_or_turning_flag=float(np.asarray(amo.obs_turning_flag).reshape(-1)[0]),
            target_yaw=float(np.asarray(amo.obs_target_yaw).reshape(-1)[0]),
        )
        if self.auto_save and (terminated or truncated):
            self.finish_episode(float(reward) > 0.9)
        return obs, reward, terminated, truncated, info

    def finish_episode(self, success: bool):
        """Save the buffered episode (with its environment_config) on success, else discard it."""
        if success and self.writer.episode_length > 0:
            self.writer.save_episode(self.task.instruction, self.env_conf)
            self.episode_index = self.writer.num_episodes
        else:
            self.writer.discard_episode()

    def close(self):
        self.writer.close()
        super().close()
