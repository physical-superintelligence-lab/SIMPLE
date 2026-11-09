# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import os

import numpy as np
from simple.agents.primitive_agent import PrimitiveAgent
from simple.constants import GripperAction
from simple.envs.wrappers.episode_extractor import EpisodeExtractor
from simple.core.action import ActionCmd
from simple.robots.protocols import Humanoid, DualArm, Wholebody
from simple.baselines.client import HttpActionClient

from PIL import Image
from queue import Queue

STATE_SLICES = [ # shoule be consistent with scripts/postprocess_psi0.py
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
]

def from_psi0_upper_joints(psi0_action):
    return np.concatenate([
        psi0_action[14:28],
        psi0_action[0:3], # left thumb
        psi0_action[5:7], # left index
        psi0_action[3:5], # left middle
        psi0_action[7:14], # right hand
    ])

def psi0_action_to_cmd(robot, psi0_action) -> ActionCmd:
    """One psi0 action (the model output, also the `action` of the psi0 training data, 36 dims) ->
    the g1_wholebody command: upper-body joint targets, waist joints and the AMO lower-body command.
    Shared by Psi0Agent and ReplayAgent so replaying psi0 data executes exactly what eval executes."""
    target_qpos = dict(zip(robot.joint_names[15:], from_psi0_upper_joints(psi0_action[:28])))
    target_waist_qpos = {
        "waist_yaw_joint": psi0_action[30],
        "waist_roll_joint": psi0_action[28],
        "waist_pitch_joint": psi0_action[29],
    }
    command = [
        psi0_action[32],  # vx
        psi0_action[35],  # target yaw
        psi0_action[33],  # vy
        psi0_action[31] - 0.75,  # d_height
        psi0_action[30],  # torso yaw
        psi0_action[29],  # torso pitch
        psi0_action[28],  # torso roll
        psi0_action[34],  # turning flag
    ]
    return ActionCmd(
        "eval_move_actuators",
        target_qpos=target_qpos,
        action_command=command,
        waist_qpos=target_waist_qpos,
    )


class Psi0Agent(PrimitiveAgent):

    def __init__(
        self,
        robot,
        host: str,
        port: int,
        upsample_factor=1,
        client=None,
        **kwargs,
    ):
        super().__init__(robot, **kwargs)

        self.server_ip = host # if access server host inside docker container
        self.server_port = port
        self.upsample_factor = upsample_factor

        self.client = client or HttpActionClient(self.server_ip, self.server_port)
        self._global_step_idx = 0
        self._session_idx = 0
        self._session_id = self._make_session_id()

        # last command (high level input to lower policy)
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.75]) # FIXME hardcoded for g1 wholebody, need to be more general in the future
        self._reset_history = True

    def needs_render_next(self) -> bool:
        # the image is only sent to the server when the queued action chunk runs out
        return len(self._action_queue) == 0

    def _make_session_id(self) -> str:
        return f"psi0-{os.getpid()}-{self._session_idx}"

    def get_action(
        self, 
        observation, 
        instruction=None, 
        info=None, 
        conditions=None, 
        **kwargs
    ):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if len(self._action_queue) == 0:
            # send query to server
            observations = {
                "rgb_head_stereo_left": observation["head_stereo_left"], # np.zeros_like()
            }

            proprio = observation["joint_qpos"][None]
            states = np.concatenate(
                [proprio[:, s:e] for _, s, e in STATE_SLICES] + [
                    self._last_cmd_torso_rpyh[None]
                ],
                axis=1,
            ).astype(np.float32) # (1, 32)
            state_dict = {"states": states} # np.zeros_like()

            if self._reset_history:
                history = {
                    "reset": True,
                    "session_id": self._session_id,
                    "episode_index": int(info.get("episode_index", -1)) if info is not None else -1,
                    "step_index": int(self._global_step_idx),
                }
                self._reset_history = False
            else:
                history = {
                    "session_id": self._session_id,
                    "episode_index": int(info.get("episode_index", -1)) if info is not None else -1,
                    "step_index": int(self._global_step_idx),
                }
            pred_action, *_ = self.client.query_action(
                observations, 
                instruction or "bend to pick up the object", 
                state_dict, 
                {},
                history=history, 
                dataset="simple",
            )
            print(f"Received {pred_action.shape[0]} actions from server.")
            for i in range(pred_action.shape[0]):
                for _ in range(self.upsample_factor): # account for upsampling during training
                    action_cmd = psi0_action_to_cmd(self.robot, pred_action[i])
                    self.queue_action(action_cmd)
                    command = action_cmd["action_command"]
                    print(f"step:{self._global_step_idx},vx:{command[0]},target_yaw:{command[1]}")

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type == "eval_move_actuators":
            self._last_cmd_torso_rpyh = np.array([
                action_cmd["action_command"][6], # torso roll
                action_cmd["action_command"][5], # torso pitch
                action_cmd["action_command"][4], # torso yaw
                action_cmd["action_command"][3] + 0.75,
            ], dtype=np.float32)
        self._last_pred_action = action_cmd
        self._global_step_idx += 1
        return action_cmd
    
    def reset(self):
        super().reset()  # clear queue

        self._global_step_idx = 0
        self._session_idx += 1
        self._session_id = self._make_session_id()
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.75])
