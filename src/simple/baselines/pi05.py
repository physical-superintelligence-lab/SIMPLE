"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import numpy as np
from simple.agents.primitive_agent import PrimitiveAgent
from simple.constants import GripperAction
from simple.envs.wrappers.episode_extractor import EpisodeExtractor
from simple.core.action import ActionCmd
from simple.robots.protocols import Humanoid, DualArm, Wholebody
# from simple.baselines.client import HttpActionClient
from openpi_client import websocket_client_policy as _websocket_client_policy

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

class Pi05Agent(PrimitiveAgent):

    def __init__(self, robot, host: str, port: int, upsample_factor=1, **kwargs):
        super().__init__(robot, **kwargs)
        
        self.server_ip = host # if access server host inside docker container
        self.server_port = port
        self.upsample_factor = upsample_factor

        self.client = _websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
            api_key=None,
        )
        self._global_step_idx = 0

        # last command (high level input to lower policy)
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.75]) # FIXME hardcoded for g1 wholebody, need to be more general in the future
        self._reset_history = True

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

        if self._global_step_idx == 0:
            # standing up
            for _ in range(60):
                self.queue_loco_command(
                    command=[0,0,0,0,0,0,0,0],
                    motion_type="stand",
                    keep_waist_pose=False
                )

        if len(self._action_queue) == 0:
            # send query to server

            proprio = observation["joint_qpos"][None]
            states = np.concatenate(
                [proprio[:, s:e] for _, s, e in STATE_SLICES] + [
                    self._last_cmd_torso_rpyh[None]
                ],
                axis=1,
            ).astype(np.float32) # (1, 32)
            # state_dict = {"states": states} # np.zeros_like()
            obs = {
                "observation/image": observation["head_stereo_left"],
                "states": states[0, :28], 
                "prompt": instruction or "bend to pick up the object",
            }
            
            if self._reset_history:
                obs["reset"] = True
                self._reset_history = False
            else:
                obs["reset"] = False

            pred_action = self.client.infer(obs)["actions"]
            print(f"Received {pred_action.shape[0]} actions from server.")
            for i in range(pred_action.shape[0]):
                for _ in range(self.upsample_factor): # account for upsampling during training
                    target_qpos = dict(
                        zip(
                            self.robot.joint_names[15:],
                            from_psi0_upper_joints(pred_action[i][:28])
                        )
                    )

                    target_waist_qpos = {
                        "waist_yaw_joint": pred_action[i][30], 
                        "waist_roll_joint": pred_action[i][28], 
                        "waist_pitch_joint": pred_action[i][29] 
                    }
                    # FIXME very ugly code here!
                    command = [
                        pred_action[i][32], # vx
                        pred_action[i][35], # target yaw
                        pred_action[i][33], # vy
                        pred_action[i][31] - 0.75,  # d_height
                        pred_action[i][30], # torso yaw
                        pred_action[i][29], # torso pitch
                        pred_action[i][28],  # torso roll
                        pred_action[i][34], #turing flag 
                    ]
                    self.queue_action(ActionCmd(
                        "eval_move_actuators", 
                        target_qpos=target_qpos,
                        action_command=command,
                        waist_qpos=target_waist_qpos
                    ))

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
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.75])