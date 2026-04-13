"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import time
import numpy as np
from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd
from openpi_client import websocket_client_policy as _websocket_client_policy

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

class Pi05DecoupledWbcAgent(SonicDecoupledWbcAgent):
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
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.74]) # FIXME hardcoded for g1 wholebody, need to be more general in the future
        self._reset_history = True

        indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.sonic_upper_joint_names = [name for name, idx in self._dwbc_robot_model.joint_to_dof_index.items() if idx in indices]

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

            proprio = observation["joint_qpos"][None]
            waist_rpy = [proprio[:,13:15],proprio[:,12:13]]
            states = np.concatenate(
                [proprio[:, s:e] for _, s, e in STATE_SLICES] + waist_rpy + [
                    self._last_cmd_torso_rpyh[None][:, -1:]
                ],
                axis=1,
            ).astype(np.float32) # (1, 32)
            # state_dict = {"states": states} # np.zeros_like()
            obs = {
                "observation/image": observation["head_stereo_left"],
                "states": states[0, :28], 
                "prompt": instruction or "As a smart robot agent, what to do next?",
            }
            
            if self._reset_history:
                obs["reset"] = True
                self._reset_history = False
            else:
                obs["reset"] = False

            pred_action = self.client.infer(obs)["actions"]
            print(f"step {self._global_step_idx}: Received {pred_action.shape[0]} actions from server.")
            
            """ # DEBUG
            if self._global_step_idx < 180:
                pred_action = pred_action.copy()
                pred_action[:, 31] = 0.72

            if self._global_step_idx <= 180:
                pred_action = pred_action.copy()
                pred_action[:, 32] = 0.2 """

            print(pred_action[:, 31])
            print(pred_action[:, 32])

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
                    self.queue_action(ActionCmd(
                        "vla_cmd", 
                        target_upper_body_pose={**target_qpos, **target_waist_qpos},  # (31,)
                        navigate_cmd=pred_action[i][32:36],
                        base_height_command=pred_action[i][31:32],
                    ))

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type == "vla_cmd":
            proprio = self.robot.prepare_obs()
            wbc_obs = self._build_wbc_observation(proprio)
            self._wbc_policy.set_observation(wbc_obs)
            t_now = time.monotonic()
            control_freq = self._control_frequency
            target_time = t_now + 1 / control_freq

            target_upper_body_pose = np.array([
                action_cmd["target_upper_body_pose"][jName] for jName in self.sonic_upper_joint_names
            ], dtype=np.float32)
            goal = {
                "target_upper_body_pose": target_upper_body_pose,
                "navigate_cmd": action_cmd["navigate_cmd"],
                "base_height_command": action_cmd["base_height_command"],
                "target_time": target_time,
                "interpolation_garbage_collection_time": t_now - 2 / control_freq,
                "timestamp": t_now,
            }
            self._wbc_policy.set_goal(goal)
            wbc_action = self._wbc_policy.get_action(time=t_now)
            self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
            self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
            self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")

            # createa a new ActionCmd for the g1_sonic robot
            action_cmd = ActionCmd(
                "decoupled_wbc",
                target_q=self._cached_target_q,
                left_hand_q=self._cached_left_hand_q,
                right_hand_q=self._cached_right_hand_q,
            )
            self._last_cmd_torso_rpyh = np.array([0, 0, 0, goal["base_height_command"][0]])
        else:
            raise ValueError(f"Unexpected action type {action_cmd.type} from queue.")

        self._last_pred_action = action_cmd
        self._global_step_idx += 1
        return action_cmd
    
    def reset(self, **kwargs):
        super().reset(**kwargs)  # clear queue

        self._global_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.74])