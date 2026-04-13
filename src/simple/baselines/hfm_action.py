"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import numpy as np
import transforms3d as t3d
from PIL import Image
from simple.agents.base_agent import BaseAgent, GripperAction, GripperState
from simple.baselines.client import HttpActionClient
from queue import Queue

from simple.core.action import ActionCmd
from simple.robots.protocols import HasParallelGripper

def hack_gripper_action(gripper_act):
    return 0 if gripper_act < 0.7 else 1

class FlowActionAgent(BaseAgent):
    def __init__(self, robot, host: str, port: int):
        super().__init__(robot)
        
        self.server_ip = host # if access server host inside docker container
        self.server_port = port

        self.client = HttpActionClient(self.server_ip, self.server_port)
        self.action_buffer = Queue() # to support action chunking

        # self.step_idx = 0
        self.ik_failed_cnt = 0

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        assert instruction is not None, "Instruction must be provided for OpenVLAAgent."

        self._last_observation = observation

        if self.action_buffer.qsize() == 0:
            image_dict = {
                "front_stereo_left": observation["front_stereo_left"],
                "front_stereo_right": observation["front_stereo_right"],
                "wrist_mount_cam": observation["wrist"],
                "side": observation["side_left"],
            }
            state_dict = {
                "proprio_joint_positions": observation["joint_qpos"], # (9,)
                "proprio_eef_pose": observation["eef_pose"], # (7,)
            }
            history_dict = {k: [] for k in image_dict.keys()}
            condition_dict = conditions if conditions is not None else {}
            self.condition_dict = condition_dict
            self.state_dict = state_dict
            self.history_dict = history_dict

            pred_actions, *_ = self.client.query_action(
                image_dict=image_dict, 
                instruction=instruction, 
                state_dict=state_dict, 
                condition_dict=condition_dict, 
                history=history_dict
            ) # delta: xyz, rpy, openness

            pred_actions = np.array(pred_actions, dtype=np.float32)

            if len(pred_actions.shape) == 2: # N x 7 for action chunking
                for act in pred_actions:
                    self.action_buffer.put(act)
                    break
                # print(f"received {pred_actions.shape[0]} actions") # for openvla (7,)
            else: # single action
                self.action_buffer.put(pred_actions)
                # print(f"received 1 action") # for openvla (7,)

        pred_action = self.action_buffer.get()
        self._last_pred_action = pred_action

        # get current qpos 
        curr_qpos = observation["agent"]

        assert isinstance(self.robot, HasParallelGripper), "Robot must have parallel gripper for OpenVLAAgent."

        # convert to eef pose
        p, q = self.robot.fk(curr_qpos[:7])
        p0, q0 = self.robot.get_eef_pose_from_hand_pose(p, q)

        # apply delta action
        delta_q = t3d.euler.euler2quat(*pred_action[3:6])
        q1 =  t3d.quaternions.qmult(delta_q, q0)
        p1 = p0 + pred_action[:3]
        self._last_pred_eef_pose = np.concatenate([p1, q1])

        p, q = self.robot.get_hand_pose_from_eef_pose(p1, q1)
        # ik to get target qpos
        try:
            target_qpos = self.robot.ik(p, q, curr_qpos[:7])
            gripper_state = "open_eef" if hack_gripper_action(pred_action[-1]) == 1 else "close_eef"
            self.ik_failed_cnt = 0 # reset
        except Exception as e:
            self.ik_failed_cnt += 1
            print(f"ik failed={self.ik_failed_cnt}, keep still, {e}")
            # return np.concatenate([curr_qpos, [GripperAction.keep]])
            target_qpos = curr_qpos[:7]
            gripper_state = "keep"
        
        self._last_qpos = curr_qpos

        # gripper_action = GripperAction(hack_gripper_action(pred_action[-1]))
        # return np.concatenate([target_qpos, [gripper_action]])
        arm_dim = len(target_qpos)
        return ActionCmd(
            "queue_move_qpos_with_eef", 
            target_qpos=dict(zip(self.robot.joint_names[:arm_dim], target_qpos)),
            eef_state=gripper_state
        )
    
    def reset(self, **kwargs):
        super().reset(**kwargs)

        self.ik_failed_cnt = 0
        self.action_buffer = Queue()