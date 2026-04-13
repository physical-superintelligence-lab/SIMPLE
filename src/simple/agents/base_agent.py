"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from simple.constants import GripperState
from simple.core.robot import Robot
from collections import deque
from abc import ABC, abstractmethod

from enum import IntEnum
class TaskState(IntEnum):
    """ ActionConverterMujoco.convert_plan_to_action """
    rest = 0
    approach = 1
    grasp = 2
    lift = 3
    wait = 4

from enum import IntEnum
class GripperAction(IntEnum):
    open = 1
    close = 0
    keep = -1

class GripperState(IntEnum):
    closed = 0
    open = 1
    opening = 2
    closing = 3

class BaseAgent(ABC):
    state: str = "rest"

    def __init__(self, robot:Robot, **kwargs):
        
        self.robot = robot
        
        # self._gripper_action_tick = 0
        # self._gripper_state = GripperState.open # FIXME should read the status from env

        self._last_qpos = robot.init_joint_states
        # self._target_qpos = None

        self._last_pred_action = None
        self._last_observation = None
        self._last_pred_eef_pose = None

        # self.client = None

        
        self._finger_state = None

        # # FIXME
        # if robot.uid == "aloha":
        #     self.ee_link = self.robot.robot_cfg["kinematics"]["ee_link"]

        # FIXME
        # self.ik_solver = ik_solver
        # self.robot_type = robot_type.lower()
        # self.ee_link = ee_link
        # self._configure_robot()

    # def _configure_robot(self):
    #     """Configure robot-specific parameters."""
    #     if "franka" in self.robot.uid:
    #         self.arm_dof = 7
    #         self.gripper_indices = [7, 8]
    #         self.openness_index = 7
            
    #         # Franka gripper values
    #         self.gripper_close_intermediate = [0.04, 0.04]
    #         self.gripper_close_final = [0.01, 0.01]
    #         self.gripper_open_intermediate = [0.022, 0.022]
    #         self.gripper_open_final = [0.0205, 0.0205]
            
    #         # Franka finger length for IK
    #         self.finger_length = 0.1034  # FRANKA_FINGER_LENGTH
            
    #     elif self.robot.uid == "aloha":
    #         self.arm_dof = 13  # ALOHA dual arm
    #         self.is_right_arm = self.ee_link == "right_gripper_site"
    #         self.openness_index = 13
            
    #         if self.is_right_arm:
    #             self.gripper_indices = [13]
    #             self.gripper_close_intermediate = [0.038]
    #             self.gripper_close_final = [0.02]
    #             self.gripper_open_intermediate = [0.22]
    #             self.gripper_open_final = [0.205]
    #         else:
    #             self.gripper_indices = [6]
    #             self.gripper_close_intermediate = [0.038]
    #             self.gripper_close_final = [0.02]
    #             self.gripper_open_intermediate = [0.22]
    #             self.gripper_open_final = [0.205]
                
    #         # ALOHA finger length (if needed for IK)
    #         self.finger_length = 0.0  # Approximate value

    #     elif self.robot.uid == "vega_1":
    #         self.arm_dof = 17  # Dexmate Vega 1
    #         self.openness_index = 17
            
    #     else:
    #         raise ValueError(f"Unsupported robot type: {self.robot.uid}")

    @abstractmethod
    def get_action(self, observation, instruction=None, **kwargs):
        raise NotImplementedError
    
    def get_last_pred_action(self):
        return self._last_pred_action
    
    def get_last_pred_eef_pose(self):
        return self._last_pred_eef_pose
    
    def get_last_qpose(self):
        return self._last_qpos
    
    def get_last_observation(self):
        return self._last_observation

    def is_opening_or_closing_gripper(self):
        return self._finger_state == GripperState.opening or self._finger_state == GripperState.closing
    
    def query_action(self, obs_image, instruction, gt_action=None):
        raise NotImplementedError
    
    def reset(self, **kwwargs):
        self._last_qpos = self.robot.init_joint_states

        self._last_pred_action = None
        self._last_observation = None
        self._last_pred_eef_pose = None

        self._finger_state = None
