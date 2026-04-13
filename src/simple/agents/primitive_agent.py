"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import copy
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum

from simple.core.robot import Robot
from simple.core.action import ActionCmd
from simple.robots.protocols import Controllable
from .base_agent import BaseAgent

# class PrimitiveActionEnum(str, Enum):
#     # MOVE_TO_QPOS = "move_to_qpos"
#     MOVE_EEF_TO = "move_eef_to"

#     CLOSE_EEF = "close_eef"
#     OPEN_EEF = "open_eef"

#     # CLOSE_GRIPPER = "close_gripper"
#     # OPEN_GRIPPER = "open_gripper"
#     # DEXTEROUS_GRASP = "dexterous_grasp"
#     # DEXTEROUS_RELEASE = "dexterous_release"

# class PrimitiveAction:
#     def __init__(self, action_type: PrimitiveActionEnum, parameters: dict):
#         self.action_type = action_type
#         self.parameters = parameters



class PrimitiveAgent(BaseAgent):

    def __init__(self, robot: Robot, **kwargs):
        super().__init__(robot)

        # merge old base agent
        self._action_queue = deque()
        # self._ctrl_queue_callbacks = []

    def pad_non_exist_dim(self, qpos:dict, robot: Robot):
        assert isinstance(robot, Controllable)
        if len(qpos) != robot.dof:
            qpos_full = copy.deepcopy(qpos)
            for jname, _ in qpos.items():
                if jname not in robot.joint_names:
                    qpos_full[jname] = robot.init_joint_states[jname]

    def queue_action(self, action_cmd: ActionCmd):
        self._action_queue.append(action_cmd)

    def queue_loco_command(self, command: list, **kwargs):
        self._action_queue.append(
            ActionCmd(
                "loco_command",
                command=command,
                **kwargs
            )
        )

    def queue_move_qpos(self, target_qpos: dict, **kwargs):
        self._action_queue.append(
            ActionCmd(
                "move_qpos", 
                target_qpos=target_qpos,
                **kwargs
            )
        )
        self._last_qpos = target_qpos
        
    def queue_move_qpos_with_eef(self, target_qpos, eef_state, **kwargs):
        self._action_queue.append(
            ActionCmd(
                "move_qpos_with_eef", 
                target_qpos=target_qpos, 
                eef_state=eef_state,
                **kwargs
            )
        )
        self._last_qpos = target_qpos

    def queue_move_eef(self, target_pose):
        ...

    def queue_follow_path(self, path, **kwargs):
        for p in path:
            # p_pad = self.pad_non_exist_dim(p, self.robot)
            self.queue_move_qpos(p, **kwargs)

    def queue_follow_path_with_eef(self, path, eef_state, **kwargs):
        for p in path:
            self.queue_move_qpos_with_eef(p, eef_state, **kwargs)

    def queue_close_gripper(self, preemptive=False, **kwargs):
        # if preemptive:
        #     self._action_queue.clear()
        # self._action_queue.append(
        #     ActionCmd(hand)
        # )
        self._action_queue.append(
            ActionCmd(
                "close_eef",
                target_qpos = self._last_qpos,
                **kwargs
            )
            )

    def queue_open_gripper(self, preemptive=False, **kwargs):
        # self.robot.open_gripper()
        self._action_queue.append(
            ActionCmd(
                "open_eef",
                target_qpos = self._last_qpos,
                **kwargs

            )
        )

    def queue_dexterous_grasp(self, hand_qpos):
        ...

    def get_action(self, observation, instruction=None, **kwargs):
        try:
            if observation is not None and "agent" in observation:
                self._last_qpos = dict(zip(self.robot.joint_names, observation["agent"]))
            action = self._action_queue.popleft()
            self._last_pred_action = action
            return action
        except IndexError:
            raise StopIteration("No more queued actions in PrimitiveAgent.")
        
    def __len__(self):
        return len(self._action_queue)

    def reset(self):
        super().reset()
        self._action_queue = deque()
        # self._ctrl_queue_callbacks = []