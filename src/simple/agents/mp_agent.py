"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""


import numpy as np
from .base_agent import BaseAgent
from simple.constants import GripperAction
from simple.datagen.action_converter import ActionConverterMujoco
from simple.datagen.agents import PlanAgent


class MotionPlanAgent(BaseAgent):
    def __init__(self, plan, robot,render_hz,init_finger=[0.04, 0.04]):
        super().__init__(robot)
        self._gt_step_idx = 0
        self._plan = plan
        self._render_hz = render_hz
        self._init_finger = init_finger
        if self.robot.uid == "aloha":
            ee_link=self.robot.robot_cfg["kinematics"]["ee_link"]
            self.action_converter = ActionConverterMujoco(self._render_hz,self.robot.uid,ee_link=ee_link)
        else:
            self.action_converter = ActionConverterMujoco(self._render_hz,self.robot.uid)

    def get_action_commands(self):
        action_commands=self.action_converter.convert_plan_to_action(self._plan,self._init_finger)
        return action_commands
    
    def init_plan_agent(self):
        action_commands=self.get_action_commands()
        if self.robot.uid == "aloha":
            ee_link=self.robot.robot_cfg["kinematics"]["ee_link"]
            self._plan_agent = PlanAgent(self.robot,action_commands, self.robot.uid,ee_link=ee_link)
        else:
            self._plan_agent = PlanAgent(self.robot,action_commands, self.robot.uid)

    def get_action(self, observation=None, 
                   info=None, 
                   instruction=None, 
                   conditions=None, 
                   replay_gt_action=None, 
                   check_error=False): 
        action=self._plan_agent.step()
        
        
        return action
    def get_state(self):
        return self._plan_agent.state

    def query_action(self, observation, instruction, gt_action=None):
        return self._last_pred_action, 1.0

    def reset(self):
        self._gt_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None