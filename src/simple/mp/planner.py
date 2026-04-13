"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from simple.core.robot import Robot
    from simple.core.layout import Layout
    from simple.core.task import Task
    # from simple.robots.protocols import WristCamMountable
    # from simple.sensors.config import CameraCfg


class MotionPlanner(ABC):

    

    def __init__(self,) -> None:
        ...

    # @abstractmethod
    # def solve(self, task: Task):
    #     raise NotImplementedError

    def batch_plan_for_approach_bodex(self, 
        goal_poses: list[dict], 
        grasp_qpos: dict[str, float], 
        squeeze_qpos: dict[str, float], 
        lift_qpos: dict[str, float],
        **kwargs
    ) -> tuple[list, list]:
        ...

    @abstractmethod
    def batch_plan_for_approach(self, goal_poses, **kwargs) -> tuple[list, list]:
        ...

    @abstractmethod
    def batch_plan_for_lift(self, current_joint_states, lift_height=0.1, step_distance=0.01) -> list[dict[str, float]]:
        ...
        
    @abstractmethod
    def batch_plan_for_move(self, goal_poses, current_joint_states, **kwargs) -> tuple[list, list]:
        ...
        
    # @abstractmethod
    # def batch_plan_for_lower(self, current_joint_states, lower_height=0.1, step_distance=0.01) -> list[dict[str, float]]:
    #     ...        
    
    # @abstractmethod
    # def plan_to_move_eef(self, eef_pose, **kwargs):
    #     ...

