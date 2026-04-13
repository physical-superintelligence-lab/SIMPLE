"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
from gymnasium import spaces
# from simple.core.robot import Robot
from typing import Any, List, Type

class ControllerCfg:
    clazz: Type[Controller]
    dof: int 

class Controller:

    # def __init__(self, robot: Any, *args, **kwargs):
    #     ...

    joints: List[Any]
    actuators: List[Any]

    def __init__(self, cfg: "ControllerCfg", **kwargs) -> None:
        self.cfg = cfg

    @property
    def action_space(self) -> spaces.Space:
        """Returns the action space of the controller."""
        raise NotImplementedError
    
    @property
    def init_joint_states(self) -> list[float]:
        """Returns the initial joint states of the controller."""
        raise NotImplementedError

    def set_initial_qpos(self, actuators: dict, joints: dict) -> None:
        """Set the initial joint positions of the robot."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the controller state."""
        raise NotImplementedError
    
    def apply_action(self, action_cmd) -> None:
        """Apply the given action to the controller."""
        raise NotImplementedError
        
        
