"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import Union, Sequence, Type
from simple.core.controller import Controller, ControllerCfg
from simple.core.robot import Robot
from gymnasium import spaces
import numpy as np

# from simple.robots.controllers.config import ControllerCfg, PDJointPosControllerCfg


class PDJointPosController(Controller):
    """
    PD Joint Position Controller for a robot.
    
    This controller uses a proportional-derivative (PD) control strategy to manage joint positions.
    It is designed to work with robots that have joint position control capabilities.
    """

    dof: int

    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]

    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    
    def __init__(
        self, 
        cfg: "PDJointPosControllerCfg",
        robot: Robot | None= None,  # FIXME
        stiffness: Union[float, Sequence[float]] = 1000.0, 
        damping: Union[float, Sequence[float]] = 200,
        **kwargs
    ) -> None:
        """
        Initializes the PDJointPosController with a robot and control parameters.
        """
        # self.robot = robot
        self.cfg = cfg
        self.dof = cfg.dof

        self.stiffness = stiffness
        self.damping = damping

        # FIXME read joint limits from robot
        self.lower = -1
        self.upper = 1

    @property
    def action_space(self) -> spaces.Space:
        """
        Returns the action space of the controller.
        
        The action space is defined as a continuous space with bounds for each joint.
        """
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        return spaces.Box(low, high, dtype=np.float32)

    def _get_joint_limits(self):
        # FIXME read joint limits from robot
        # qlimits = (
        #     self.articulation.get_qlimits()[0, self.active_joint_indices].cpu().numpy()
        # )
        # Override if specified
        # if self.lower is not None:
        #     qlimits[:, 0] = self.lower
        # if self.upper is not None:
        #     qlimits[:, 1] = self.upper
        # return qlimits
        
        return np.array([[-1.0, 1.0]] * self.dof, dtype=np.float32)

    def setup(self, joints, actuators) -> None:
        self.joints = joints
        self.actuators = actuators

    def set_initial_qpos(self, actuators: dict, joints: dict) -> None:
        for jname, qpos in zip(self.cfg.joint_names, self.cfg.init_qpos):
            joints[jname].qpos = qpos
            joints[jname].qvel = 0
            joints[jname].qacc = 0
            # if act[0] < 7:
            actuators[jname].ctrl = qpos

        # # open gripper by default
        # self.actuators[7].ctrl = 0.0205
        # self.actuators[8].ctrl = 0.0205

    @property
    def init_joint_states(self) -> list[float]:
        return self.cfg.init_qpos

class PDJointPosControllerCfg(ControllerCfg):
    clazz: Type[Controller] = PDJointPosController

    def __init__(self, joint_names: list[str], init_qpos: list[float]):
        self.joint_names = joint_names
        self.dof = len(joint_names)
        self.init_qpos = init_qpos

    def __call__(self, *args, **kwargs) -> PDJointPosController:
        return PDJointPosController(self, *args, **kwargs) #  self.joint_names, self.init_qpos,

    # @property
    # def dof(self) -> int:
    #     return len(self.joint_names)