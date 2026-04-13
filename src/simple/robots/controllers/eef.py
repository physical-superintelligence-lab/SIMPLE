"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""


from typing import Any
from simple.core.controller import ControllerCfg
from simple.core.controller import Controller
from gymnasium import spaces
import math
import numpy as np

class BinaryEEFController(Controller):
    cfg: "BinaryEEFControllerCfg"

    def __init__(self, cfg: "BinaryEEFControllerCfg",  joint_names: list[str], init_qpos: list[float]):
        super().__init__(cfg)
        self.joint_names = joint_names
        self.init_qpos = init_qpos

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @property
    def action_space(self) -> spaces.Space:
        return spaces.MultiBinary(1)
    
    def open_gripper(self, actuators: dict) -> None:
        raise NotImplementedError

    def close_gripper(self, actuators: dict) -> None:
        raise NotImplementedError

class BinaryEEFControllerCfg(ControllerCfg):
    clazz: type[Controller] = BinaryEEFController

    def __init__(self, joint_names: list[str], init_qpos: list[float]):
        # super().__init__(cfg)
        # self.eef_link = eef_link
        # self.init_offset = init_offset
        self.joint_names = joint_names
        self.init_qpos = init_qpos

    def __call__(self) -> BinaryEEFController:
        return BinaryEEFController(self, self.joint_names, self.init_qpos)
    

class ParallelGripperEEFController(BinaryEEFController):
    def __init__(self, cfg, joint_names: list[str], init_qpos: list[float]):
        super().__init__(cfg, joint_names, init_qpos)

    def set_initial_qpos(self, actuators: dict, joints: dict) -> None:
        """Set the initial joint positions of the eed effector."""
        for jname, qpos in zip(self.cfg.joint_names, self.cfg.init_qpos):
            joints[jname].qpos = qpos
            joints[jname].qvel = 0
            joints[jname].qacc = 0

            actuators[jname].ctrl = qpos
            # actuators[jname].ctrl = 0.0205

    def open_gripper(self, actuators: dict) -> None:
        for jname in self.cfg.joint_names:
            actuators[jname].ctrl = 0.0205

    def close_gripper(self, actuators: dict) -> None:
        for jname in self.cfg.joint_names:
            actuators[jname].ctrl = 0.


class ParallelGripperEEFControllerCfg(BinaryEEFControllerCfg):
    clazz: type[Controller] = ParallelGripperEEFController

    def __call__(self) -> ParallelGripperEEFController:
        return ParallelGripperEEFController(self, self.joint_names, self.init_qpos)
    
    # @property
    # def init_joint_states(self) -> list[float]:
    #     return []

class DexHandEEFController(Controller):
    cfg: "DexHandEEFControllerCfg"
    def __init__(self, cfg: "DexHandEEFControllerCfg"):
        super().__init__(cfg)
        # self.cfg = cfg

    @property
    def action_space(self) -> spaces.Space:
        dof = len(self.cfg.joint_names)
        joint_limits = np.array([[-np.pi, np.pi]]*dof, dtype=np.float32)
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        return spaces.Box(low, high, dtype=np.float32)

    def set_initial_qpos(self, actuators: dict, joints: dict) -> None:
        for jname, qpos in zip(self.cfg.joint_names, self.cfg.init_qpos):
            joints[jname].qpos = qpos
            joints[jname].qvel = 0
            joints[jname].qacc = 0
            actuators[jname].ctrl = qpos

    def open_gripper(self, actuators: dict) -> None:
        # step_distance = 0.15
        # for jname in self.cfg.joint_names:
        #     current = actuators[jname].ctrl

        #     if current > 0:
        #         new = max(0.0, current - step_distance)
        #     else:
        #         new = min(0.0, current + step_distance)

        #     actuators[jname].ctrl = new
        for jname in self.cfg.joint_names:
            actuators[jname].ctrl = 0.0


    def close_gripper(self, actuators: dict) -> None:
        ctrl=self.cfg.close_qpos
        for jname, c in zip(self.cfg.joint_names, ctrl):
            actuators[jname].ctrl = c

class DexHandEEFControllerCfg(ControllerCfg):
    clazz: type[Controller] = DexHandEEFController

    def __init__(self, joint_names: list[str], init_qpos: list[float], close_qpos: list[float]):
        self.joint_names = joint_names
        self.init_qpos = init_qpos
        self.close_qpos = close_qpos

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # return super().__call__(*args, **kwds)
        return DexHandEEFController(self)