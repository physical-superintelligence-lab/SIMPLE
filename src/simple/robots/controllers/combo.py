"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""


from simple.core.controller import Controller
# from simple.robots.controllers.config import ControllerCfg
from simple.core.controller import ControllerCfg
from .base import BaseController
# from .arm import ArmPDJointPosController
from .eef import BinaryEEFController, BinaryEEFControllerCfg, ParallelGripperEEFController, ParallelGripperEEFControllerCfg,DexHandEEFController, DexHandEEFControllerCfg
from .qpos import PDJointPosController, PDJointPosControllerCfg
from typing import Type, Optional
from gymnasium import spaces
import numpy as np

class SingleArmBinaryEEFController(Controller): 
    arm: PDJointPosController
    eef: BinaryEEFController

    def __init__(self, cfg: "SingleArmBinaryEEFControllerCfg"):
        super().__init__(cfg)
        self.arm = PDJointPosController(cfg.arm_cfg) 
        self.eef = cfg.eef_cfg() #BinaryEEFController(cfg.eef_cfg)

    @property
    def action_space(self):
        return spaces.Dict({
            "arm": self.arm.action_space,
        })
    
    @property
    def eef_action_space(self) -> spaces.Space:
        return self.eef.action_space
    
    def set_initial_qpos(self, actuators:dict, joints: dict) -> None:
        self.arm.set_initial_qpos(actuators, joints)
        self.eef.set_initial_qpos(actuators, joints)
    
class SingleArmBinaryEEFControllerCfg(ControllerCfg):
    clazz: Type[Controller] = SingleArmBinaryEEFController

    def __init__(
        self, 
        arm: PDJointPosControllerCfg, 
        eef: BinaryEEFControllerCfg, 
    ):
        self.arm_cfg = arm
        self.eef_cfg = eef


class BaseSingleArmBinaryEEFController(Controller): 
    base: PDJointPosController
    arm: PDJointPosController
    eef: BinaryEEFController

    def __init__(self, controller_cfg: "BaseSingleArmBinaryEEFControllerCfg"):
        super().__init__(controller_cfg)
        self.base = PDJointPosController(controller_cfg.base_cfg)
        self.arm = PDJointPosController(controller_cfg.arm_cfg) 
        self.eef = BinaryEEFController(controller_cfg.eef_cfg)

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict({
            "base": self.base.action_space,
            "arm": self.arm.action_space,
            "eef": self.eef.action_space
        })
        # base_action_space = self.base.action_space
        # arm_action_space = self.arm.action_space
        # eef_action_space = self.eef.action_space

    @property
    def eef_action_space(self) -> spaces.Space:
        return spaces.Dict({
            "eef": self.eef.action_space,
        }) 

    # @property
    # def init_joint_states(self) -> list[float]:
    #     return np.concatenate([
    #         self.base.init_joint_states,
    #         self.arm.init_joint_states,
    #         self.eef.init_joint_states
    #     ]).tolist()

    
    def set_initial_qpos(self, actuators:dict, joints: dict) -> None:
        self.base.set_initial_qpos(actuators, joints)
        self.arm.set_initial_qpos(actuators, joints)
        self.eef.set_initial_qpos(actuators, joints)


class DualArmBinaryEEFController(Controller):
    left_arm: PDJointPosController
    right_arm: PDJointPosController
    left_eef: BinaryEEFController
    right_eef: BinaryEEFController

    def __init__(self, controller_cfg):
        self.left_arm = PDJointPosController(controller_cfg.left_arm_cfg)
        self.right_arm = PDJointPosController(controller_cfg.right_arm_cfg)
        # self.left_eef = BinaryEEFController(controller_cfg.left_eef_cfg)
        # self.right_eef = BinaryEEFController(controller_cfg.right_eef_cfg)

        self.left_eef = controller_cfg.left_eef_cfg()
        self.right_eef = controller_cfg.right_eef_cfg()

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict({
            "left_arm": self.left_arm.action_space,
            "right_arm": self.right_arm.action_space,
            "left_eef": self.left_eef.action_space,
            "right_eef": self.right_eef.action_space
        })
    
    @property
    def eef_action_space(self) -> spaces.Space:
        return spaces.Dict({
            "left_eef": self.left_eef.action_space,
            "right_eef": self.right_eef.action_space
        })     
    
    def set_initial_qpos(self, actuators:dict, joints: dict) -> None:
        self.left_arm.set_initial_qpos(actuators, joints)
        self.right_arm.set_initial_qpos(actuators, joints)
        self.left_eef.set_initial_qpos(actuators, joints)
        self.right_eef.set_initial_qpos(actuators, joints)

class DualArmBinaryEEFControllerCfg(ControllerCfg):
    clazz: Type[Controller] = DualArmBinaryEEFController

    def __init__(
        self, 
        waist: Optional[PDJointPosControllerCfg],
        left_arm: PDJointPosControllerCfg, 
        right_arm: PDJointPosControllerCfg, 
        left_eef: BinaryEEFControllerCfg, 
        right_eef: BinaryEEFControllerCfg, 
    ):
        if waist is not None:
            self.waist_cfg = waist
        else:
            self.waist_cfg = None
        self.left_arm_cfg = left_arm
        self.right_arm_cfg = right_arm
        self.left_eef_cfg = left_eef
        self.right_eef_cfg = right_eef

class DualArmDexEEFController(Controller):
    left_arm: PDJointPosController
    right_arm: PDJointPosController
    left_eef: DexHandEEFController
    right_eef: DexHandEEFController

    def __init__(self, controller_cfg):
        self.left_arm = PDJointPosController(controller_cfg.left_arm_cfg)
        self.right_arm = PDJointPosController(controller_cfg.right_arm_cfg)
        self.left_eef = DexHandEEFController(controller_cfg.left_eef_cfg)
        self.right_eef = DexHandEEFController(controller_cfg.right_eef_cfg)

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict({
            "left_arm": self.left_arm.action_space,
            "right_arm": self.right_arm.action_space,
            "left_eef": self.left_eef.action_space,
            "right_eef": self.right_eef.action_space
        })
    def set_initial_qpos(self, actuators:dict, joints: dict) -> None:
        self.left_arm.set_initial_qpos(actuators, joints)
        self.right_arm.set_initial_qpos(actuators, joints)
        self.left_eef.set_initial_qpos(actuators, joints)
        self.right_eef.set_initial_qpos(actuators, joints)

class DualArmDexEEFControllerCfg(ControllerCfg):
    clazz: Type[Controller] = DualArmDexEEFController

    def __init__(
        self, 
        waist: Optional[PDJointPosControllerCfg],
        left_arm: PDJointPosControllerCfg, 
        right_arm: PDJointPosControllerCfg, 
        left_eef: DexHandEEFControllerCfg, 
        right_eef: DexHandEEFControllerCfg, 
    ):
        if waist is not None:
            self.waist_cfg = waist
        else:
            self.waist_cfg = None
        self.left_arm_cfg = left_arm
        self.right_arm_cfg = right_arm
        self.left_eef_cfg = left_eef
        self.right_eef_cfg = right_eef


class DualArmController(Controller):
    waist: PDJointPosController
    left_arm: PDJointPosController
    right_arm: PDJointPosController

    def __init__(self, controller_cfg):
        self.waist = PDJointPosController(controller_cfg.waist_cfg)
        self.left_arm = PDJointPosController(controller_cfg.left_arm_cfg)
        self.right_arm = PDJointPosController(controller_cfg.right_arm_cfg)
    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict({
            "waist": self.waist.action_space,
            "left_arm": self.left_arm.action_space,
            "right_arm": self.right_arm.action_space
        })
    def set_initial_qpos(self, actuators:dict, joints: dict) -> None:
        self.waist.set_initial_qpos(actuators, joints)
        self.left_arm.set_initial_qpos(actuators, joints)
        self.right_arm.set_initial_qpos(actuators, joints)

class DualArmControllerCfg(ControllerCfg):
    clazz: Type[Controller] = DualArmController

    def __init__(
        self, 
        waist: PDJointPosControllerCfg,
        left_arm: PDJointPosControllerCfg, 
        right_arm: PDJointPosControllerCfg, 
    ):
        self.waist_cfg = waist
        self.left_arm_cfg = left_arm
        self.right_arm_cfg = right_arm

class BaseDualArmDexEEFController(Controller):
    cfg: "BaseDualArmDexEEFControllerCfg"
    waist: PDJointPosController
    left_arm: PDJointPosController
    right_arm: PDJointPosController
    left_eef: DexHandEEFController
    right_eef: DexHandEEFController

    def __init__(self, cfg: "BaseDualArmDexEEFControllerCfg", **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        if cfg.waist_cfg is not None:
            self.waist = cfg.waist_cfg()
        self.left_arm = cfg.left_arm_cfg()
        self.right_arm = cfg.right_arm_cfg()
        self.left_eef = cfg.left_eef_cfg()
        self.right_eef = cfg.right_eef_cfg()

    def set_initial_qpos(self, actuators:dict, joints: dict) -> None:
        if self.cfg.waist_cfg is not None:
            self.waist.set_initial_qpos(actuators, joints)
        self.left_arm.set_initial_qpos(actuators, joints)
        self.right_arm.set_initial_qpos(actuators, joints)
        self.left_eef.set_initial_qpos(actuators, joints)
        self.right_eef.set_initial_qpos(actuators, joints)

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict({
            # "base": self.base.action_space,
            "waist": self.waist.action_space,
            "left_arm": self.left_arm.action_space,
            "right_arm": self.right_arm.action_space,
            "left_eef": self.left_eef.action_space,
            "right_eef": self.right_eef.action_space
        })

class BaseDualArmDexEEFControllerCfg(ControllerCfg):
    clazz: Type[Controller] = BaseDualArmDexEEFController

    def __init__(
        self, 
        waist: Optional[PDJointPosControllerCfg],
        left_arm: PDJointPosControllerCfg, 
        right_arm: PDJointPosControllerCfg, 
        left_eef: DexHandEEFControllerCfg, 
        right_eef: DexHandEEFControllerCfg, 
    ):
        if waist is not None:
            self.waist_cfg = waist
        else:
            self.waist_cfg = None
        self.left_arm_cfg = left_arm
        self.right_arm_cfg = right_arm
        self.left_eef_cfg = left_eef
        self.right_eef_cfg = right_eef

class WholeBodyEEFController(Controller):
    cfg: "WholeBodyEEFControllerCfg"
    left_leg: PDJointPosController
    right_leg: PDJointPosController
    waist: PDJointPosController
    left_arm: PDJointPosController
    right_arm: PDJointPosController
    left_eef: DexHandEEFController
    right_eef: DexHandEEFController

    def __init__(self, cfg: "WholeBodyEEFControllerCfg", **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.left_leg = cfg.left_leg_cfg()
        self.right_leg = cfg.right_leg_cfg()
        if cfg.waist_cfg is not None:
            self.waist = cfg.waist_cfg()
        self.left_arm = cfg.left_arm_cfg()
        self.right_arm = cfg.right_arm_cfg()
        self.left_eef = cfg.left_eef_cfg()
        self.right_eef = cfg.right_eef_cfg()

    def set_initial_qpos(self, actuators:dict, joints: dict) -> None:
        self.left_leg.set_initial_qpos(actuators, joints)
        self.right_leg.set_initial_qpos(actuators, joints)  
        if self.cfg.waist_cfg is not None:
            self.waist.set_initial_qpos(actuators, joints)
        self.left_arm.set_initial_qpos(actuators, joints)
        self.right_arm.set_initial_qpos(actuators, joints)
        self.left_eef.set_initial_qpos(actuators, joints)
        self.right_eef.set_initial_qpos(actuators, joints)

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict([
            ("left_leg", self.left_leg.action_space),
            ("right_leg", self.right_leg.action_space),
            ("waist", self.waist.action_space),
            ("left_arm", self.left_arm.action_space),
            ("right_arm", self.right_arm.action_space),
            ("left_eef", self.left_eef.action_space),
            ("right_eef", self.right_eef.action_space)
        ])

class WholeBodyEEFControllerCfg(ControllerCfg):
    clazz: Type[Controller] = WholeBodyEEFController

    def __init__(
        self, 
        left_leg: PDJointPosControllerCfg,
        right_leg: PDJointPosControllerCfg,
        waist: Optional[PDJointPosControllerCfg],
        left_arm: PDJointPosControllerCfg, 
        right_arm: PDJointPosControllerCfg, 
        left_eef: DexHandEEFControllerCfg, 
        right_eef: DexHandEEFControllerCfg, 
    ):
        self.left_leg_cfg = left_leg
        self.right_leg_cfg = right_leg
        if waist is not None:
            self.waist_cfg = waist
        else:
            self.waist_cfg = None
        self.left_arm_cfg = left_arm
        self.right_arm_cfg = right_arm
        self.left_eef_cfg = left_eef
        self.right_eef_cfg = right_eef


class BaseSingleArmBinaryEEFControllerCfg(ControllerCfg):
    clazz: Type[Controller] = BaseSingleArmBinaryEEFController

    def __init__(
        self, 
        base: PDJointPosControllerCfg,
        arm: PDJointPosControllerCfg, 
        eef: BinaryEEFControllerCfg, 
    ):
        self.base_cfg = base
        self.arm_cfg = arm
        self.eef_cfg = eef
