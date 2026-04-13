"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .controller import Controller
    from simple.core.controller import ControllerCfg

from typing import List
# from simple.core.actor import Actor
import numpy as np
import transforms3d as t3d
# from simple.mp.curobo import CuRoboPlanner

class Robot: #(Actor)
    uid: str
    label: str
    dof: int

    FRANKA_FINGER_LENGTH: float
    robot_cfg: dict | str

    curobo_usd: str
    mjcf_path: str
    usd_path: str

    robot_ns: str
    hand_prim_path: str

    eef_link_name: str
    eef_prim_path: str
    # eef_prim_path: str

    robot_eef_offset: float

    # TODO merge into single config
    # wrist_cam_link: str
    wrist_camera_orientation: List[float]

    joint_names: List[str]
    init_joint_states: dict[str, float]
    joint_limits: dict[str, tuple[float, float]] = {}
    # supported_controllers: dict[str, ControllerCfg]
    controller_cfg: ControllerCfg

    visulize_spheres : bool

    # pregrasp_distance: List[float] = [0.05, 0.08]

    def __init__(
        self,
        uid: str,
        dof: int,
    ) -> None:
        self.uid = uid
        self.dof = dof
        # self.kin_model = None # lazy init
        self.planner = None
        self.ik_solver = None

    def jname_mujoco_to_isaac(self, mujoco_joint_name) -> str:
        """Map mujoco joint name to robot joint name."""
        return mujoco_joint_name
    
    def fk(self, qpos):
        raise NotImplementedError("fk not implemented")

    def ik(self, p, q, current_joint = None):
        raise NotImplementedError("ik not implemented")
    
    def reset(self, **kwargs):
        """Reset the robot to initial state."""
        ...
