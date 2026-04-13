"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
import os
from simple.core.robot import Robot
from simple.core.controller import Controller
# from simple.robots.controllers import PDJointPosControllerCfg
# from simple.robots.controllers import ControllerCfg
from simple.core.controller import ControllerCfg
from simple.robots.controllers.combo import SingleArmBinaryEEFControllerCfg   
from simple.robots.controllers.eef import BinaryEEFControllerCfg, ParallelGripperEEFControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.registry import RobotRegistry
from typing import Any, List

from simple.robots.franka import FrankaMixin
from simple.utils import resolve_res_path, resolve_data_path, load_yaml

import numpy as np
import transforms3d as t3d

@RobotRegistry.register("franka_fr3")
class FrankaResearch3(FrankaMixin, Robot):

    uid: str = "franka_fr3"
    label: str = "Franka Research 3"
    dof: int = 9
    FRANKA_FINGER_LENGTH: float = 0.1034 # same as panda
    robot_cfg: Any = "robots/franka_fr3/curobo/franka_fr3.yml"
    mjcf_path: str = "robots/franka_fr3/panda.xml"
    usd_path: str = "robots/franka_fr3/FR3.usd"
    

    robot_ns: str = "FR3"
    hand_prim_path: str = "fr3_hand"
    eef_prim_path: str = "fr3_hand_tcp"
    wrist_cam_link: str = "fr3_hand"
    
    # supported_controllers: dict[str, ControllerCfg] = dict(
    #     pd_joint_pos=PDJointPosControllerCfg(
    #         # self,
    #     ),
    # )

    wrist_camera_orientation: List[float] =  [0.0, 0.64278, 0, 0.766]
    init_joint_states: dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.3,
        "joint3": 0.0,
        "joint4": -2.5,
        "joint5": 0.0,
        "joint6": 1.0,
        "joint7": 0.0,
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    }

    robot_eef_offset: float = 0.1034 # same as panda
    is_single_arm:bool = True
    visulize_spheres: bool = False

    controller_cfg: ControllerCfg = SingleArmBinaryEEFControllerCfg(
        arm=PDJointPosControllerCfg(
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            init_qpos=[0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0],
        ),
        eef=ParallelGripperEEFControllerCfg(
            joint_names=["finger_joint1", "finger_joint2"],
            init_qpos=[0.04, 0.04],
        ),
    )

    # _controller: Controller | None = None

    def __init__(self):
        super().__init__(self.uid, self.dof)


    def fk(self, qpos):
        """  
        Args:
            qpos: list of joint angles
        Returns:
            p: position of end effector
            q: quaternion of end effector
        """
        assert len(qpos) >= 7, "are you kidding?"
        if isinstance(qpos, dict):
            qpos = list(qpos.values())[:7]
        else:
            qpos = qpos[:7]
        dh_params = [[0, 0.333, 0, qpos[0]],
                [0, 0, -np.pi/2, qpos[1]],
                [0, 0.316, np.pi/2, qpos[2]],
                [0.0825, 0, np.pi/2, qpos[3]],
                [-0.0825, 0.384, -np.pi/2, qpos[4]],
                [0, 0, np.pi/2, qpos[5]],
                [0.088, 0, np.pi/2, qpos[6]],
                [0, 0.107, 0, 0],
                [0, 0, 0, -np.pi/4],
                # [0.0, 0.006, 0, 0] for extended finger, i observe this value by debugging curobo
                [0.0, 0.000, 0, 0] # for original finger
                ]
        T = np.eye(4)
        for i in range(7 + 3):
            T = T @ self.get_tf_mat(i, dh_params)
        
        p = T[:3, 3]
        q = t3d.quaternions.mat2quat(T[:3, :3])
        """ # safe test
        p1, q1 = super(Franka, self).fk(qpos)
        assert np.allclose(p, p1), f"p: {p}, p1: {p1}"
        assert np.allclose(q, q1), f"q: {q}, q1: {q1}" """
        return p.tolist(), q.tolist()
    
    def get_tf_mat(self, i, dh):
        # TODO move to fh fk
        a = dh[i][0]
        d = dh[i][1]
        alpha = dh[i][2]
        theta = dh[i][3]
        q = theta

        return np.array([[np.cos(q), -np.sin(q), 0, a],
                        [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                        [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                        [0, 0, 0, 1]])
    
    def get_robot_qpos(self) -> dict[str,float]:
        """ Get the current joint positions of the robot. """
        return {j: v.qpos[0] for j,v in self.joints.items()}
    
    def jname_mujoco_to_isaac(self, mujoco_joint_name) -> str:
        """Map mujoco joint name to robot joint name."""
        return "fr3_" + mujoco_joint_name
    
    def get_eef_pose_from_hand_pose(self, p, q):
        return p + t3d.quaternions.rotate_vector([0,0, self.FRANKA_FINGER_LENGTH], q), q

    def get_hand_pose_from_eef_pose(self, p, q, hack=0):
        EEF_TO_HAND = np.array([0., 0., -(self.FRANKA_FINGER_LENGTH-hack)]) #  FIXME
        p = p + t3d.quaternions.rotate_vector(EEF_TO_HAND, q)
        return p, q