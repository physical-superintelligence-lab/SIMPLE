"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from typing import Tuple, List, Any
from simple.core.action import ActionCmd
from simple.core.robot import Robot
from simple.core.controller import Controller
# from simple.robots.controllers import ControllerCfg
from simple.core.controller import ControllerCfg
from simple.robots.controllers.config import *
from simple.robots.controllers.combo import BaseDualArmDexEEFController
from simple.robots.controllers.combo import BaseDualArmDexEEFControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.controllers.eef import DexHandEEFControllerCfg
from simple.robots.protocols import *
from simple.robots.registry import RobotRegistry
from simple.robots.mixin import CuRoboMixin
from simple.utils import resolve_res_path, resolve_data_path, load_yaml

import os
import numpy as np
from torch._tensor import Tensor
import transforms3d as t3d
import torch


@RobotRegistry.register("vega_1")
class Vega1(CuRoboMixin, Humanoid, WristCamMountable, Robot, HasDexterousHand, HeadCamMountable):
    uid: str = "vega_1"
    label: str = "Dexmate Vega 1"
    dof: int = 39   # 29 + 10 (mimic joints)
    
    robot_cfg: Any = "robots/vega_1/curobo/vega1_with_dexmate.yml"
    mjcf_path: str = "robots/vega_1/vega.xml"
    usd_path: str = "robots/vega_1/vega_1.usd"

    robot_ns: str = "vega_1"
    hand_yaml: str = "robots/vega_1/curobo/dexmate_right.yml"
    hand_uid: str = "dexmate_right"
    hand_dof: int = 6
    pregrasp_distance: List[float] = [0.05, 0.08]

    wrist_cam_link: str = "R_arm_l7"    # Overwritten by *_wrist_cam_link properties

    # *_arm_l8 and *_hand_base does NOT correctly follow the wrist movement
    left_wrist_cam_link: str = "L_arm_l7"
    right_wrist_cam_link: str = "R_arm_l7"
    
    wrist_camera_orientation: List[float] = [1, 0, 0, 0]
    head_camera_orientation: List[float] =[1, 0, 0, 0]

    robot_eef_offset: float = 0.0
    # is_single_arm: bool = False # FIXME
    visulize_spheres: bool = False

    ee_joint: str = "R_arm_j7"
    eef_prim_path: str = "R_ee"
    hand_prim_path: str = "R_hand_base"

    z_offset: float = +0.055 # to avoid ground collision with wheels

    @property
    def head_cam_link(self) -> str:
        """ Get the head camera link name. """
        return "head_l3"

    joint_limits: dict[str, tuple[float, float]] = {}
    
    init_joint_states: dict[str, float] = {
        "torso_j1": 0.0,
        "torso_j2": 0.0,
        "torso_j3": 0.0,
        
        # "L_arm_j1": 1.57,
        "L_arm_j1": 0.0,
        "L_arm_j2": 0.0,
        "L_arm_j3": 0.0,
        # "L_arm_j4": -1.57,
        "L_arm_j4": 0.0,
        "L_arm_j5": 0.0,
        "L_arm_j6": 0.0,
        "L_arm_j7": 0.0,
        
        # "R_arm_j1": -1.57,
        "R_arm_j1": 0.0,
        "R_arm_j2": 0.0,
        "R_arm_j3": 0.0,
        # "R_arm_j4": -1.57,
        "R_arm_j4": 0.0,
        "R_arm_j5": 0.0,
        "R_arm_j6": 0.0,
        "R_arm_j7": 0.0,

        "L_th_j0": 1.5,
        "L_th_j1": 0.15,
        "L_ff_j1": 0.0,
        "L_mf_j1": 0.0,
        "L_rf_j1": 0.0,
        "L_lf_j1": 0.0,

        "R_th_j0": 1.5,
        "R_th_j1": 0.15,
        "R_ff_j1": 0.0,
        "R_mf_j1": 0.0,
        "R_rf_j1": 0.0,
        "R_lf_j1": 0.0,

        # Mimic joints
        "L_th_j2": 0.0,
        "L_ff_j2": 0.0,
        "L_mf_j2": 0.0,
        "L_rf_j2": 0.0,
        "L_lf_j2": 0.0,

        "R_th_j2": 0.0,
        "R_ff_j2": 0.0,
        "R_mf_j2": 0.0,
        "R_rf_j2": 0.0,
        "R_lf_j2": 0.0,
    }

    joint_names=[
        # "head_j1", 
        # "head_j2", 
        # "head_j3",

        "torso_j1", 
        "torso_j2", 
        "torso_j3", 
        
        "L_arm_j1", 
        "L_arm_j2", 
        "L_arm_j3", 
        "L_arm_j4", 
        "L_arm_j5", 
        "L_arm_j6", 
        "L_arm_j7",
        
        "R_arm_j1", 
        "R_arm_j2", 
        "R_arm_j3", 
        "R_arm_j4", 
        "R_arm_j5", 
        "R_arm_j6", 
        "R_arm_j7",
        
        "L_th_j0", 
        "L_th_j1", 
        "L_ff_j1", 
        "L_mf_j1", 
        "L_rf_j1", 
        "L_lf_j1",
        
        "R_th_j0", 
        "R_th_j1", 
        "R_ff_j1", 
        "R_mf_j1", 
        "R_rf_j1", 
        "R_lf_j1",

        # Mimic joints
        "L_th_j2", 
        "L_ff_j2", 
        "L_mf_j2", 
        "L_rf_j2", 
        "L_lf_j2",
        
        "R_th_j2", 
        "R_ff_j2", 
        "R_mf_j2", 
        "R_rf_j2", 
        "R_lf_j2",
    ]

    # Define mimic joint mappings: {mimic_joint: (primary_joint, multiplier, offset)}
    mimic_joints = {
        # Left hand
        "L_th_j2": ("L_th_j1", 1.35316, 0.00765),
        "L_ff_j2": ("L_ff_j1", 1.13028, -0.00053),
        "L_mf_j2": ("L_mf_j1", 1.13311, -0.00079),
        "L_rf_j2": ("L_rf_j1", 1.12935, 0.00065),
        "L_lf_j2": ("L_lf_j1", 1.15037, 0.00186),
        
        # Right hand
        "R_th_j2": ("R_th_j1", 1.35316, 0.00765),
        "R_ff_j2": ("R_ff_j1", 1.13028, -0.00053),
        "R_mf_j2": ("R_mf_j1", 1.13311, -0.00079),
        "R_rf_j2": ("R_rf_j1", 1.12935, 0.00065),
        "R_lf_j2": ("R_lf_j1", 1.15037, 0.00186),
    }

    # controller_cfg: dict[str, ControllerCfg] = dict(
    #     arm=PDJointPosControllerCfg(),
    #     eef=BinaryEEFControllerCfg(),
    # )

    controller_cfg: ControllerCfg = BaseDualArmDexEEFControllerCfg(
        waist = PDJointPosControllerCfg(
            joint_names = ["torso_j1", "torso_j2", "torso_j3"], 
            init_qpos = [0, 0, 0]
        ),
        left_arm = PDJointPosControllerCfg(
            joint_names = ["L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4", "L_arm_j5", "L_arm_j6", "L_arm_j7"], 
            init_qpos = [0.0, 0, 0, 0.0, 0, 0, 0],
            # init_qpos = [1.57, 0, 0, -1.57, 0, 0, 0],
        ),
        right_arm = PDJointPosControllerCfg(
            joint_names = ["R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4", "R_arm_j5", "R_arm_j6", "R_arm_j7"], 
            init_qpos = [0.0, 0, 0, 0.0, 0, 0, 0],
            # init_qpos = [-1.57, 0, 0, -1.57, 0, 0, 0],
        ),
        left_eef = DexHandEEFControllerCfg(
            joint_names = ["L_th_j0", "L_th_j1", "L_ff_j1", "L_mf_j1", "L_rf_j1", "L_lf_j1",
                           "L_th_j2", "L_ff_j2", "L_mf_j2", "L_rf_j2", "L_lf_j2",],
            init_qpos = [1.5, 0.1834, 0.2891, 0.2801, 0.2840, 0.2811,
                         0.2731, 0.3681, 0.3533, 0.3599, 0.4014],
            close_qpos = [0.963, -0.208, -0.6568, -0.6506, -0.6092, -0.6071,
                          -0.282, -0.744, -0.738, -0.69, -0.702],   # 60% of max limit
        ),
        right_eef = DexHandEEFControllerCfg(
            joint_names = ["R_th_j0", "R_th_j1", "R_ff_j1", "R_mf_j1", "R_rf_j1", "R_lf_j1",
                           "R_th_j2", "R_ff_j2", "R_mf_j2", "R_rf_j2", "R_lf_j2",],
            init_qpos = [1.5, 0.1834, 0.2891, 0.2801, 0.2840, 0.2811,
                         0.2731, 0.3681, 0.3533, 0.3599, 0.4014],
            close_qpos = [0.963, -0.208, -0.6568, -0.6506, -0.6092, -0.6071,
                          -0.282, -0.744, -0.738, -0.69, -0.702],   # 60% of max limit
        )
    )

    # @property
    # def eef_prim_path(self) -> str: # type: ignore
    #     return self._eef_prim_path

    def __init__(self,):
        super().__init__(self.uid, self.dof)

        # if controller_uid not in self.supported_controllers:
        #     raise ValueError(f"Controller '{controller_uid}' is not supported by {self.label}")
        
        # self.controller_uid = controller_uid
        
        # self.robot_cfg=load_yaml(
        #     resolve_data_path(self.robot_cfg, auto_download=True)
        # )["robot_cfg"]

        # assert isinstance(self.robot_cfg, dict)
        # for key in ["external_asset_path", "external_robot_configs_path"]:
        #     abs_path = self.robot_cfg["kinematics"][key].replace("<PROJECT_DIR>", os.getcwd())
        #     self.robot_cfg["kinematics"][key] = abs_path

        # self._robot_cfg=robot_cfg
        
    # @property
    # def init_joint_states(self) -> List[float]:
    #     return self.controller.init_joint_states
    
    def _apply_mimic_joints(self):
        """Apply mimic joint behavior by copying primary joint values with ratio."""
        for mimic_joint, (primary_joint, ratio, offset) in self.mimic_joints.items():
            if primary_joint in self.actuators and mimic_joint in self.actuators:
                # Get primary joint target
                primary_ctrl = self.actuators[primary_joint].ctrl
                
                # Set mimic joint target
                self.actuators[mimic_joint].ctrl = primary_ctrl * ratio + offset

    def setup_control(self, mjData, mjModel,mjSpec) -> Tuple[dict[str, Any], dict[str, Any]]:
        actuators = {}
        joints = {}

        for name in self.joint_names:
            actuators[name] = mjData.actuator(f'{name}_ctrl')
            joints[name] = mjData.joint(f'{name}')

        self.actuators = actuators
        self.joints = joints

        self.controller.set_initial_qpos(actuators, joints)

        # # set initial posistions
        # self.last_action = [0 for _ in range(self.dof)] # TODO check
        # ctrl_action = list(enumerate(self.init_joint_states))
        # for idx, qpos_val in ctrl_action:
        #     self.joints[idx].qpos = qpos_val
        #     self.joints[idx].qvel = 0
        #     self.joints[idx].qacc = 0

        # save joint limits
        if not self.joint_limits:
            for j in self.joints.values():
                limits = mjModel.jnt_range[j.id]
                self.joint_limits[j.name] = (limits[0], limits[1])
        return joints, actuators

    def get_robot_qpos(self) -> dict[str, float]:
        """ Get the current joint positions of the robot. """
        # return np.array([j.qpos[0] for j in self.joints])
        return {j.name: j.qpos[0] for j in self.joints.values()}

    def apply_action(self, action_cmd) -> None:
        assert isinstance(self.controller, BaseDualArmDexEEFController)
        # target_qpos = np.array(np.concatenate([
        #     target_qpos[:3], target_qpos[3:17:2], target_qpos[4:17:2]
        # ])).tolist()

        if action_cmd.type == "open_eef":
            self.controller.left_eef.open_gripper(self.actuators)
            self.controller.right_eef.open_gripper(self.actuators)
            self._apply_mimic_joints()
            return
        elif action_cmd.type == "close_eef":
            self.controller.left_eef.close_gripper(self.actuators)
            self.controller.right_eef.close_gripper(self.actuators)
            self._apply_mimic_joints()
            return
        else:
            target_qpos = action_cmd.parameters["target_qpos"]
            # assert len(target_qpos) == 17

            for jname, jval in target_qpos.items():
                # if jidx >= 6:
                #     jidx += 1
                self.actuators[jname].ctrl = jval
            # self.last_action[jidx] = jval
            self._apply_mimic_joints()

    def fk(self, qpos) -> tuple[list[float], list[float]]:
        """
            forward kinematics implmented by curobo
            overriden by subclasses which uses simple forward kinematics based on DH parameters
        """
        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        import curobo.util_file
        from curobo.types.math import Pose
        from curobo.wrap.reacher.ik_solver import IKSolver
        from curobo.types.base import TensorDeviceType
        kin_model = self.get_kinematic_model()

        if isinstance(qpos, dict):
            qpos_list = []
            for jname in self.kin_model.joint_names:
                qpos_list.append(qpos[jname])
            qpos = qpos_list
        
        else:
            if type(qpos) is list:
                qpos = np.array(qpos)
            if "right"in self.eef_prim_path:
                qpos=np.concatenate([qpos[:3],qpos[10:17]])
            else :
                qpos=qpos[:10]
            joint_traj = torch.from_numpy(qpos).to(torch.float32)

        assert len(qpos) == len(self.kin_model.joint_names)
        joint_traj = torch.as_tensor(qpos, dtype=torch.float32)
        tensor_args = TensorDeviceType()
        joint_traj = tensor_args.to_device(joint_traj).unsqueeze(0)
        out = self.kin_model.get_state(joint_traj)
        hand_pose = out.ee_position[0].cpu().numpy(), out.ee_quaternion[0].cpu().numpy()
        return hand_pose

    @property
    def kin_model(self):
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
        from curobo.types.robot import RobotConfig
        from curobo.types.base import TensorDeviceType
        from curobo.util_file import join_path, get_robot_configs_path
        if type(self.robot_cfg) == str:
            cfg = load_yaml(join_path(get_robot_configs_path(), self.robot_cfg))
            curobo_robot_cfg = cfg["robot_cfg"]
        elif type(self.robot_cfg) == dict:
            curobo_robot_cfg = self.robot_cfg

        robot_config = RobotConfig.from_dict(curobo_robot_cfg)
        kin_model = CudaRobotModel(robot_config.kinematics)
        return kin_model

    def get_kinematic_model(self): # TODO DO NOT PUBLIC THIS FUNCTION
        if self.kin_model is None:
            from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
            from curobo.types.robot import RobotConfig
            from curobo.types.base import TensorDeviceType
            from curobo.util_file import join_path, get_robot_configs_path
            if type(self.robot_cfg) == str:
                cfg = load_yaml(join_path(get_robot_configs_path(), self.robot_cfg))
                curobo_robot_cfg = cfg["robot_cfg"]
            elif type(self.robot_cfg) == dict:
                curobo_robot_cfg = self.robot_cfg
            
            urdf_file = curobo_robot_cfg["kinematics"]["urdf_path"]
            base_link = curobo_robot_cfg["kinematics"]["base_link"]
            ee_link = curobo_robot_cfg["kinematics"]["ee_link"]
            
            if ("external_asset_path" in curobo_robot_cfg["kinematics"] and curobo_robot_cfg["kinematics"]["external_asset_path"] is not None):
                urdf_file = join_path(curobo_robot_cfg["kinematics"]["external_asset_path"], urdf_file)
            
            robot_config = RobotConfig.from_basic(urdf_file, base_link, ee_link, TensorDeviceType())
            return CudaRobotModel(robot_config.kinematics)  
        else:
            return self.kin_model

    def get_grasp_pose_wrt_robot(self, grasp_info: dict, pregrasp: bool=False, robot_pose=None):
        assert robot_pose is not None, "not implemented"
        # world_grasp_mat = np.eye(4)
        # world_grasp_mat[:3, 3] = grasp_info['position']
        # world_grasp_mat[:3, :3] = t3d.quaternions.quat2mat(grasp_info['orientation'])
        # position = world_grasp_mat[:3, 3]
        # orientation = t3d.quaternions.mat2quat(world_grasp_mat[:3, :3])

        # grasp pose
        T_ee_hand = np.eye(4, dtype=np.float32)
        T_ee_hand[:3, 3] = np.array([0, 0, -self.robot_eef_offset], dtype=np.float32) # TODO move to robot class

        T_grasp_ee = np.array([
            [1,0,0,0], 
            [0,1,0,0], 
            [0,0,1,0], 
            [0,0,0,1]
        ], dtype=np.float32) # only for graspnet

        R_world_grasp = t3d.quaternions.quat2mat(grasp_info['orientation'])
        # T_grasp_ee = np.eye(4, dtype=np.float32) # FIXME cube
        T_world_grasp = np.eye(4, dtype=np.float32)
        T_world_grasp[:3, 3] = grasp_info['position'] + grasp_info['depth'] * R_world_grasp[:, 0]
        T_world_grasp[:3, :3] = R_world_grasp # t3d.quaternions.quat2mat(orientation)
        
        if robot_pose is None:
            T_world_robot = np.eye(4, dtype=np.float32)
            T_world_robot[:3, 3] = np.array([0., 0., 0.])
            T_world_robot[:3, :3] = t3d.quaternions.quat2mat(np.array([1., 0., 0., 0.]))
        else:
            T_world_robot = robot_pose

        T_robot_hand = np.linalg.inv(robot_pose) @ T_world_grasp @ T_grasp_ee @ T_ee_hand
        grasp_pos_in_robot = T_robot_hand[:3, 3]
        grasp_ori_in_robot = t3d.quaternions.mat2quat(T_robot_hand[:3, :3])
    
        if pregrasp:
            # pre-grasp pose
            T_grasp_pregrasp = np.eye(4, dtype=np.float32)
            T_grasp_pregrasp[0, 3] = -np.random.uniform(self.pregrasp_distance[0], self.pregrasp_distance[1])
            T_pregrasp_robot_hand = np.linalg.inv(robot_pose) @ T_world_grasp @ T_grasp_pregrasp @ T_grasp_ee @ T_ee_hand
            grasp_pos_in_robot = T_pregrasp_robot_hand[:3, 3]
            grasp_ori_in_robot = t3d.quaternions.mat2quat(T_pregrasp_robot_hand[:3, :3])
        else:
            T_robot_hand = np.linalg.inv(robot_pose) @ T_world_grasp @ T_grasp_ee @ T_ee_hand
            grasp_pos_in_robot = T_robot_hand[:3, 3]
            grasp_ori_in_robot = t3d.quaternions.mat2quat(T_robot_hand[:3, :3])

        return (grasp_pos_in_robot, grasp_ori_in_robot)