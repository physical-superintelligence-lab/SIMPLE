"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast

from simple.robots.protocols import BatchPlannable, Controllable, Graspable, HasKinematics
from simple.utils import load_yaml, resolve_data_path
if TYPE_CHECKING:
    from simple.core.robot import Robot
    # from simple.mp.curobo import CuRoboPlanner

import torch
import numpy as np
# from simple.mp.curobo import CuRoboPlanner

try:
    # import curobo
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver
    from curobo.types.robot import RobotConfig
    from curobo.util_file import join_path
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
except ImportError:
    raise RuntimeError("curobo not installed, uv pip install --groups curobo")


class CuRoboMixin(BatchPlannable, Controllable, Graspable, HasKinematics):

    def __init__(
        self, 
        # default_batch_size: int = 10, 
        *args: Any, 
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self._kin_model = None
        self._ik_solver = None
        # self._controller = None
        # self.plan_batch_size = default_batch_size

        self.robot_cfg=load_yaml(
            resolve_data_path(self.robot_cfg, auto_download=True)
        )["robot_cfg"]

        assert isinstance(self.robot_cfg, dict)
        for key in ["external_asset_path", "external_robot_configs_path"]:
            # abs_path = self.robot_cfg["kinematics"][key].replace("<PROJECT_DIR>", os.getcwd())
            abs_path = resolve_data_path(self.robot_cfg["kinematics"][key])
            self.robot_cfg["kinematics"][key] = abs_path

    @property
    def kin_model(self):
        return self._get_kinematic_model()
    
    def fk(self, qpos)-> tuple[list[float], list[float]]:
        """ forward kinematics implemented by curobo
        @param qpos: list of joint positions
        @return: hand_pose: (position, quaternion)
        """
        if isinstance(qpos, dict):
            qpos_list = []
            for jname in self.kin_model.joint_names:
                if jname in qpos:
                    qpos_list.append(qpos[jname])
            qpos = qpos_list
        else:
            assert len(qpos) == len(self.kin_model.joint_names)
        joint_traj = torch.as_tensor(qpos, dtype=torch.float32)
        tensor_args = TensorDeviceType()
        joint_traj = tensor_args.to_device(joint_traj).unsqueeze(0)
        out = self.kin_model.get_state(joint_traj)
        hand_pose = out.ee_position[0].cpu().numpy(), out.ee_quaternion[0].cpu().numpy()
        return hand_pose
    
    def ik(self, p, q, current_joint = None):
        """
            inverse kinematics implmented by curobo
            required by agents using (delta) eef control such as OpenVLA, Octo, etc.
        """
        ik_solver = self._create_empty_world_ik_solver()
        
        tensor_args = TensorDeviceType()
        p, q = tensor_args.to_device(p), tensor_args.to_device(q)
        goal = Pose(p, q)
        retract_cfg = seed_cfg = None
        if current_joint is not None:
            retract_cfg = torch.tensor(current_joint).cuda().float()
            seed_cfg = torch.tensor(current_joint).float()
            seed_cfg = seed_cfg.unsqueeze(0).repeat(64, 1).cuda().unsqueeze(0)
        result = ik_solver.solve_single(goal, retract_cfg, seed_cfg)
        success = result.success[0, 0]
        if success:
            qpos = result.solution[0,0].cpu().numpy()
            return qpos
        raise RuntimeError("IK failed!")
    
    def planning_init_joint_states(self, batch_size=10) -> np.ndarray:
        """ Get initial joint states for planning. 
        Parameters:
        --------------
        batch_size: int 
            number of samples to generate

        Return
        --------------
            np.ndarray: initial joint states
        """
        robot = cast("Robot", self)
        # return np.asarray(robot.init_joint_states[:3]+robot.init_joint_states[10:17]).reshape(1, 10).repeat(batch_size, axis=0)
        return np.asarray(list(robot.init_joint_states.values())).reshape(1, -1).repeat(batch_size, axis=0)
    
    def planning_init_quats(self, batch_size=10) -> np.ndarray:
        robot = cast("Robot", self)
        _, init_quats = self.fk(robot.init_joint_states) # [:6]+robot.init_joint_states[8:14]
        return np.array(init_quats).reshape(1, 4).repeat(batch_size, 0)
    
    def get_link_pose(self, link_name: str, joint_qpos: dict[str, float] | list[float]) -> list[float]:
        """ Get the pose of a link given joint positions.
        
        Args:
            link_name: Name of the link to get the pose for.
            joint_qpos: Joint positions as a dict or list.
        
        Returns:
            position (3D np.ndarray), orientation (quaternion as 4D np.ndarray)
        """
        robot = cast("Robot", self)
        if not isinstance(joint_qpos, dict):
            joint_qpos = dict(zip(robot.joint_names, joint_qpos))
        
        assert isinstance(robot.robot_cfg, dict)
        urdf_file = robot.robot_cfg["kinematics"]["urdf_path"]
        base_link = robot.robot_cfg["kinematics"]["base_link"]
        if ("external_asset_path" in robot.robot_cfg["kinematics"] and 
                robot.robot_cfg["kinematics"]["external_asset_path"] is not None):
                urdf_file = join_path(robot.robot_cfg["kinematics"]["external_asset_path"], urdf_file)
        robot_config = RobotConfig.from_basic(urdf_file, base_link, link_name, TensorDeviceType())
        kin_model = CudaRobotModel(robot_config.kinematics)
        
        qpos_list = []
        for jname in kin_model.joint_names:
            qpos_list.append(joint_qpos[jname])
        
        joint_traj = torch.as_tensor(qpos_list, dtype=torch.float32)
        tensor_args = TensorDeviceType()
        joint_traj = tensor_args.to_device(joint_traj).unsqueeze(0)
        out = kin_model.get_state(joint_traj)
        hand_pose = out.ee_position[0].cpu().numpy(), out.ee_quaternion[0].cpu().numpy()
        return np.concatenate(hand_pose, axis=0).tolist()
    
    def _get_kinematic_model(self): # TODO DO NOT PUBLIC THIS FUNCTION
        if self._kin_model is None:
            robot = cast("Robot", self)
            assert isinstance(robot.robot_cfg, dict)
            # urdf_file = robot.robot_cfg["kinematics"]["urdf_path"]  # Send global path starting with "/"
            # base_link = robot.robot_cfg["kinematics"]["base_link"]
            # ee_link = robot.robot_cfg["kinematics"]["ee_link"] # "right_gripper_camera" # "right_gripper_site" # 
            # for key in ["external_asset_path", "external_robot_configs_path"]:
            #     # abs_path = self.robot_cfg["kinematics"][key].replace("<PROJECT_DIR>", os.getcwd())
            #     abs_path = resolve_data_path(self.robot_cfg["kinematics"][key])
            #     self.robot_cfg["kinematics"][key] = abs_path
            # if ("external_asset_path" in robot.robot_cfg["kinematics"] and 
            #     robot.robot_cfg["kinematics"]["external_asset_path"] is not None):
            #     urdf_file = join_path(robot.robot_cfg["kinematics"]["external_asset_path"], urdf_file)
            # robot_config = RobotConfig.from_basic(urdf_file, base_link, ee_link, TensorDeviceType())
            tensor_args = TensorDeviceType()
            robot_config = RobotConfig.from_dict(robot.robot_cfg, tensor_args)
            
            self._kin_model = CudaRobotModel(robot_config.kinematics)
        return self._kin_model
    
    def _create_empty_world_ik_solver(self): # TODO DO NOT PUBLIC THIS FUNCTION
        robot = cast("Robot", self)
        if self._ik_solver is None:
            ik_solver = IKSolver(IKSolver.load_from_robot_config(
                robot.robot_cfg,
                regularization=True,
                # world_cfg,
                tensor_args = TensorDeviceType(),
            ))
            self._ik_solver = ik_solver
        return self._ik_solver

