"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import torch
from typing import List, Protocol, ClassVar, Tuple, List, Any
from typing import runtime_checkable
import numpy as np

from simple.core.controller import Controller, ControllerCfg
from simple.core.action import ActionCmd

try:
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.wrap.reacher.ik_solver import IKSolver
except ImportError:
    CUROBO = None

class HasDexterousHand:
    hand_yaml: str
    hand_uid: str
    hand_dof: int

class BatchPlannable:
    """ Protocol for objects that can be planned with a planner. """
    
    init_joint_states: dict[str, float]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def planning_init_joint_states(self, batch_size: int = 10) -> np.ndarray:
        """ Get the current state of the object. """
        raise NotImplementedError

    def planning_init_quats(self, batch_size: int = 10) -> np.ndarray:
        """ Get the initial quaternions of the object. """
        raise NotImplementedError
    
    def rotate_round_approach_dir_if_needed(self, target_quat: np.ndarray, ref_quat: np.ndarray, approach_axis: str = 'z'):
        """ Rotate the target quaternion around the approach direction if needed. """
        return target_quat

@runtime_checkable
class Graspable(Protocol):
    """ Protocol for objects that have a eef gripper. """

    robot_eef_offset: float
    
    def get_grasp_pose_wrt_robot(self, grasp_info: dict, pregrasp: bool=False, robot_pose=None): # -> List[List[float]]:
        """ Get the grasp pose with respect to the robot. 
        Args:
            grasp_info (dict): The grasp information in the object frame.
            pregrasp (bool): Whether to return the pre-grasp pose.
            robot_pose (np.ndarray): The grasp pose in robot frame.
        """
        raise NotImplementedError("Must implement get_grasp_pose_wrt_robot method.")
    
    def open_gripper(self):
        """ Open the gripper. """
        raise NotImplementedError("Must implement open_gripper method.")
    
    def close_gripper(self):
        """ Close the gripper. """
        raise NotImplementedError("Must implement close_gripper method.")

@runtime_checkable
class HasKinematics(Protocol):
    """ Protocol for objects that have a forward and inverse kinematics. """

    dof: int
    
    def fk(self, qpos: List[float]): #  -> Tuple[List[float], List[float]]
        """ Forward kinematics to compute end-effector pose from joint values. """
        raise NotImplementedError("Must implement fk method.")
    
    def ik(self,  p, q, current_joint = None) : # -> List[float]
        """ Inverse kinematics to compute joint values from end-effector pose. """
        raise NotImplementedError("Must implement ik method.")
    
    def get_link_pose(self, link_name: str, joint_qpos: dict[str, float] | list[float]):
        """ Get the pose of a link given joint positions. """
        raise NotImplementedError("Must implement get_link_pose method.")

@runtime_checkable
class HasParallelGripper(Protocol):

    def get_eef_pose_from_hand_pose(self, p, q) -> tuple[list[float], list[float]]: ...

    def get_hand_pose_from_eef_pose(self, p, q, hack=0) -> tuple[list[float], list[float]]: ...
       

@runtime_checkable
class WristCamMountable(Protocol):
    # wrist_cam_link: str
    
    wrist_camera_orientation: List[float]

    @property
    def wrist_cam_link(self) -> str:
        """ Get the wrist camera link name. """
        raise NotImplementedError("Must implement wrist_cam_link property.")
@runtime_checkable   
class HeadCamMountable(Protocol):
    # wrist_cam_link: str
    
    head_camera_orientation: List[float]

    @property
    def head_cam_link(self) -> str:
        """ Get the head camera link name. """
        raise NotImplementedError("Must implement head_cam_link property.")
# @runtime_checkable
class Controllable:

    joint_limits: dict[str, Tuple[float, float]]
    controller_cfg: ControllerCfg
    actuators: dict[str, Any]
    joint_names: List[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._controller = None
        

    @property
    def controller(self) -> Controller:
        if self._controller is None:
            self._controller = self.controller_cfg.clazz(
                self.controller_cfg
            )
        return self._controller


    def setup_control(self, mjData, mjModel, **kwargs) -> Tuple[dict[str, Any], dict[str, Any]]: ...

    # def initialize_controller(self, physics_data) -> Tuple[List[Any], List[Any]]: ...

    # def get_robot_qpos(self) -> dict[str, float]: ...
    
    def apply_action(self, action_cmd) -> None: ...
    
    def random_action(self) -> ActionCmd:
        random_actions = self.controller.action_space.sample()
        rand_action_dict = {}
        rand_eef_states = {}
        for ctrl_name, ctrl_value in random_actions.items():
            if "eef" in ctrl_name:
                rand_eef_states[ctrl_name] = "open_eef" if (ctrl_value == 0).mean() > 0.5 else "close_eef"
            else:
                rand_action_dict.update(zip(
                    getattr(self.controller, ctrl_name).cfg.joint_names, 
                    # ctrl_value
                    getattr(self.controller, ctrl_name).cfg.init_qpos #ctrl_value
                ))
        return ActionCmd(
            "move_qpos_with_eef", 
            target_qpos=rand_action_dict, 
            eef_state=rand_eef_states
        )
    
    def get_actuators_action(self) -> dict[str, float]:
        """ Get the current actuator actions of the robot. """
        return {a: v.ctrl[0] for a,v in self.actuators.items()}

class CuroboCompatible(Protocol):
    
    robot_cfg: ClassVar[dict]  # Robot configuration dictionary
    # kin_model: ClassVar[CudaRobotModel]  # Forward kinematic model
    # ik_solver: ClassVar[IKSolver]  # Inverse kinematic solver

@runtime_checkable
class DualArm(Protocol):
    LEFT_ARM_EE_LINK: str 
    RIGHT_ARM_EE_LINK: str 
    
class Humanoid(DualArm): # (Protocol)
    """ Protocol for humanoid robots. """
    
    # position: List[float]
    # orientation: List[float]
@runtime_checkable
class Wholebody(Protocol):
    """ Protocol for wholebody robots. """
    
    wholebody_dof: int


