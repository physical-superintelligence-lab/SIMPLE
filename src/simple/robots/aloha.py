"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""


from typing import Tuple, List, Any, ClassVar
from simple.core.robot import Robot
from simple.core.controller import Controller
# from simple.robots.controllers import PDJointPosControllerCfg
# from simple.robots.controllers import ControllerCfg
from simple.core.controller import ControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.controllers.eef import BinaryEEFControllerCfg, ParallelGripperEEFController, ParallelGripperEEFControllerCfg
from simple.robots.protocols import WristCamMountable, DualArm
from simple.robots.mixin import CuRoboMixin
from simple.robots.controllers.combo import DualArmBinaryEEFControllerCfg, DualArmBinaryEEFController# , DualArmEEFControllerCfg
from simple.robots.registry import RobotRegistry
from simple.utils import resolve_res_path, resolve_data_path, load_yaml

import os
import torch
import numpy as np
from torch._tensor import Tensor
import transforms3d as t3d

try:
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.util_file import join_path
except ImportError:
    raise RuntimeError("curobo not installed, uv pip install --groups curobo")

@RobotRegistry.register("aloha")
class Aloha(CuRoboMixin, WristCamMountable, Robot, DualArm):

    uid: str = "aloha"
    label: str = "Aloha"
    dof: int = 16
    robot_cfg: str | dict = "robots/aloha/curobo/aloha.yml"
    mjcf_path="robots/aloha/aloha.xml"
    usd_path="robots/aloha/aloha.usd"
    pregrasp_distance: List[float] = [0.05, 0.08]
    wrist_camera_orientation: List[float] = [ 0.5, 0.5,  -0.5,  0.5] #[0,0,0,1] # xyzw
    init_joint_states: dict[str, float] = {
        "left_waist": 0.0,
        "left_shoulder": -0.96,
        "left_elbow": 1.16,
        "left_forearm_roll": 0.0,
        "left_wrist_angle": -0.3,
        "left_wrist_rotate": 0.0,

        "right_waist": 0.0,
        "right_shoulder": -0.96,
        "right_elbow": 1.16,
        "right_forearm_roll": 0.0,
        "right_wrist_angle": -0.3,
        "right_wrist_rotate": 0.0,
        "left_left_finger": 0.04,
        "left_right_finger": 0.04,
        "right_left_finger": 0.04,
        "right_right_finger": 0.04,
    }
    robot_ns:str = "aloha"
    hand_prim_path: str = "right_gripper_site"
    eef_prim_path: str = "right_gripper_site"
    
    LEFT_ARM_EE_LINK: str = "left_gripper_site"
    RIGHT_ARM_EE_LINK: str = "right_gripper_site"
    
    left_wrist_cam_link:str= "left_gripper_camera"
    right_wrist_cam_link: str="right_gripper_camera"
    robot_eef_offset: float = 0.00 # HACK to avoid collision

    controller_cfg: ControllerCfg = DualArmBinaryEEFControllerCfg(
        left_arm=PDJointPosControllerCfg(
            joint_names=["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate"],
            init_qpos=[0.0, -0.96, 1.16, 0.0, -0.3, 0.0],
        ),
        right_arm=PDJointPosControllerCfg(
            joint_names=["right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate"],
            init_qpos=[0.0, -0.96, 1.16, 0.0, -0.3, 0.0],
        ),
        left_eef=ParallelGripperEEFControllerCfg(
            joint_names=["left_left_finger", "left_right_finger"],
            init_qpos=[0.04, 0.04],
        ),
        right_eef=ParallelGripperEEFControllerCfg(
            joint_names=["right_left_finger", "right_right_finger"],
            init_qpos=[0.04, 0.04],
        ),
        waist = None,
    )

    joint_names = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", 
        "left_wrist_angle", "left_wrist_rotate", 
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", 
        "right_wrist_angle", "right_wrist_rotate", 
        "left_left_finger", "left_right_finger","right_left_finger", "right_right_finger"
    ]
    # supported_controllers: dict[str, ControllerCfg] = dict(
    #     pd_joint_pos=PDJointPosControllerCfg(
    #         # self,
    #     ),
    # )
    joint_limits: dict[str, Tuple[float, float]] = {}
    FRANKA_FINGER_LENGTH:float = 0.0#
    is_single_arm:bool = False
    visulize_spheres: bool = False
    active_arm: str | None = None
    

    # _robot_cfg: Any = None  # lazy init
    _kin_model: Any = None  # lazy init (right arm, the default arm)
    _kin_model_left: Any = None  # lazy init (left arm)
    # _ik_solver: Any = None  # lazy init
    _ik_solver_left: Any = None  # lazy init (left arm IK solver)
    _ik_solver_right: Any = None  # lazy init (right arm IK solver)

    _controller: Controller | None = None

    def __init__(self):
        super().__init__(self.uid, self.dof)
        # print(self.robot_cfg["kinematics"]["ee_link"])
        
    def switch_arm(self, arm: str):
        if arm not in ["left", "right", "all"]:
            raise ValueError(f"arm must be 'left' or 'right', got {arm}")
        if self.active_arm == arm:
            return False
         
        self.active_arm = arm
        self.update_ee_link()
        return True
    
    def update_ee_link(self): 
        ee_link = self.LEFT_ARM_EE_LINK if self.active_arm == "left" else self.RIGHT_ARM_EE_LINK
        self.robot_cfg["kinematics"]["ee_link"] = ee_link # type: ignore
        aa = self.robot_cfg["kinematics"]["ee_link"]
        print(f"now ee_link: {aa}")
    
    # @property
    # def controller(self) -> Controller:
    #     """Get controller instance"""
    #     if self._controller is None:
    #         self._controller = self.controller_cfg.clazz(self.controller_cfg)
    #     return self._controller

    def setup_control(self, mjData, mjModel,mjSpec) -> Tuple[dict[str, Any], dict[str, Any]]:
        actuators = {}
        joints = {}
          
        

        for name in self.joint_names:
            actuators[name] = mjData.actuator(name)
            joints[name] = mjData.joint(name)

        self.joints = joints
        self.actuators = actuators


        # self.last_action = [0 for _ in range(14)]

        self.controller.set_initial_qpos(actuators, joints)

        if not self.joint_limits:
            for jname, joint in self.joints.items():
                limits = mjModel.jnt_range[joint.id]
                self.joint_limits[jname] = (limits[0], limits[1])
        
        return joints, actuators

    # def _set_initial_qpos(self,init_robot_qpos):
    #     init_robot_qpos=self.init_joint_states 
    #     ctrl_action = list(enumerate(init_robot_qpos))
    #     for idx, qpos_val in ctrl_action:
    #         self.joints[idx].qpos = qpos_val
    #         self.joints[idx].qvel = 0
    #         self.joints[idx].qacc = 0

    #     init_ctrl=[0, -0.96, 1.16, 0 ,-0.3 ,0, 0.04,0 ,-0.96, 1.16, 0, -0.3 ,0 ,0.04 ]
    #     for i, ctrl_val in enumerate(init_ctrl):
    #         self.actuators[i].ctrl = ctrl_val

    #     # open gripper by default
    #     self.actuators[6].ctrl = 0.04
    #     self.actuators[13].ctrl = 0.04

    def get_robot_qpos(self) -> dict[str, float]:
        """ Get the current joint positions of the robot. """
        # return np.array([j.qpos[0] for j in self.joints])
        return {j: v.qpos[0] for j,v in self.joints.items()}
    
    # def random_action(self) -> List[float]:
    #     action = {}
    #     for jname, limits in self.joint_limits.items():
    #         action[jname] = np.random.uniform(limits[0], limits[1])
    #     return list(action.values())
    
    # def extract_arm_joints(self, qpos, arm='left'):
    #     """Extract arm joints from full joint state.
        
    #     Args:
    #         qpos: Full joint positions (16D with gripper fingers)
    #         arm: Which arm to extract ('left' or 'right')
        
    #     Returns:
    #         6D array of arm joint positions (excluding gripper fingers)
    #     """
    #     if isinstance(qpos, dict):
    #         qpos = np.array([qpos[name] for name in qpos.keys()])
        
    #     if len(qpos) == 16:
    #         return qpos[0:6] if arm == 'left' else qpos[8:14]
    #     elif len(qpos) == 12: 
    #         return qpos[0:6] if arm == 'left' else qpos[6:12]
    

    def apply_action(self, action_cmd) -> None:
        assert isinstance(self.controller, DualArmBinaryEEFController)
        """ TODO change it to step() """
        desired_arm = None
        if action_cmd.parameters is not None:
            desired_arm = action_cmd.parameters.get("hand_uid")
        if desired_arm is not None and self.active_arm is not None and desired_arm != self.active_arm:
            self.switch_arm(desired_arm)
        
        # CRITICAL: Maintain non-active arm position by setting ctrl to current qpos
        # This prevents drift when only controlling one arm
        # other_arm_prefix = "right_" if self.active_arm == "left" else "left_"
        if self.active_arm == "left":
            other_arm_prefix = "right_"
        elif self.active_arm == "right":
            other_arm_prefix = "left_"
        else:
            other_arm_prefix = None
        
        if other_arm_prefix is not None:
            for jname in self.joints.keys():
                if jname.startswith(other_arm_prefix):
                    if "finger" in jname:
                        continue
                    self.actuators[jname].ctrl = self.joints[jname].qpos[0]
        
        if action_cmd.type == "open_eef":
            if self.active_arm == "left":
                self.controller.left_eef.open_gripper(self.actuators)
            else:
                self.controller.right_eef.open_gripper(self.actuators)
            return
        elif action_cmd.type == "close_eef":
            if self.active_arm == "left":
                self.controller.left_eef.close_gripper(self.actuators)
            else:
                self.controller.right_eef.close_gripper(self.actuators)
            return
        else:
            target_qpos = action_cmd.parameters["target_qpos"]
            # action: (act_id, action)
            # assert len(ctrl_action) == 7
            arm_prefix = f"{self.active_arm}_"
            for jname, jval in target_qpos.items():
                # HACK: to match  joint names
                if jname.startswith("finger"):
                    continue
                if self.active_arm is None:
                    self.actuators[jname].ctrl = jval
                elif jname.startswith(arm_prefix):
                    self.actuators[jname].ctrl = jval

            if "eef_state" in action_cmd.parameters and action_cmd.parameters["eef_state"] is not None:
                eef_state = action_cmd.parameters["eef_state"]
                if type(eef_state) == str: # all grippers sahre the same state
                    if eef_state == "open_eef":
                        if self.active_arm == "left":
                            self.controller.left_eef.open_gripper(self.actuators)
                        elif self.active_arm == "right":
                            self.controller.right_eef.open_gripper(self.actuators)
                        else:
                            self.controller.left_eef.open_gripper(self.actuators)
                            self.controller.right_eef.open_gripper(self.actuators)
                    elif eef_state == "close_eef":
                        if self.active_arm == "left":
                            self.controller.left_eef.close_gripper(self.actuators)
                        elif self.active_arm == "right":
                            self.controller.right_eef.close_gripper(self.actuators)
                        else:
                            self.controller.left_eef.close_gripper(self.actuators)
                            self.controller.right_eef.close_gripper(self.actuators)
                    elif eef_state == "dual_eef":
                        left_eef_state = action_cmd.parameters.get("left_eef")
                        right_eef_state = action_cmd.parameters.get("right_eef")
                        if left_eef_state == "open_eef":
                            self.controller.left_eef.open_gripper(self.actuators)
                        else:
                            self.controller.left_eef.close_gripper(self.actuators)
                        if right_eef_state == "open_eef":
                            self.controller.right_eef.open_gripper(self.actuators)
                        else:
                            self.controller.right_eef.close_gripper(self.actuators)
                    else:
                        raise ValueError(f"Unknown eef_state: {eef_state}")
                else:
                    for eef_name, state in eef_state.items():
                        if state == "open_eef":
                            getattr(self.controller, eef_name).open_gripper(self.actuators)
                        elif state == "close_eef":
                            getattr(self.controller, eef_name).close_gripper(self.actuators)
                        else:
                            raise ValueError(f"Unknown eef_state for {eef_name}: {state}")


    def setup_render(self): 
        ... # TODO

    @property
    def wrist_cam_link(self) -> str: # FIXME
        assert isinstance(self.robot_cfg, dict)
        return "right_gripper_camera" #self.robot_cfg["kinematics"]["ee_link"].replace("site","camera")
        # return self.left_wrist_cam_link if self.active_arm == "left" else self.right_wrist_cam_link
    
    # @property
    # def eef_prim_path(self): # type: ignore[override]
    #     # return self.robot_cfg["kinematics"]["ee_link"] # , "right_gripper_site"
    #     return self.LEFT_ARM_EE_LINK if self.active_arm == "left" else self.RIGHT_ARM_EE_LINK

    # @property
    # def hand_prim_path(self): # type: ignore[override]
    #     # return "right_gripper_base"
    #     # return self.robot_cfg["kinematics"]["ee_link"].replace("site","base")
    #     ee_link = self.LEFT_ARM_EE_LINK if self.active_arm == "left" else self.RIGHT_ARM_EE_LINK
    #     return ee_link.replace("site", "base")

    def get_grasp_pose_wrt_robot(self, grasp_info: dict, pregrasp: bool=False, robot_pose=None):
        assert robot_pose is not None, "not implemented"

        # grasp pose
        T_ee_hand = np.eye(4, dtype=np.float32)
        T_ee_hand[:3, 3] = np.array([-self.robot_eef_offset,0, 0], dtype=np.float32)

        T_grasp_ee = np.array([
            [1,0,0,0], 
            [0,-1,0,0], 
            [0,0,-1,0], 
            [0,0,0,1]
        ], dtype=np.float32) # only for graspnet

        R_world_grasp = t3d.quaternions.quat2mat(grasp_info['orientation'])
        
        T_world_grasp = np.eye(4, dtype=np.float32)
        # HACK: 0.01 offset to avoid collision
        T_world_grasp[:3, 3] = grasp_info['position'] + (grasp_info['depth']-0.01) * R_world_grasp[:, 0]
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

    def rotate_round_approach_dir_if_needed(self, target_quat: np.ndarray, ref_quat: np.ndarray, approach_axis: str = 'x'):
        """
        Returns target_quat flipped around approach axis if it is closer to ref_quat after flipping.
        """
        # Calculate rotation from reference to target
        ref_to_target_quat = t3d.quaternions.qmult(target_quat, t3d.quaternions.qinverse(ref_quat))
        _, ref_to_target_angle = t3d.quaternions.quat2axangle(ref_to_target_quat)
        # Account for rotation direction ambiguity (can go either way)
        ref_to_target_angle = min(ref_to_target_angle, 2*np.pi - ref_to_target_angle)
        
        # Create 180° flip around specified approach axis
        flip_axis = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[approach_axis]
        flipped_target_quat = t3d.quaternions.qmult(target_quat, t3d.quaternions.axangle2quat(flip_axis, np.pi))
        
        # Calculate rotation from reference to flipped target
        ref_to_flipped_target_quat = t3d.quaternions.qmult(flipped_target_quat, t3d.quaternions.qinverse(ref_quat))
        _, ref_to_flipped_target_angle = t3d.quaternions.quat2axangle(ref_to_flipped_target_quat)
        ref_to_flipped_target_angle = min(ref_to_flipped_target_angle, 2*np.pi - ref_to_flipped_target_angle)
        
        # Return the orientation that requires less rotation
        return target_quat if ref_to_target_angle < ref_to_flipped_target_angle else flipped_target_quat

    # def fk(self, qpos, arm='right'):
    #     from curobo.types.base import TensorDeviceType

    #     if isinstance(qpos, list):
    #         qpos = np.array(qpos)
        
    #     if arm == 'both':
    #         if len(qpos) != 12:
    #             raise ValueError(f"For arm='both', qpos must be 12D, got {len(qpos)}D")
    #         return {
    #             'left': self.fk(qpos[:6], arm='left'),
    #             'right': self.fk(qpos[6:], arm='right')
    #         }
        
    #     if len(qpos) == 6:
    #         arm_qpos = qpos
    #     elif len(qpos) == 12:
    #         arm_qpos = qpos[:6] if arm == 'left' else qpos[6:]
    #     else:
    #         raise ValueError(f"Expected qpos to be 6D or 12D, got {len(qpos)}D")
        
    #     full_qpos = np.concatenate([arm_qpos, np.zeros(6)])
    #     kin_model = self.kin_model_left if arm == 'left' else self.kin_model
        
    #     tensor_args = TensorDeviceType()
    #     joint_traj = torch.from_numpy(full_qpos).to(torch.float32)
    #     joint_traj = tensor_args.to_device(joint_traj).unsqueeze(0)
    #     out = kin_model.get_state(joint_traj)
        
    #     return out.ee_position[0].cpu().numpy(), out.ee_quaternion[0].cpu().numpy()
    
    # def ik(self, p, q, current_joint=None, arm='right'):
    #     import torch
    #     from curobo.types.base import TensorDeviceType
    #     from curobo.types.math import Pose

    #     ik_solver = self.create_left_arm_ik_solver() if arm == 'left' else self.create_right_arm_ik_solver()
    #     tensor_args = TensorDeviceType()
    #     p, q = tensor_args.to_device(p), tensor_args.to_device(q)
    #     goal = Pose(p, q)
        
    #     retract_cfg = seed_cfg = None
    #     if current_joint is not None:
    #         if len(current_joint) == 6:
    #             arm_joints = current_joint
    #         elif len(current_joint) == 12:
    #             arm_joints = current_joint[:6] if arm == 'left' else current_joint[6:]
    #         else:
    #             raise ValueError(f"Expected current_joint to be 6D or 12D, got {len(current_joint)}D")
            
    #         full_joints = np.concatenate([arm_joints, np.zeros(6)])
    #         retract_cfg = torch.tensor(full_joints).cuda().float()
    #         seed_cfg = torch.tensor(full_joints).float().unsqueeze(0).repeat(64, 1).cuda().unsqueeze(0)
    #     result = ik_solver.solve_single(goal, retract_cfg, seed_cfg)
    #     if result.success[0, 0]:
    #         qpos_full = result.solution[0, 0].cpu().numpy()
    #         qpos_arm = qpos_full[:6]
    #         return qpos_arm
        
    #     raise RuntimeError(f"IK failed for {arm} arm!")
