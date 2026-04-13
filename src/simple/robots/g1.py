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
from simple.robots.controllers.combo import BaseDualArmDexEEFController, BaseDualArmDexEEFControllerCfg 
from simple.robots.controllers.eef import  DexHandEEFControllerCfg 
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.registry import RobotRegistry
from typing import Any, List
from simple.robots.mixin import CuRoboMixin
from simple.robots.franka import FrankaMixin
from simple.utils import resolve_res_path, resolve_data_path, load_yaml
from simple.robots.protocols import Humanoid,HeadCamMountable
from simple.robots.protocols import HasDexterousHand, Humanoid
import numpy as np
import transforms3d as t3d
from typing import Tuple


@RobotRegistry.register("g1")
class G1(CuRoboMixin,Humanoid,Robot,HeadCamMountable,HasDexterousHand):
    uid: str = "g1"
    label: str = "Unitree G1"
    #TODO verify
    dof: int = 31

    robot_cfg: Any = "robots/g1/curobo/g1_29dof_with_dex3.yml"
    mjcf_path: str = "robots/g1/g1_29dof_with_dex3.xml"
    usd_path: str = "robots/g1/g1_29dof_with_dex3.usd"
    robot_ns: str = "g1_29dof_with_dex3_hand"
    hand_yaml: str = "robots/g1/curobo/dex3_right.yml"
    hand_uid: str = "dex3_right"
    hand_dof: int = 7
    pregrasp_distance: List[float] = [0.05, 0.08]

    wrist_camera_orientation: List[float] =[ 1,0,0,0]
    head_camera_orientation: List[float] =[ 1,0,0,0]
    # head_cam_link: str = "d435_link"


    init_joint_states: dict[str, float]={

                                        "waist_yaw_joint" : 0,
                                        "waist_roll_joint" : 0,
                                        "waist_pitch_joint" : 0,
                        "left_shoulder_pitch_joint":0, "left_shoulder_roll_joint":0, "left_shoulder_yaw_joint":0, "left_elbow_joint":0,"left_wrist_roll_joint":0,"left_wrist_pitch_joint":0,"left_wrist_yaw_joint":0,
                        "right_shoulder_pitch_joint":0, "right_shoulder_roll_joint":0, "right_shoulder_yaw_joint":0, "right_elbow_joint":0,"right_wrist_roll_joint":0,"right_wrist_pitch_joint":0,"right_wrist_yaw_joint":0,
                        "left_hand_thumb_0_joint":0, "left_hand_thumb_1_joint":0, "left_hand_thumb_2_joint":0, "left_hand_index_0_joint":0, "left_hand_index_1_joint":0, "left_hand_middle_0_joint":0, "left_hand_middle_1_joint":0,
                        "right_hand_thumb_0_joint":0, "right_hand_thumb_1_joint":0, "right_hand_thumb_2_joint":0, "right_hand_index_0_joint":0, "right_hand_index_1_joint":0, "right_hand_middle_0_joint":0, "right_hand_middle_1_joint":0  
                        }  
    joints_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
                    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint", "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint"
                    ]

    robot_eef_offset: float = 0

    controller_cfg: ControllerCfg = BaseDualArmDexEEFControllerCfg(
        waist=PDJointPosControllerCfg(
            joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            init_qpos=[0.0, 0.0, 0.0],
        ),
        left_arm=PDJointPosControllerCfg(
            joint_names=["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"],
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        right_arm=PDJointPosControllerCfg(
            joint_names=["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"],
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        left_eef=DexHandEEFControllerCfg(
            joint_names=["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint"],
            init_qpos=[0, 0, 0, 0, 0, 0, 0],
            close_qpos=[0.3523,-0.0964, 0.2790,  -0.5058,  -1.1950,  -0.5389,  -0.9835]
        ),
        right_eef=DexHandEEFControllerCfg(
            joint_names=["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint", "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint"],
            init_qpos=[0, 0, 0, 0, 0, 0, 0],
            close_qpos=[ 0.02331954, -0.02398408, -0.22170663,  0.25662386,  1.3371105 ,
        0.3085137 ,  0.9805285]
        ),
    )

    hand_prim_path="right_hand_base_link"
    eef_prim_path: str ="right_hand_palm_link"
    wrist_cam_link: str ="right_hand_camera_base_link"
    z_offset= 0.793

    LEFT_ARM_EE_LINK: str = "left_hand_palm_link"
    RIGHT_ARM_EE_LINK: str = "right_hand_palm_link"


  

    joint_limits:dict[str, tuple[float, float]]={}

    FRANKA_FINGER_LENGTH=0


    _controller: Controller | None = None

    visulize_spheres : bool = False

    def __init__(self):
        super().__init__(self.uid, self.dof)
        self.joint_names = self.joints_names

    def update_ee_link(self,hand_uid):
        if "left" in hand_uid:
            ee_link = self.LEFT_ARM_EE_LINK
            link_names = [ "waist_yaw_link", "waist_roll_link", "torso_link",
                    "left_hand_palm_link", "left_hand_index_1_link","left_hand_middle_1_link","left_hand_thumb_2_link",
                    "right_hand_palm_link", "right_hand_index_1_link","right_hand_middle_1_link","right_hand_thumb_2_link",]
            collision_link_names = ['left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_pitch_link', 'torso_link', 
                                    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_pitch_link', 
                                    'left_hand_palm_link', 'left_hand_thumb_0_link', 'left_hand_thumb_1_link', 'left_hand_thumb_2_link', 'left_hand_middle_0_link', 'left_hand_middle_1_link', 'left_hand_index_0_link', 'left_hand_index_1_link',
                                    ]
            
        else:
            ee_link = self.RIGHT_ARM_EE_LINK
            link_names = [ "waist_yaw_link", "waist_roll_link", "torso_link",
                    "right_hand_palm_link", "right_hand_index_1_link","right_hand_middle_1_link","right_hand_thumb_2_link",
                    "left_hand_palm_link", "left_hand_index_1_link","left_hand_middle_1_link","left_hand_thumb_2_link",]
            collision_link_names = ['left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_pitch_link', 'torso_link',
                                    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_pitch_link', 
                                     'right_hand_palm_link', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link', 'right_hand_thumb_2_link', 'right_hand_middle_0_link', 'right_hand_middle_1_link', 'right_hand_index_0_link', 'right_hand_index_1_link']
        self.robot_cfg["kinematics"]["ee_link"] = ee_link
        self.robot_cfg["kinematics"]["link_names"] = link_names

        self.robot_cfg["kinematics"]["collision_link_names"] = collision_link_names
        self.hand_yaml =f"robots/g1/curobo/{hand_uid}.yml"
        print(f'now the ee_link is {self.robot_cfg["kinematics"]["ee_link"]}')


    def setup_control(self, mjData, mjModel)-> Tuple[dict[str, Any], dict[str, Any]]:
        actuators = {}
        joints = {}
   
        joints_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
                    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint", "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint"
                    ]
        for name in joints_names:
            actuators[name] = mjData.actuator(name)
            joints[name] = mjData.joint(name)
        
        self.joints=joints
        self.actuators=actuators
        # self.last_action=[0 for _ in range(len(actuators))]
        self.controller.set_initial_qpos(actuators, joints)

        if not self.joint_limits:
            for j,v in self.joints.items():
                limits = mjModel.jnt_range[v.id]
                self.joint_limits[j] = (limits[0], limits[1])
        return self.joints, self.actuators
    

    def get_actuators_action(self) -> dict[str, float]:
        """ Get the current actuator actions of the robot. """
        return {a: v.ctrl[0] for a,v in self.actuators.items()}

    def get_robot_qpos(self) -> dict[str, float]:
        """ Get the current joint positions of the robot. """
        # return np.array([j.qpos[0] for j in self.joints])

        return {j: v.qpos[0] for j,v in self.joints.items()}
    
    def apply_action(self, action_cmd) -> None:
        assert isinstance(self.controller, BaseDualArmDexEEFController)
        """ TODO change it to step() """

        #TODO make left and right gripper independent
        if action_cmd.type == "open_eef":
            hand_uid = action_cmd.parameters.get("hand_uid",None)
            if "left" in hand_uid:
                self.controller.left_eef.open_gripper(self.actuators)
            elif "right" in hand_uid:
                self.controller.right_eef.open_gripper(self.actuators)
            else:
                self.controller.left_eef.open_gripper(self.actuators)
                self.controller.right_eef.open_gripper(self.actuators)
            return
        elif action_cmd.type == "close_eef":
            hand_uid = action_cmd.parameters.get("hand_uid",None)
            if "left" in hand_uid:
                self.controller.left_eef.close_gripper(self.actuators)
            elif "right" in hand_uid:
                self.controller.right_eef.close_gripper(self.actuators)
            else:
                self.controller.left_eef.close_gripper(self.actuators)
                self.controller.right_eef.close_gripper(self.actuators)
            return
        else:
            target_qpos = action_cmd.parameters["target_qpos"]
            keep_force = action_cmd.parameters.get("keep_force", False)
            # TODO
            if keep_force:
                keys = list(target_qpos.keys())[:-7]#keep other arm force
            else:
                keys = list(target_qpos.keys())[:]
            for jname in keys:
                jval = target_qpos[jname]
                # HACK: to match  joint names
                # jname = jname.replace("panda_", "")
                self.actuators[jname].ctrl = jval
                

    
 


    def fk(self, qpos) -> tuple[List[float], List[float]]:
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
    
    @property
    def head_cam_link(self) -> str:
        """ Get the head camera link name. """
        return "d435_link"


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