"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""


from typing import TypeVar, Generic
from typing import Tuple, List, Any
from simple.core.robot import Robot
import transforms3d as t3d

from simple.robots.controllers.combo import SingleArmBinaryEEFController
from simple.robots.mixin import CuRoboMixin
from simple.robots.protocols import HasParallelGripper, WristCamMountable
import numpy as np

# T = TypeVar("T", bound=Robot)  #Union[Robot, HasGripper]

class FrankaMixin(CuRoboMixin, WristCamMountable, HasParallelGripper):
    # FRANKA_FINGER_LENGTH: float = 0.1034 # same as panda
    pregrasp_distance: List[float] = [0.05, 0.08]

    joint_names = [
        "joint1", "joint2","joint3", "joint4", "joint5", "joint6", "joint7",
        "finger_joint1", "finger_joint2"
    ]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # ###  default impl. for BatchPlannable ###
    # def planning_init_joint_states(self, batch_size: int):
    #     return np.asarray(list(self.init_joint_states.values())[:7]).reshape(1, -1).repeat(batch_size, axis=0)

    # def planning_init_quats(self, batch_size: int):
    #     _, init_quats = self.fk(list(self.init_joint_states.values())[:7])
    #     return np.array(init_quats).reshape(1, 4).repeat(batch_size, 0)
        
    ### default impl. for Graspable ###
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
            [0,0,1,0], 
            [0,1,0,0], 
            [-1,0,0,0], 
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

        # return [
        #     (pregrasp_pos_in_robot, pregrasp_ori_in_robot), 
        #     (grasp_pos_in_robot, grasp_ori_in_robot)
        # ]

        return (grasp_pos_in_robot, grasp_ori_in_robot)
    
    def rotate_round_approach_dir_if_needed(self, target_quat: np.ndarray, ref_quat: np.ndarray, approach_axis: str = 'z'):
        """
        Returns target_quat flipped by z axis if it is closer to ref_quat after flipping (180 degrees rotation around z axis).
        """
        ref_to_target_quat = t3d.quaternions.qmult(target_quat, t3d.quaternions.qinverse(ref_quat))
        _, ref_to_target_angle = t3d.quaternions.quat2axangle(ref_to_target_quat)
        # you can rotate inversely
        ref_to_target_angle = min(ref_to_target_angle, 2*np.pi - ref_to_target_angle)
        flip_axis = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[approach_axis]
        flipped_target_quat = t3d.quaternions.qmult(target_quat, t3d.quaternions.axangle2quat(flip_axis, np.pi))
        ref_to_flipped_target_quat = t3d.quaternions.qmult(flipped_target_quat, t3d.quaternions.qinverse(ref_quat))
        _, ref_to_flipped_target_angle = t3d.quaternions.quat2axangle(ref_to_flipped_target_quat)
        ref_to_flipped_target_angle = min(ref_to_flipped_target_angle, 2*np.pi - ref_to_flipped_target_angle)
        return target_quat if ref_to_target_angle < ref_to_flipped_target_angle else flipped_target_quat
    
    
    # def initialize_controller(self, data) -> Tuple[List[Any], List[Any]]:
    #     self.actuators = []
    #     self.joints = []
    #     for i in range(1, 8):
    #         self.actuators.append(data.actuator(f'actuator{i}'))
    #         self.joints.append(data.joint(f'joint{i}'))
    #     for i in range(1, 3):
    #         self.actuators.append(data.actuator(f'finger_actuator{i}'))
    #         self.joints.append(data.joint(f'finger_joint{i}'))

    #     self.controller.setup(self.joints, self.actuators)
        
    #     self.last_action = [0 for _ in range(9)]
    #     self.controller.set_initial_qpos(self.init_joint_states[:7])
    #     return self.joints, self.actuators

    def setup_control(self, mjData, mjModel,**kwargs)-> Tuple[dict[str, Any], dict[str, Any]]:
        actuators = {}
        joints = {}
        for i in range(1, 8):
            actuators[f"joint{i}"] = mjData.actuator(f'actuator{i}')
            joints[f"joint{i}"] = mjData.joint(f'joint{i}')
        for i in range(1, 3):
            actuators[f"finger_joint{i}"] = mjData.actuator(f'finger_joint{i}')
            joints[f"finger_joint{i}"] = mjData.joint(f'finger_joint{i}')
        self.joints = joints
        self.actuators = actuators

        # self.last_action = [0 for _ in range(9)]
        # self._set_initial_qpos(init_robot_qpos)

        self.controller.set_initial_qpos(actuators, joints)

        # TODO merge into controller
        if not self.joint_limits:
            for j in self.joints.values():
                limits = mjModel.jnt_range[j.id]
                self.joint_limits[j.name] = (limits[0], limits[1])

        return self.joints,self.actuators

    def apply_action(self, action_cmd) -> None:
        assert isinstance(self.controller, SingleArmBinaryEEFController)
        """ TODO change it to step() """
        if action_cmd.type == "open_eef":
            self.controller.eef.open_gripper(self.actuators)
            return
        elif action_cmd.type == "close_eef":
            self.controller.eef.close_gripper(self.actuators)
            return
        else:
            target_qpos = action_cmd.parameters["target_qpos"]
            # action: (act_id, action)
            # assert len(ctrl_action) == 7
            for jname, jval in target_qpos.items():
                # HACK: to match  joint names
                jname = jname.replace("panda_", "").replace("fr3_", "") # an ugly hack
                if jname.startswith("finger_joint"):
                    continue
                self.actuators[jname].ctrl = jval
                # self.last_action[act[0]] = act[1]
            
            if "eef_state" in action_cmd.parameters:
                eef_state = action_cmd.parameters["eef_state"]
                if eef_state == "open_eef":
                    self.controller.eef.open_gripper(self.actuators)
                elif eef_state == "close_eef":
                    self.controller.eef.close_gripper(self.actuators)
                else:
                    # raise ValueError(f"Unknown eef_state: {eef_state}")
                    ... # do nothing
        
            # self._last_ctrl_action = ctrl_action # for DEBUG
            # return self.last_action
    
    # def open_gripper(self):
    #     # set finger force to 0.125 N, in the opposite direction?
    #     self.actuators[7].ctrl = 0.0205
    #     self.actuators[8].ctrl = 0.0205

    # def close_gripper(self):
    #     # set finger force to 0.125 N
    #     self.actuators[7].ctrl = 0.
    #     self.actuators[8].ctrl = 0.

    def get_robot_qpos(self) -> dict[str,float]:
        """ Get the current joint positions of the robot. """
        return {j.name.replace("panda/", ""): j.qpos[0] for j in self.joints}

    # def _set_initial_qpos(self, init_robot_qpos):
    #     ctrl_action = list(enumerate(init_robot_qpos))
    #     for act in ctrl_action:
    #         self.joints[act[0]].qpos = act[1]
    #         self.joints[act[0]].qvel = 0
    #         self.joints[act[0]].qacc = 0
    #         if act[0] < 7:
    #             self.actuators[act[0]].ctrl = act[1]

    #     # open gripper by default
    #     self.actuators[7].ctrl = 0.0205
    #     self.actuators[8].ctrl = 0.0205