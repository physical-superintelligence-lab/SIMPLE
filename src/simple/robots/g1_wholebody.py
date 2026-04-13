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
from simple.robots.controllers.combo import BaseDualArmDexEEFController, BaseDualArmDexEEFControllerCfg ,WholeBodyEEFController, WholeBodyEEFControllerCfg
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
from simple.robots.policy.AMO_Policy import AMO_Policy
import mujoco

LEFT_LEG_JOINTS = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
RIGHT_LEFT_JOINTS = ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINTS = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"]
RIGHT_ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
LEFT_HAND_JOINTS = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint"]
RIGHT_HAND_JOINTS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint", "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint"]

WHOLE_BODY_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEFT_JOINTS + WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

@RobotRegistry.register("g1_wholebody")
class G1Wholebody(CuRoboMixin,Humanoid,Robot,HeadCamMountable,HasDexterousHand):
    uid: str = "g1_wholebody"
    label: str = "Unitree G1 Wholebody"

    #TODO verify
    wholebody_dof: int = 43

    dof: int = 31

    robot_cfg: Any = "robots/g1/curobo/g1_29dof_with_dex3.yml"
    mjcf_path: str = "robots/g1/g1_29dof_wholebody_dex3.xml"
    usd_path: str = "robots/g1/g1_29dof_wholebody_dex3.usd"

    robot_ns: str = "g1_29dof_wholebody_dex3"
    hand_yaml: str = "robots/g1/curobo/dex3_right.yml"
    hand_uid: str = "dex3_right"
    hand_dof: int = 7

    pregrasp_distance: List[float] = [0.05, 0.08]

    wrist_camera_orientation: List[float] =[ 1,0,0,0]
    head_camera_orientation: List[float] =[ 1,0,0,0]
    # head_cam_link: str = "d435_link"

    init_joint_states: dict[str, float] = dict(zip(WHOLE_BODY_JOINTS, [0]*len(WHOLE_BODY_JOINTS)))
    joints_names= WHOLE_BODY_JOINTS
    hand_names = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
    robot_eef_offset: float = 0
    #TODO May have problem here,not all joint are pd control
    controller_cfg: ControllerCfg = WholeBodyEEFControllerCfg(
        left_leg=PDJointPosControllerCfg(
            joint_names=LEFT_LEG_JOINTS,
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        right_leg=PDJointPosControllerCfg(
            joint_names=RIGHT_LEFT_JOINTS,
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        waist =PDJointPosControllerCfg(
            joint_names=WAIST_JOINTS,
            init_qpos=[0.0, 0.0, 0.0],
        ),
        left_arm=PDJointPosControllerCfg(
            joint_names=LEFT_ARM_JOINTS,
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        right_arm=PDJointPosControllerCfg(
            joint_names=RIGHT_ARM_JOINTS,
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        left_eef=DexHandEEFControllerCfg(
            joint_names=LEFT_HAND_JOINTS,
            init_qpos=[0, 0, 0, 0, 0, 0, 0],
            close_qpos=[0.3523,-0.0964, 0.2790,  -0.5058,  -1.1950,  -0.5389,  -0.9835]
        ),
        right_eef=DexHandEEFControllerCfg(
            joint_names=RIGHT_HAND_JOINTS,
            init_qpos=[0, 0, 0, 0, 0, 0, 0],
            close_qpos=[ 0.02331954, -0.02398408, -0.22170663,  0.25662386,  1.3371105 , 0.3085137 ,  0.9805285]
        ),
    )

    hand_prim_path="right_hand_base_link"
    eef_prim_path: str ="right_hand_palm_link"
    wrist_cam_link: str ="right_hand_camera_base_link"
    z_offset= 0.785

    LEFT_ARM_EE_LINK: str = "left_hand_palm_link"
    RIGHT_ARM_EE_LINK: str = "right_hand_palm_link"

    joint_limits:dict[str, tuple[float, float]]={}

    FRANKA_FINGER_LENGTH=0

    _controller: Controller | None = None

    visulize_spheres : bool = False

    def __init__(self):
        super().__init__(self.uid, self.dof)
        self.joint_names = self.joints_names
        self.amo_policy = AMO_Policy(robot_type="g1_dex3_wholebody", device="cuda",joint_names=self.joint_names)

        self.stiffness = np.array([
                150, 150, 150, 300, 80, 20,
                150, 150, 150, 300, 80, 20,
                400, 400, 400,
                80, 80, 40, 60,40,40,40,
                80, 80, 40, 60,40,40,40,
                80, 40, 40, 60, 40, 40, 40,
                80, 40, 40, 60, 40, 40, 40, 
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 1,
                2, 2, 2, 4, 2, 1,
                15, 15, 15,
                2, 2, 1, 1, 1, 1, 1,
                2, 2, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
            ])
        self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,25,25,25,
                25, 25, 25, 25,25,25,25,
                5, 5, 3, 5, 3, 5, 3,
                5, 5, 3, 5, 3, 5, 3,
            ])
        self.count = 0
        self.sim_decimation = 1
        self.command = None # [vx, yaw, vy, d_height, torso_yaw, torso_pitch, torso_roll, turning_flag] shape: (1,8)
        self.desired_robot_pose = None
        self.is_replay = False
        self.is_eval = False
        self.target_waist_qpos = None
        self.target_hand_qpos = None
        self.height = None
        self._debug_ctrl_err = {}
        self._last_target_yaw = 0
        self.last_target_hand_qpos = None


    def reset(self):
        self.joint_names = self.joints_names
        self.amo_policy = AMO_Policy(robot_type="g1_dex3_wholebody", device="cuda",joint_names=self.joint_names)
        self.count = 0
        self.sim_decimation = 1
        self.command = None
        self.desired_robot_pose = None
        self.is_replay = False
        self.is_eval = False
        self.target_waist_qpos = None
        self.target_hand_qpos = None
        self.height = None
        self._debug_ctrl_err = {}
        self.last_target_hand_qpos = None

    def update_ee_link(self,hand_uid):
        if "left" in hand_uid:
            ee_link = self.LEFT_ARM_EE_LINK
            link_names = [
                "waist_yaw_link", "waist_roll_link", "torso_link",
                "left_hand_palm_link", "left_hand_index_1_link","left_hand_middle_1_link","left_hand_thumb_2_link",
                "right_hand_palm_link", "right_hand_index_1_link","right_hand_middle_1_link","right_hand_thumb_2_link"
            ]
            collision_link_names = [
                'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_pitch_link', 'torso_link', 
                'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_pitch_link', 
                'left_hand_palm_link', 'left_hand_thumb_0_link', 'left_hand_thumb_1_link', 'left_hand_thumb_2_link', 'left_hand_middle_0_link', 'left_hand_middle_1_link', 'left_hand_index_0_link', 'left_hand_index_1_link',
            ]
        else:
            ee_link = self.RIGHT_ARM_EE_LINK
            link_names = [ 
                "waist_yaw_link", "waist_roll_link", "torso_link",
                "right_hand_palm_link", "right_hand_index_1_link","right_hand_middle_1_link","right_hand_thumb_2_link",
                "left_hand_palm_link", "left_hand_index_1_link","left_hand_middle_1_link","left_hand_thumb_2_link"
            ]
            collision_link_names = [
                'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_pitch_link', 'torso_link',
                'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_pitch_link', 
                'right_hand_palm_link', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link', 'right_hand_thumb_2_link', 'right_hand_middle_0_link', 'right_hand_middle_1_link', 'right_hand_index_0_link', 'right_hand_index_1_link'
            ]
        
        self.robot_cfg["kinematics"]["ee_link"] = ee_link
        self.robot_cfg["kinematics"]["link_names"] = link_names

        self.robot_cfg["kinematics"]["collision_link_names"] = collision_link_names
        self.hand_yaml =f"robots/g1/curobo/{hand_uid}.yml"
        print(f'now the ee_link is {self.robot_cfg["kinematics"]["ee_link"]}')

    def setup_control(self, mjData, mjModel, **kwargs)-> Tuple[dict[str, Any], dict[str, Any]]:
        actuators = {}
        joints = {}
        
        for name in self.joints_names:
            actuators[name] = mjData.actuator(name)
            joints[name] = mjData.joint(name)
        
        self.joints=joints
        self.actuators=actuators
        # self.last_action=[0 for _ in range(len(actuators))]
        self.controller.set_initial_qpos(actuators, joints)

        self.mjdata = mjData
        self.mjdata.qvel = 0
        self.mjmodel = mjModel

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

    def pd_control(self,target_q, q, stiffness, target_dq, damping,torque_limits):
        """Calculates torques from position commands"""
        torque = (target_q - q) * stiffness - target_dq  * damping
        torque = np.clip(torque, -torque_limits, torque_limits)
        return torque

    import numpy as np

    def inverse_pd_control(self, applied_torque, q, stiffness, target_dq, damping):

        if stiffness == 0:
            return q 
            

        target_q = q + (applied_torque + target_dq * damping) / stiffness
        
        return target_q

    def step(self,command,replay=False,eval=False):
        if not replay: # FIXME VERY BAD LOGIC
            if not eval:
                self.pd_target,self.output_joint_names = self.amo_policy.get_action(self.joints,self.actuators,self.mjdata,command)
            else:
                self.pd_target,self.output_joint_names = self.amo_policy.get_eval_action(self.joints,self.actuators,self.mjdata,command)

        # HACK overwrite waist joints
        if self.target_waist_qpos is not None:
            self.pd_target[12:15] = self.target_waist_qpos
        
        #control left and right leg ,waist
        for _ in range(10):
            for target_q,joint_name in zip(self.pd_target,self.output_joint_names):
                joint_index = self.joints_names.index(joint_name)
                torque = self.pd_control(target_q, self.joints[joint_name].qpos.item(), self.stiffness[joint_index],
                                        self.joints[joint_name].qvel.item(),self.damping[joint_index], self.torque_limits[joint_index])
                # self.actuators[joint_name].ctrl = torque
                self.mjdata.ctrl[joint_index] = torque
            mujoco.mj_step(self.mjmodel, self.mjdata, nstep=1)

        # for jname in self.hand_names:
        #     # print("Line 298", f"self.count in self._debug_ctrl_err = {self.count in self._debug_ctrl_err}")
        #     # if self.count in self._debug_ctrl_err:
        #         # print("L300", f"jname in self._debug_ctrl_err[self.count] = {jname in self._debug_ctrl_err[self.count]}")

        #         if self.count in self._debug_ctrl_err:
        #             assert jname in self._debug_ctrl_err[self.count]
        #             assert len(self._debug_ctrl_err[self.count][jname]) == 3 
        #             # assert "final_q" not in self._debug_ctrl_err[self.count][jname]
        #             self._debug_ctrl_err[self.count][jname]["final_q"] = self.joints[jname].qpos.item()
                    # print("=================record final_q =================")

            # self._debug_ctrl_err[self.count][jname]["torque"] = self.mj

        self.count += 1
    
    def get_robot_pose(self):
        return np.round(self.mjdata.qpos[:7],3)
    
    def apply_action(self, action_cmd) -> None:
        assert isinstance(self.controller, WholeBodyEEFController)
        """ TODO change it to step() """
        # command: [vx, yaw, vy, d_height, torso yaw, torso pitch, torso roll]  
        # FIXME  command types in general:
        # sholud always go through amo policy
        if action_cmd.type == "loco_command": # FIXME change to reset or standing etc...
            keep_waist_pose = action_cmd.parameters.get("keep_waist_pose", False)
            motion_type = action_cmd.parameters.get("motion_type")

            command = action_cmd["command"]
            if not command:
                command = self.amo_policy._last_commands

            #if keep_waist_pose is True, then the waist pose will not be changed
            if keep_waist_pose:
                self.command[0] = command[0]#vx
                self.command[1] = command[1]#yaw
                self.command[2] = command[2]#vy
                self.command[3] = command[3]#d_height
            else:
                self.command = command
                self.target_waist_qpos = None
            self._last_target_yaw = self.command[1]
            
            self.height = self.command[3]

            # HACK the initial waist pitch of amo policy is not the same as the robot
            # if motion_type == "stand":
            self.command[5] = -0.15
            
            if motion_type == "walk":
                self.desired_robot_pose = action_cmd.parameters["desired_robot_pose"]
            else:
                self.desired_robot_pose = None

        elif action_cmd.type == "replay_move_actuators":
            #TODO HERE self.command is not used in replay mode
            self.command = [0,0,0,0,0,-0.15,0]
            self.is_replay = True
            self.pd_target = []
            self.output_joint_names = self.joints_names[:15]

            target_qpos = action_cmd.parameters["target_qpos"]
            self.target_waist_qpos = [target_qpos["waist_yaw_joint"],target_qpos["waist_roll_joint"],target_qpos["waist_pitch_joint"]]
            
            for jname ,ctrl in action_cmd.parameters["target_qpos"].items():
                if jname in self.joints_names[:15]:
                    self.pd_target.append(ctrl)
                else:
                    if jname in self.hand_names:  
                        joint_index = self.joints_names.index(jname)
                        ctrl = self.pd_control(ctrl, self.joints[jname].qpos.item(), self.stiffness[joint_index],
                                        self.joints[jname].qvel.item(),self.damping[joint_index], self.torque_limits[joint_index])
                    self.actuators[jname].ctrl = ctrl
        
        elif "eval" in action_cmd.type:
            # print(f"eval {self.count}: =====================================================\n")
            self.is_eval = True
            self.command = action_cmd.parameters["action_command"]
            self.target_qpos = action_cmd.parameters["target_qpos"]  # hand+arm
            self.waist_qpos = action_cmd.parameters["waist_qpos"] # waist 3 dof
            self.target_waist_qpos = [self.waist_qpos["waist_yaw_joint"],self.waist_qpos["waist_roll_joint"],self.waist_qpos["waist_pitch_joint"]]     
            self._debug_ctrl_err[self.count] = {} 
            # print("Line 371", self.count in self._debug_ctrl_err)
            # exit(0)
            for jname ,ctrl in action_cmd.parameters["target_qpos"].items():
                if jname in self.hand_names:  
                    joint_index = self.joints_names.index(jname)
                    torque = self.pd_control(ctrl, self.joints[jname].qpos.item(), self.stiffness[joint_index],
                        self.joints[jname].qvel.item(),self.damping[joint_index], self.torque_limits[joint_index])
                    # if self.count >= 152:
                        # if "right_hand_thumb_2_joint" in jname:
                        #     # print(f"{jname}: qpos={self.joints[jname].qpos.item()}, torque={torque}")
                        #     torque -= 2
                        #     print(f"{jname}: qpos={self.joints[jname].qpos.item()}, torque={torque}")
                            
                        # elif "right_hand_index_1_joint" in jname:
                        #     torque += 2
                        # elif "right_hand_middle_1_joint" in jname:
                        #     torque += 2
                        # ...
                    
                    self._debug_ctrl_err[self.count][jname] = {"target_q": ctrl, "current_q": self.joints[jname].qpos.item(), "torque": torque}
                    self.actuators[jname].ctrl = torque
                else:
                    self.actuators[jname].ctrl = ctrl
            # print("Line 383", self.count in self._debug_ctrl_err)

        elif "move_qpos" in action_cmd.type or "open_eef" in action_cmd.type or "close_eef" in action_cmd.type:
            self.command = [0,0,0,0,0,-0.15,0,0] # NOTE: assume standing mode when doing motion planning
            self.command[1] = self._last_target_yaw
            target_qpos = action_cmd.parameters["target_qpos"]
            if self.height is not None:
                self.command[3]=self.height
            if action_cmd.type == "open_eef":
                hand_uid = action_cmd.parameters.get("hand_uid",None)
                if "left" in hand_uid:
                    self.controller.left_eef.open_gripper(self.actuators)
                    left_hand_jnames = [jname for jname in self.hand_names if "left" in jname]
                    self.target_hand_qpos = self.last_target_hand_qpos.copy()
                    for jname in left_hand_jnames:
                        self.target_hand_qpos[jname] = 0

                    
                    
                elif "right" in hand_uid:
                    self.controller.right_eef.open_gripper(self.actuators)
                    right_hand_jnames = [jname for jname in self.hand_names if "right" in jname]
                    self.target_hand_qpos = self.last_target_hand_qpos.copy()
                    for jname in right_hand_jnames:
                        self.target_hand_qpos[jname] = 0

                    
                else:
                    self.controller.left_eef.open_gripper(self.actuators)
                    self.controller.right_eef.open_gripper(self.actuators)
                    self.target_hand_qpos = {jname: 0. for jname in self.hand_names}
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
            else: # move_qpos
                keep_force = action_cmd.parameters.get("keep_force", False) # FIXME this logic is also ambiguious

                # motion control waist and position control hand and arm
                # self.command[4] += round(target_qpos["waist_yaw_joint"],3)
                # self.command[5] += round(target_qpos["waist_pitch_joint"],3)
                # self.command[6] += round(target_qpos["waist_roll_joint"],3)

                # HACK save waist joints obtained form motion planner
                self.target_waist_qpos = [target_qpos["waist_yaw_joint"],target_qpos["waist_roll_joint"],target_qpos["waist_pitch_joint"]]  

                if keep_force:
                    keys = list(target_qpos.keys())[3:24]#HACK keep other arm force
                else:
                    keys = list(target_qpos.keys())[3:]#HACK don't need waist
                    
                for jname in keys:
                    # jval = target_qpos[jname]
                    # robot hand use motor control
                    if jname in self.hand_names: #  and jname not in ["right_hand_thumb_2_joint"]
                        joint_index = self.joints_names.index(jname)
                        jval = self.pd_control(target_qpos[jname], self.joints[jname].qpos.item(), self.stiffness[joint_index],
                                        self.joints[jname].qvel.item(), self.damping[joint_index], self.torque_limits[joint_index])
                        # if jname == "right_hand_thumb_2_joint": # 
                        #     print(f"L447 {self.count}: {jname} qpos={self.joints[jname].qpos.item()}, target_q={target_qpos[jname]}, qvel={self.joints[jname].qvel.item()}, torque={jval}")
                        if self.count not in self._debug_ctrl_err:
                            self._debug_ctrl_err[self.count] = {}
                        self._debug_ctrl_err[self.count][jname] = {
                            "target_q": target_qpos[jname], 
                            "current_q": self.joints[jname].qpos.item(), 
                            "torque": jval
                        }
                    else:
                        jval = target_qpos[jname]

                    # HACK: to match  joint names
                    # jname = jname.replace("panda_", "")
                    self.actuators[jname].ctrl = jval

                    
                    # if keep_force:
                        
                       

                self.target_hand_qpos = {jname: target_qpos[jname] for jname in self.hand_names if jname in target_qpos}
                if keep_force:
                    right_hand_jnames = [jname for jname in self.hand_names if "right" in jname]
                    for rjname in right_hand_jnames:
                        force = self.actuators[rjname].force
                        jval = self.actuators[rjname].ctrl
                        joint_index = self.joints_names.index(rjname)
                        # print(f" {self.count}: {rjname} force={force},ctrl={jval}")
                        target_q = self.inverse_pd_control(jval, self.joints[rjname].qpos.item(), self.stiffness[joint_index],
                                        self.joints[rjname].qvel.item(), self.damping[joint_index])
                        self.target_hand_qpos[rjname] = target_q.item()
                self.last_target_hand_qpos = self.target_hand_qpos.copy()



        else:
            raise ValueError("Unknown action command type")
        # self.apply_lower_body_control(command)

    def hand_target_qpos(self):
        if not self.target_hand_qpos:
            return {jname: 0. for jname in self.hand_names}
        return self.target_hand_qpos

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
        from curobo.util.logger import setup_curobo_logger
        # kin_model = self.get_kinematic_model()
        setup_curobo_logger("error")

        if isinstance(qpos, dict):
            qpos_list = []
            for jname in self.kin_model.joint_names:
                qpos_list.append(qpos[jname])
            qpos = qpos_list
        else:
            if type(qpos) is list:
                qpos = np.array(qpos)
            qpos = qpos[12:]
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
    
    
