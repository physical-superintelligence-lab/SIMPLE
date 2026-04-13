"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from typing import Any, List, Dict, Tuple
from threading import Lock, Thread

from simple.core.robot import Robot
from simple.core.controller import Controller
from simple.core.controller import ControllerCfg
from simple.robots.controllers.combo import WholeBodyEEFController, WholeBodyEEFControllerCfg
from simple.robots.controllers.eef import  DexHandEEFControllerCfg 
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.registry import RobotRegistry
from simple.robots.mixin import CuRoboMixin
from simple.utils import resolve_res_path, resolve_data_path, load_yaml
from simple.robots.protocols import Humanoid,HeadCamMountable
from simple.robots.protocols import HasDexterousHand, Humanoid
from simple.core.action import ActionCmd
from simple.core.types import Pose

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.mujoco_sim.robot import Robot as GearSonicRobot
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand

LEFT_LEG_JOINTS = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
RIGHT_LEFT_JOINTS = ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINTS = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"]
RIGHT_ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
LEFT_HAND_JOINTS = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint"]
RIGHT_HAND_JOINTS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint", "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint"]

WHOLE_BODY_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEFT_JOINTS + WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
STABILIZE_VEL_THRESHOLD: float = 1e-4 # max |qvel[0:6]| to consider robot stable
MIN_STABILIZE_STEPS: int = 100 # min steps before velocity check is valid (1s at 200Hz)

@RobotRegistry.register("g1_sonic")
class G1Sonic(CuRoboMixin,Humanoid,Robot,HeadCamMountable,HasDexterousHand):
    uid: str = "g1_sonic"
    label: str = "Unitree G1 Wholebody"

    # #TODO verify
    wholebody_dof: int = 43
    dof: int = 29

    robot_cfg: Any = "robots/g1/curobo/g1_29dof_with_dex3.yml" # FIXME
    mjcf_path: str = "robots/g1_sonic/g1_29dof_with_hand.xml"
    usd_path: str = "robots/g1/g1_29dof_wholebody_dex3.usd" # FIXME

    robot_ns: str = "g1_29dof_wholebody_dex3" # FIXME
    hand_yaml: str = "robots/g1/curobo/dex3_right.yml" # FIXME
    hand_uid: str = "dex3_right" # FIXME
    hand_dof: int = 7  # FIXME

    # pregrasp_distance: List[float] = [0.05, 0.08]

    wrist_camera_orientation: List[float] =[ 1,0,0,0]
    head_camera_orientation: List[float] =[ 1,0,0,0]
    # head_cam_link: str = "d435_link"

    init_joint_states: dict[str, float] = dict(zip(WHOLE_BODY_JOINTS, [0]*len(WHOLE_BODY_JOINTS)))
    joint_names= WHOLE_BODY_JOINTS
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
    # z_offset= 0.785

    LEFT_ARM_EE_LINK: str = "left_hand_palm_link"
    RIGHT_ARM_EE_LINK: str = "right_hand_palm_link"

    joint_limits:dict[str, tuple[float, float]] = {}

    # FRANKA_FINGER_LENGTH=0

    _controller: Controller | None = None

    visulize_spheres : bool = False
    
    sonic_config: dict = {}

    spawn_pose: Pose  

    def __init__(self, sonic_config: dict, **kwargs) -> None:
        super().__init__(self.uid, self.dof)
        self.sonic_config = sonic_config

        self.command = None # [vx, yaw, vy, d_height, torso_yaw, torso_pitch, torso_roll] shape: (1,7)
        self._debug_ctrl_err = {}

        # adapted from run_sim_loop.main
        self.sonic_robot_model = instantiate_g1_robot_model()

        # adapted from BaseSimulator.__init__
        self.env_name = self.sonic_config["ENV_NAME"]
        self.sim_dt = self.sonic_config["SIMULATE_DT"]
        self.reward_dt = self.sonic_config.get("REWARD_DT", 0.02)
        self.image_dt = self.sonic_config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.sonic_config.get("VIEWER_DT", 0.02)
        self.sonic_robot = GearSonicRobot(self.sonic_config) # FIXME rename

        # adapated from DefaultEnv.__init__
        self.num_body_dof = self.sonic_robot.NUM_JOINTS
        self.num_hand_dof = self.sonic_robot.NUM_HAND_JOINTS
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.sonic_robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = {}
        self.reward_lock = Lock()
        self.unitree_bridge = None

    def reset(self, **kwargs):
        self.command = None # FIXME
        self._debug_ctrl_err = {}
        self.spawn_pose = kwargs["spawn_pose"]
        self._stabilized = False
        self._stabilize_step_count = 0

    @property
    def stabilized(self) -> bool:
        """True once max(|floating-base qvel[0:6]|) drops below STABILIZE_VEL_THRESHOLD.

        Requires MIN_STABILIZE_STEPS accesses before the velocity check is
        evaluated, preventing an immediate latch when qvel=0 right after reset.
        Latches to True and stays there until the next reset().
        """
        self._stabilize_step_count += 1
        if not self._stabilized and self._stabilize_step_count >= MIN_STABILIZE_STEPS:
            self._stabilized = bool(
                np.max(np.abs(self.mjData.qvel[0:6])) < STABILIZE_VEL_THRESHOLD
            )
        return self._stabilized

    def setup_control(self, mjData, mjModel, **kwargs)-> Tuple[dict[str, Any], dict[str, Any]]:
        """ Called after environment reset, ... 
        """
        self.mjData = mjData
        # self.mjData.qvel = 0
        self.mjModel = mjModel

        # adapted from base_sim.init_scene
        self.torso_index = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = mjModel.body(self.root_body).id
        self.joint_class_map = self._get_dof_indices_by_class(kwargs["mjSpec"], mjModel)
        self.perform_sysid_search = self.sonic_config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            mjModel.joint(i).name for i in range(mjModel.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            mjModel.joint(i).name for i in range(mjModel.njnt)
        ]

        # Enable the elastic band
        if self.sonic_config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand(point=np.array(self.spawn_pose.position[:2] + [1.0]))
            if "g1" in self.sonic_config["ROBOT_TYPE"]:
                if self.sonic_config["enable_waist"]:
                    self.band_attached_link = mjModel.body("pelvis").id
                else:
                    self.band_attached_link = mjModel.body("torso_link").id
            elif "h1" in self.sonic_config["ROBOT_TYPE"]:
                self.band_attached_link = mjModel.body("torso_link").id
            else:
                self.band_attached_link = mjModel.body("base_link").id
 
        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        body_joint_index = []
        left_hand_index = []
        right_hand_index = []
        for i in range(self.mjModel.njnt):
            name = self.mjModel.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ) and name in self.joint_names:
                body_joint_index.append(i)
            elif "left_hand" in name:
                left_hand_index.append(i)
            elif "right_hand" in name:
                right_hand_index.append(i)
        
        assert len(body_joint_index) == self.sonic_robot.NUM_JOINTS
        assert len(left_hand_index) == self.sonic_robot.NUM_HAND_JOINTS
        assert len(right_hand_index) == self.sonic_robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(body_joint_index)
        self.left_hand_index = np.array(left_hand_index)
        self.right_hand_index = np.array(right_hand_index)

        actuators = {}
        joints = {}
        
        for name in self.joint_names:
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
    
    def _get_dof_indices_by_class(self, mjSpec, mjModel) -> dict[str, List[int]]:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            # mujoco.mj_saveLastXML(f.name, mjModel)
            f.write(mjSpec.to_xml())
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        mjModel, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = mjModel.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map
    
    @property
    def pelvis_vz(self) -> float:
        return self.mjData.qvel[2]

    @property
    def pelvis_z(self) -> float:
        return self.mjData.qpos[2]

    def prepare_obs(self) -> Dict[str, Any]:
        obs = {}
        if self.use_floating_root_link: # move to this class
            obs["floating_base_pose"] = self.mjData.qpos[:7]
            obs["floating_base_vel"] = self.mjData.qvel[:6]
            obs["floating_base_acc"] = self.mjData.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mjData.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mjModel.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mjModel, self.mjData, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mjData.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mjData.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mjData.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mjData.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mjData.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mjData.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mjData.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mjData.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mjData.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mjData.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mjData.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mjData.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mjData.time
        return obs # joints in mjcf order (thumb, middle, index)
    
    def get_actuators_action(self) -> dict[str, float]:
        """ Get the current actuator actions of the robot. """
        return {a: v.ctrl[0] for a,v in self.actuators.items()}

    def get_robot_qpos(self) -> dict[str, float]:
        """ Get the current joint positions of the robot. """
        # return np.array([j.qpos[0] for j in self.joints])
        return {j: v.qpos[0] for j,v in self.joints.items()}
    
    def step(self,command,replay=False,eval=False): # FIXME remove
        ...
    
    def get_robot_pose(self):
        return np.round(self.mjData.qpos[:7],3)
    
    def apply_action(self, action_cmd: ActionCmd) -> None:
        assert isinstance(self.controller, WholeBodyEEFController)

        match action_cmd.type:
            case "elastic_band":
                pose = np.concatenate(
                    [
                        self.mjData.xpos[self.band_attached_link],
                        self.mjData.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mjModel,
                    self.mjData,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mjData.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)

            case "wbc_torque":
                low_cmd = action_cmd["low_cmd"]
                use_sensor = action_cmd["use_sensor"]
                left_hand_cmd = action_cmd["left_hand_cmd"]
                right_hand_cmd = action_cmd["right_hand_cmd"]

                body_torques = self.compute_body_torques(low_cmd, use_sensor)
                hand_torques = self.compute_hand_torques(left_hand_cmd, right_hand_cmd)

                # -1: actuator array is 0-based while joint indices from the model are 1-based
                self.torques[self.body_joint_index - 1] = body_torques
                if self.num_hand_dof > 0:
                    self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
                    self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

                self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

                if self.sonic_config["FREE_BASE"]:
                    # Prepend 6 zeros for the floating-base root DOF actuators
                    self.mjData.ctrl = np.concatenate((np.zeros(6), self.torques))
                else:
                    self.mjData.ctrl = self.torques

            case "decoupled_wbc":
                target_q = action_cmd["target_q"]  # 29 body joints in actuator order
                # PD position control using per-joint gains from decoupled_wbc config
                kp = np.array(self.sonic_config.get("MOTOR_KP", [100.0] * self.num_body_dof))
                kd = np.array(self.sonic_config.get("MOTOR_KD", [5.0] * self.num_body_dof))

                q_cur = self.mjData.qpos[self.body_joint_index + self.qpos_offset - 1]
                dq_cur = self.mjData.qvel[self.body_joint_index + self.qvel_offset - 1]

                body_torques = kp * (target_q - q_cur) + kd * (0 - dq_cur)
                self.torques[self.body_joint_index - 1] = body_torques

                # Hand PD control (driven by trigger/grip via decoupled WBC teleop IK)
                # Joint order: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
                # Index + middle work together against thumb in a power grip,
                # so their kp is halved to balance grip forces.
                if self.num_hand_dof > 0:
                    left_hand_q = action_cmd["left_hand_q"]
                    right_hand_q = action_cmd["right_hand_q"]
                    hand_kp = np.array([5.0, 5.0, 5.0, 2.5, 2.5, 2.5, 2.5])
                    hand_kd = 1.0
                    if left_hand_q is not None:
                        lh_q_cur = self.mjData.qpos[self.left_hand_index + self.qpos_offset - 1]
                        lh_dq_cur = self.mjData.qvel[self.left_hand_index + self.qvel_offset - 1]
                        self.torques[self.left_hand_index - 1] = hand_kp * (left_hand_q - lh_q_cur) + hand_kd * (0 - lh_dq_cur)
                    if right_hand_q is not None:
                        rh_q_cur = self.mjData.qpos[self.right_hand_index + self.qpos_offset - 1]
                        rh_dq_cur = self.mjData.qvel[self.right_hand_index + self.qvel_offset - 1]
                        self.torques[self.right_hand_index - 1] = hand_kp * (right_hand_q - rh_q_cur) + hand_kd * (0 - rh_dq_cur)

                self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

                if self.sonic_config["FREE_BASE"]:
                    self.mjData.ctrl = np.concatenate((np.zeros(6), self.torques))
                else:
                    self.mjData.ctrl = self.torques

    
    def compute_body_torques(self, low_cmd, use_sensor) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof) # (29,)
        # if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
        for i in range(self.num_body_dof): # self.unitree_bridge.num_body_motor
            if use_sensor: #self.unitree_bridge.use_sensor:
                body_torques[i] = (
                    low_cmd.motor_cmd[i].tau # type:ignore
                    + low_cmd.motor_cmd[i].kp # type:ignore
                    * (low_cmd.motor_cmd[i].q - self.mjData.sensordata[i]) # type:ignore
                    + low_cmd.motor_cmd[i].kd # type:ignore
                    * (
                        low_cmd.motor_cmd[i].dq # type:ignore
                        - self.mjData.sensordata[i + self.num_body_dof]
                    )
                )
            else:
                body_torques[i] = (
                    low_cmd.motor_cmd[i].tau # type:ignore
                    + low_cmd.motor_cmd[i].kp # type:ignore
                    * (
                        low_cmd.motor_cmd[i].q # type:ignore
                        - self.mjData.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                    )
                    + low_cmd.motor_cmd[i].kd # type:ignore
                    * (
                        low_cmd.motor_cmd[i].dq # type:ignore
                        - self.mjData.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                    )
                )
        return body_torques
    
    def compute_hand_torques(self, left_hand_cmd, right_hand_cmd) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        # if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
        for i in range(self.num_hand_dof):# self.unitree_bridge.num_hand_motor
            left_hand_torques[i] = (
                left_hand_cmd.motor_cmd[i].tau # type:ignore
                + left_hand_cmd.motor_cmd[i].kp # type:ignore
                * (
                    left_hand_cmd.motor_cmd[i].q # type:ignore
                    - self.mjData.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                )
                + left_hand_cmd.motor_cmd[i].kd # type:ignore
                * (
                    left_hand_cmd.motor_cmd[i].dq # type:ignore
                    - self.mjData.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                ) # type:ignore
            )
            right_hand_torques[i] = (
                right_hand_cmd.motor_cmd[i].tau # type:ignore
                + right_hand_cmd.motor_cmd[i].kp # type:ignore
                * (
                    right_hand_cmd.motor_cmd[i].q # type:ignore
                    - self.mjData.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                )
                + right_hand_cmd.motor_cmd[i].kd # type:ignore
                * (
                    right_hand_cmd.motor_cmd[i].dq # type:ignore
                    - self.mjData.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                )
            )
        return np.concatenate((left_hand_torques, right_hand_torques))
    
    @property
    def head_cam_link(self) -> str:
        """ Get the head camera link name. """
        return "d435_link"