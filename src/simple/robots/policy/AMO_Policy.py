# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import types
import numpy as np
import mujoco
import glfw
from collections import deque
import torch
import os
import transforms3d as t3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# A repo cloned with GIT_LFS_SKIP_SMUDGE=1, or without git-lfs installed at all,
# leaves a ~130 byte text pointer where each checkpoint should be. Loading one
# raises an error that never mentions LFS, so detect it by its magic line.
_LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def _ensure_materialized(path):
    """Raise a diagnostic error if `path` is missing or an unresolved LFS pointer."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model checkpoint: {path}")
    with open(path, "rb") as handle:
        head = handle.read(len(_LFS_POINTER_MAGIC))
    if head == _LFS_POINTER_MAGIC:
        raise RuntimeError(
            f"{path} is an unresolved Git LFS pointer ({os.path.getsize(path)} bytes) "
            "rather than the real checkpoint. Install git-lfs and run `git lfs pull` "
            "from the repo root to download it."
        )


def _load_jit(path, device):
    """Load a TorchScript checkpoint, reporting *which* file failed and why."""
    _ensure_materialized(path)
    try:
        return torch.jit.load(path, map_location=device)
    except Exception as exc:  # corrupt/truncated file, torch version mismatch, ...
        raise RuntimeError(f"Failed to load TorchScript checkpoint {path}: {exc}") from exc

def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec


class AMO_Policy:
    # Start from the spawn pose instead of zero history, for a robot that spawns in its settled stance
    # (G1Wholebody turns it on); False is the start from the all-zero pose.
    warm_start: bool = False

    def __init__(self, robot_type="g1", device="cuda", joint_names:list[str] | None = None):
        self.robot_type = robot_type
        self.device = device
        
        if "dex3" in robot_type:
            self.joint_names = joint_names

            self.output_joint_names = [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", 
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            ]

            self.arm_input_joint_names = [
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", 
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", 
                "right_shoulder_yaw_joint", "right_elbow_joint",
            ]

            self.arm_input_joint_indices = [
                self.joint_names.index(jname) for jname in self.arm_input_joint_names
            ]

            self.action_joint_names = self.output_joint_names + self.arm_input_joint_names
            self.action_joint_indices = [
                self.joint_names.index(jname) for jname in self.action_joint_names
            ]
            
            # action : lower body + torso
            self.num_actions = len(self.output_joint_names) 
            self.num_dofs = len(joint_names)    # type: ignore

            self.default_dof_pos = np.array([
                # -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
                # -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,   # waist_yaw, waist_roll, waist_pitch
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # left_hand_thumb_0_joint, left_hand_thumb_1_joint, left_hand_thumb_2_joint, left_hand_index_0_joint, left_hand_index_1_joint, left_hand_middle_0_joint, left_hand_middle_1_joint
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # right_hand_thumb_0_joint, right_hand_thumb_1_joint, right_hand_thumb_2_joint, right_hand_index_0_joint, right_hand_index_1_joint, right_hand_middle_0_joint, right_hand_middle_1_joint
            ])
            
            self.default_dof_pos_for_policy = np.concatenate([
                self.default_dof_pos[:15],
                self.default_dof_pos[self.arm_input_joint_indices],
               
            ])
            self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                # 25, 25, 25, 25,25,25,25,
                # 25, 25, 25, 25,25,25,25,
            ])


        elif "inspire" in robot_type:
            self.joint_names = joint_names

            self.output_joint_names = [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", 
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            ]

            self.arm_input_joint_names = [
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", 
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", 
                "right_shoulder_yaw_joint", "right_elbow_joint",
            ]

            self.arm_input_joint_indices = [
                self.joint_names.index(jname) for jname in self.arm_input_joint_names
            ]

            self.action_joint_names = self.output_joint_names + self.arm_input_joint_names
            self.action_joint_indices = [
                self.joint_names.index(jname) for jname in self.action_joint_names
            ]
            
            # action : lower body + torso
            self.num_actions = len(self.output_joint_names) 
            self.num_dofs = len(joint_names)    # type: ignore

            # G1 Inspire
            self.default_dof_pos = np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
                0.0, 0.0, 0.0,   # waist_yaw, waist_roll, waist_pitch
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # L_hand_base_link, L_thumb_distal, L_index_intermediate, L_middle_intermediate, L_ring_intermediate, L_pinky_intermediate
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # R_hand_base_link, R_thumb_distal, R_index_intermediate, R_middle_intermediate, R_ring_intermediate, R_pinky_intermediate
            ])
            
            self.default_dof_pos_for_policy = np.concatenate([
                self.default_dof_pos[:15],
                self.default_dof_pos[self.arm_input_joint_indices],
               
            ])
            self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                # 25, 25, 25, 25,25,25,25,
                # 25, 25, 25, 25,25,25,25,
            ])

        
        else:
            raise ValueError(f"Robot type {robot_type} not supported!")
        
        self.action_scale = 0.25

        self.arm_action = np.concatenate([
            self.default_dof_pos[self.arm_input_joint_indices],
        ])
        # self.prev_arm_action = self.default_dof_pos[15:]
        self.prev_arm_action = np.concatenate([
            self.default_dof_pos[self.arm_input_joint_indices],
        ])
        
        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05

        # DEFINE OBSERVATION
        self.nj = self.num_dofs
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8

        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.nj)
        
        self.last_action_for_policy = self.last_action[self.action_joint_indices]

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3, ))#arm + commond input
        self.demo_obs_template[:self._n_demo_dof] = np.concatenate([
            self.default_dof_pos[self.arm_input_joint_indices],
        ])
        self.demo_obs_template[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75

        self.target_yaw = 0.0 
        # vx, vy, 0, torso yaw/pitch/roll, [base height] * 3
        self.obs_command = np.array([0, 0, 0, 0, -0.15, 0, 0.75, 0.75, 0.75], dtype=np.float32)
        self.obs_turning_flag = np.array([0,], dtype=np.float32)

        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for i in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for i in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

        # Let load failures propagate: the eval worker wraps this call and pickles
        # the traceback back to the parent. Swallowing them and calling exit(0) made
        # a missing checkpoint look like a clean, successful worker exit.
        self.policy_jit = _load_jit(os.path.join(BASE_DIR, "amo_jit.pt"), device)
        self.adapter = _load_jit(os.path.join(BASE_DIR, "adapter_jit.pt"), device)

        self.adapter.eval()

        for param in self.adapter.parameters():
            param.requires_grad = False

        norm_stats_path = os.path.join(BASE_DIR, "adapter_norm_stats.pt")
        _ensure_materialized(norm_stats_path)
        norm_stats = torch.load(norm_stats_path, weights_only=False)
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=device, dtype=torch.float32)
       
        self.adapter_input = torch.zeros((1, 8 + 4), device=device, dtype=torch.float32)
        # adapter output:  waist qpos + leg qpos
        self.adapter_output = torch.zeros((1, 15), device=device, dtype=torch.float32)

        self._initial_quat = None
        self._last_target_yaw = 0.0
        self._last_commands = np.zeros((8,), dtype=np.float32)
        # self._init_yaw = None


    def get_observation(self, joints, actuators, mjdata, commands):
        self.dof_pos = np.array([joints[j].qpos.item() for j in self.action_joint_names], dtype=np.float32)
        self.dof_vel = np.array([joints[j].qvel.item() for j in self.action_joint_names], dtype=np.float32)
        first_step = self._initial_quat is None
        if first_step:
            # heading reference: the initial yaw only, so a spawn tilted to the settled stance reads as tilted
            init_yaw = t3d.euler.quat2euler(mjdata.sensor('orientation').data)[2]
            self._initial_quat = np.asarray(t3d.euler.euler2quat(0.0, 0.0, init_yaw), dtype=np.float32)
            # warm start: as if the policy had been holding the spawn pose (the robot spawns in its stance)
            if self.warm_start:
                self.last_action_for_policy = (self.dof_pos - self.default_dof_pos_for_policy) / self.action_scale
        #get quat relative to initial quat
        self.quat = t3d.quaternions.qmult(t3d.quaternions.qinverse(self._initial_quat), mjdata.sensor('orientation').data.astype(np.float32))
        self.ang_vel = mjdata.sensor('angular-velocity').data.astype(np.float32)

        rpy = quatToEuler(self.quat)
        #for lerobot
        self.obs_rpy = rpy.copy()

        # TODO add command control input
        self.target_yaw = commands[1]
        self.obs_target_yaw = np.array([self.target_yaw], dtype=np.float32)
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        # if self._init_yaw is None:
        #     self._init_yaw = rpy[2]
        if self._in_place_stand_flag:
            dyaw = 0.0
        # for lerobot
        self.obs_turning_flag = np.array([commands[7]], dtype=np.float32)

        obs_dof_vel = self.dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0
        
        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)
        # height, torso vyaw, torso pitch, torso roll and 8 arm qpos
        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])
        
        self.adapter_input[0] = 0.75 + commands[3]
        self.adapter_input[1] = commands[4]
        self.adapter_input[2] = commands[5]
        self.adapter_input[3] = commands[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)

        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        # get obs proprio

        self.obs_prop = np.concatenate([
            self.ang_vel * self.scales_ang_vel,
            rpy[:2],
            (np.sin(dyaw),
            np.cos(dyaw)),
            (self.dof_pos - self.default_dof_pos_for_policy),
            self.dof_vel * self.scales_dof_vel,
            self.last_action_for_policy,
            gait_obs,
            self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv, ))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]    # 8 arm qpos
        obs_demo[self._n_demo_dof] = commands[0]    # VX
        if commands[7] > 0.5:
            print(f"vx={commands[0]}, current yaw={rpy[2]}")
        obs_demo[self._n_demo_dof+1] = commands[2]    # Vy
        # self._in_place_stand_flag = np.abs(commands[0]) < 0.07 and np.abs(commands[2]) < 0.1 and np.abs(self.ang_vel[2]) < 0.1
        if abs(self.target_yaw) == 3.14 and np.sign(rpy[2]) < 0:
            self.target_yaw = -3.14
        elif abs(self.target_yaw) == 3.14 and np.sign(rpy[2]) > 0:
            self.target_yaw = 3.14
        self._in_place_stand_flag = (
            np.abs(commands[0]) < 0.1 and 
            np.abs(commands[2]) < 0.1 and 
            commands[7] < 0.5
            # (
            #     np.abs(rpy[2] - self.target_yaw) < 0.1 #or np.abs(self._init_yaw) > 0.1
            # )
        )

        obs_demo[self._n_demo_dof+3] = commands[4]    # torso yaw
        obs_demo[self._n_demo_dof+4] = commands[5]    # torso pitch
        obs_demo[self._n_demo_dof+5] = commands[6]    # torso roll
        obs_demo[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75 + commands[3]    # height

        #for lerobot 
        self.obs_command = obs_demo.copy()[self._n_demo_dof:]

        if first_step and self.warm_start:
            for _ in range(self.history_len):
                self.proprio_history_buf.append(self.obs_prop)
            for _ in range(self.extra_history_len):
                self.extra_history_buf.append(self.obs_prop)
        self.proprio_history_buf.append(self.obs_prop)
        self.extra_history_buf.append(self.obs_prop)
        self._last_target_yaw = self.target_yaw
        return np.concatenate((self.obs_prop, obs_demo, obs_priv, obs_hist))
        

    def get_action(self, joints, actuators, mjdata, commands):
        self._last_commands = commands
        self.obs = self.get_observation(joints, actuators, mjdata, commands)
        obs_tensor = torch.from_numpy(self.obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            extra_hist = torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float).view(1, -1).to(self.device)
            raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
        raw_action = np.clip(raw_action, -40., 40.)
        self.last_action_for_policy = np.concatenate([raw_action.copy(), (self.dof_pos - self.default_dof_pos_for_policy)[15:] / self.action_scale])
        scaled_actions = raw_action * self.action_scale

        pd_target = scaled_actions + self.default_dof_pos_for_policy[:15]
        self.gait_cycle = np.remainder(self.gait_cycle + 0.02 * self.gait_freq, 1.0) # TODO hard-code amo's control dt here
        if self._in_place_stand_flag and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.25])
        if (not self._in_place_stand_flag) and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.75])

        return pd_target, self.output_joint_names
    




    def get_eval_observation(self, joints, actuators, mjdata, commands):
        self.dof_pos = np.array([joints[j].qpos.item() for j in self.action_joint_names], dtype=np.float32)
        self.dof_vel = np.array([joints[j].qvel.item() for j in self.action_joint_names], dtype=np.float32)
        first_step = self._initial_quat is None
        if first_step:
            # heading reference: the initial yaw only, so a spawn tilted to the settled stance reads as tilted
            init_yaw = t3d.euler.quat2euler(mjdata.sensor('orientation').data)[2]
            self._initial_quat = np.asarray(t3d.euler.euler2quat(0.0, 0.0, init_yaw), dtype=np.float32)
            # warm start: as if the policy had been holding the spawn pose (the robot spawns in its stance)
            if self.warm_start:
                self.last_action_for_policy = (self.dof_pos - self.default_dof_pos_for_policy) / self.action_scale
        #get quat relative to initial quat
        self.quat = t3d.quaternions.qmult(t3d.quaternions.qinverse(self._initial_quat), mjdata.sensor('orientation').data.astype(np.float32))
        self.ang_vel = mjdata.sensor('angular-velocity').data.astype(np.float32)

        rpy = quatToEuler(self.quat)
        #for lerobot
        self.obs_rpy = rpy.copy()

        # TODO add command control input
        self.target_yaw = commands[1]
        self.obs_target_yaw = np.array([self.target_yaw], dtype=np.float32)
        # if self._init_yaw is None:
        #     self._init_yaw = rpy[2]
        
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0
        # for lerobot
        self.obs_turning_flag = np.array([commands[7]], dtype=np.float32)

        obs_dof_vel = self.dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0
        
        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)
        # height, torso vyaw, torso pitch, torso roll and 8 arm qpos
        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])
        
        self.adapter_input[0] = 0.75 + commands[3]
        self.adapter_input[1] = commands[4]
        self.adapter_input[2] = commands[5]
        self.adapter_input[3] = commands[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)

        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        # get obs proprio

        self.obs_prop = np.concatenate([
            self.ang_vel * self.scales_ang_vel,
            rpy[:2],
            (np.sin(dyaw),
            np.cos(dyaw)),
            (self.dof_pos - self.default_dof_pos_for_policy),
            self.dof_vel * self.scales_dof_vel,
            self.last_action_for_policy,
            gait_obs,
            self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv, ))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]    # 8 arm qpos
        obs_demo[self._n_demo_dof] = commands[0]    # VX
        obs_demo[self._n_demo_dof+1] = commands[2]    # Vy
        self._in_place_stand_flag = (
            np.abs(commands[0]) < 0.1 and 
            np.abs(commands[2]) < 0.1 and 
            commands[7] < 0.5
            # (
            #     np.abs(rpy[2] - self.target_yaw) < 0.1 
            # )
        )
        # print(f"self._in_place_stand_flag: {self._in_place_stand_flag}")
        obs_demo[self._n_demo_dof+3] = commands[4]    # torso yaw
        obs_demo[self._n_demo_dof+4] = commands[5]    # torso pitch
        obs_demo[self._n_demo_dof+5] = commands[6]    # torso roll
        obs_demo[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75 + commands[3]    # height

        #for lerobot 
        self.obs_command = obs_demo.copy()[8:]

        if first_step and self.warm_start:
            for _ in range(self.history_len):
                self.proprio_history_buf.append(self.obs_prop)
            for _ in range(self.extra_history_len):
                self.extra_history_buf.append(self.obs_prop)
        self.proprio_history_buf.append(self.obs_prop)
        self.extra_history_buf.append(self.obs_prop)
        self._last_target_yaw = self.target_yaw
        return np.concatenate((self.obs_prop, obs_demo, obs_priv, obs_hist))


    def get_eval_action(self, joints, actuators, mjdata, commands):
        self.obs = self.get_eval_observation(joints, actuators, mjdata, commands)
        obs_tensor = torch.from_numpy(self.obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            extra_hist = torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float).view(1, -1).to(self.device)
            raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
        raw_action = np.clip(raw_action, -40., 40.)
        self.last_action_for_policy = np.concatenate([raw_action.copy(), (self.dof_pos - self.default_dof_pos_for_policy)[15:] / self.action_scale])
        scaled_actions = raw_action * self.action_scale

        pd_target = scaled_actions + self.default_dof_pos_for_policy[:15]
        self.gait_cycle = np.remainder(self.gait_cycle + 0.02 * self.gait_freq, 1.0)
        if self._in_place_stand_flag and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.25])
        if (not self._in_place_stand_flag) and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.75])

        return pd_target, self.output_joint_names
