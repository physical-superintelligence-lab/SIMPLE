# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Replay agent that feeds recorded teleop commands through the decoupled WBC
pipeline with real physics stepping (no state overwriting).
"""

import collections
import time

import numpy as np
import pandas as pd

from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic

# Right-wrist retract appended after the episode data, as (pad step, offset of the wrist from its
# release position in the pelvis frame [m]): wait for the hand to open, slide outward (-y) to pull
# the fingers off the target, then keep moving outward while lifting a little (lifting much higher
# folds the forearm in front of the head camera). Interpolated with smoothstep in between.
RIGHT_WRIST_RETRACT = (
    (5, (0.0, 0.0, 0.0)),
    (20, (0.0, -0.08, 0.03)),
    (40, (0.0, -0.12, 0.06)),
)

# Stand up appended after the episode data: ramp the base height command from its last recorded value
# to the default standing height over this many padded steps (smoothstep), arms and hands held. Pick
# recordings end with the robot still bent over the table, the object lifted only a little.
STAND_UP_STEPS = 40


class ReplayDecoupledAgent:
    """Replays recorded teleop episodes through the decoupled WBC pipeline.

    Instead of live Pico streamer input, teleop commands (upper body targets,
    navigate_command, base_height_command) are read from the recorded dataset.
    The WBC policy re-runs with live sim proprioception, and physics advances
    naturally — objects are NOT set from recorded poses.

    Parameters:
        - num_pad_frames: number of frames to pad with "freeze" action after episode data exhausted
        - retract_right_hand: during the padded frames, move the right wrist away from the target
          (RIGHT_WRIST_RETRACT) instead of freezing it. Recordings stop right at the release, so
          without it the open fingers can keep touching the placed target.
    """

    def __init__(self, robot: G1Sonic, sonic_config: dict, num_pad_frames: int = 10, retract_right_hand: bool = False,
                 stand_up: bool = False):
        self.robot = robot
        self.sim_dt = sonic_config["SIMULATE_DT"]

        # --- Decoupled WBC pipeline (no teleop/streamer) ---
        self._init_decoupled_policy(sonic_config)

        # Episode data (set via load_episode)
        self._episode_data: pd.DataFrame | None = None
        self._data_row_index = 0
        self._num_pad_frames = num_pad_frames
        self._retract_right_hand = retract_right_hand
        self._stand_up = stand_up

    def _init_decoupled_policy(self, sonic_config: dict):
        """Initialize the decoupled WBC pipeline (WBC policy only, no teleop)."""
        from decoupled_wbc.control.robot_model.instantiation.g1 import (
            instantiate_g1_robot_model,
        )
        from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
        from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig

        enable_waist = sonic_config.get("enable_waist", False)
        waist_location = "lower_and_upper_body" if enable_waist else "lower_body"

        self._dwbc_robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=sonic_config.get("high_elbow_pose", False),
        )

        dwbc_config = ControlLoopConfig(
            enable_waist=enable_waist,
            high_elbow_pose=sonic_config.get("high_elbow_pose", False),
        )
        wbc_config = dwbc_config.load_wbc_yaml()
        assert wbc_config["SIMULATE_DT"] == self.sim_dt

        self._wbc_policy = get_wbc_policy(
            "g1", self._dwbc_robot_model, wbc_config,
            init_time=dwbc_config.upper_body_joint_speed,
        )

        self._upper_body_indices = self._dwbc_robot_model.get_joint_group_indices(
            "upper_body"
        )
        self._right_arm_indices = self._dwbc_robot_model.get_joint_group_indices("right_arm")
        self._right_arm_in_upper_body = [self._upper_body_indices.index(i) for i in self._right_arm_indices]
        self._retract_start = None  # (q_full, wrist position, wrist rotation) at the first padded step
        self.last_goal = None  # the goal of the last get_action, for recording
        self._control_frequency = dwbc_config.control_frequency
        self._control_dt = 4 * self.sim_dt  # 0.02 s (50 Hz main loop)
        self._sim_time = 0.0  # the WBC interpolates on this clock, so a replay does not depend on wall time
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._t_start = time.monotonic()

    def load_episode(self, episode_data: pd.DataFrame):
        """Load a new episode's data and reset the policy for replay."""
        self._episode_data = episode_data
        self._data_row_index = 0
        self._retract_start = None
        self.reset_policy()

    # ------------------------------------------------------------------
    # Observation / goal building
    # ------------------------------------------------------------------

    def _build_wbc_observation(self, sim_obs: dict) -> dict:
        """Convert SIMPLE observation dict to decoupled_wbc observation format.

        Copied from PicoDecoupledAgent._build_wbc_observation.
        """
        rm = self._dwbc_robot_model
        obs = {}

        left_hand_q = sim_obs.get("left_hand_q", np.zeros(7))
        right_hand_q = sim_obs.get("right_hand_q", np.zeros(7))
        left_hand_dq = sim_obs.get("left_hand_dq", np.zeros(7))
        right_hand_dq = sim_obs.get("right_hand_dq", np.zeros(7))

        obs["q"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_q"],
            left_hand_actuated_joint_values=left_hand_q,
            right_hand_actuated_joint_values=right_hand_q,
        )
        obs["dq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_dq"],
            left_hand_actuated_joint_values=left_hand_dq,
            right_hand_actuated_joint_values=right_hand_dq,
        )
        obs["ddq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_ddq", np.zeros(29)),
            left_hand_actuated_joint_values=sim_obs.get("left_hand_ddq", np.zeros(7)),
            right_hand_actuated_joint_values=sim_obs.get("right_hand_ddq", np.zeros(7)),
        )
        obs["tau_est"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_tau_est", np.zeros(29)),
            left_hand_actuated_joint_values=sim_obs.get("left_hand_tau_est", np.zeros(7)),
            right_hand_actuated_joint_values=sim_obs.get("right_hand_tau_est", np.zeros(7)),
        )

        obs["floating_base_pose"] = sim_obs["floating_base_pose"]
        obs["floating_base_vel"] = sim_obs["floating_base_vel"]
        obs["floating_base_acc"] = sim_obs.get("floating_base_acc", np.zeros(6))

        obs["torso_quat"] = sim_obs.get("secondary_imu_quat", np.array([1, 0, 0, 0]))
        obs["torso_ang_vel"] = (
            sim_obs.get("secondary_imu_vel", np.zeros(6))[3:6]
            if "secondary_imu_vel" in sim_obs
            else np.zeros(3)
        )

        obs["wrist_pose"] = sim_obs.get("wrist_pose", np.zeros(14))
        return obs

    def _build_wbc_goal(self, row, t_now: float, is_pad=False) -> dict:
        """Build WBC goal dict from a recorded dataset row."""
        from decoupled_wbc.control.main.constants import (
            DEFAULT_BASE_HEIGHT,
            DEFAULT_NAV_CMD,
        )

        control_freq = self._control_frequency

        if "teleop.navigate_command" in row:
            navigate_cmd = np.asarray(row["teleop.navigate_command"]).copy()
            base_height_cmd = np.atleast_1d(np.asarray(row["teleop.base_height_command"])).copy()
        else:
            # maybe we are replaying the post-processed training data
            navigate_cmd = np.asarray(row["action"][32:]).copy()
            base_height_cmd = np.atleast_1d(np.asarray(row["action"][31])).copy()
        if is_pad:
            navigate_cmd[:3] = 0.0  # zero out (vx, vy, vyaw) for padded steps after data exhausted
            if self._stand_up:
                t = min(1.0, (self._data_row_index - len(self._episode_data) + 1) / STAND_UP_STEPS)
                base_height_cmd += (DEFAULT_BASE_HEIGHT - base_height_cmd) * t * t * (3 - 2 * t)
            # NOTE we assume the upper_body_pose is absolute, otherwise we need zero-padding too

        target_time = t_now + 1 / control_freq

        action_43d_or_36d = np.array(row["action"])
        # action_43d_or_36d = np.array(row["observation.state"])
        if len(action_43d_or_36d) == 43:
            upper_body_pose = action_43d_or_36d[self._upper_body_indices]
        else:
            assert len(action_43d_or_36d) == 36, (
                f"unsupported action dim {len(action_43d_or_36d)}: expected 43 (teleop_decoupled_wbc) or 36 (psi0)")
            # maybe we are replaying the post-processed training data
            # revert the joint postprocess_psi0.py
            # Expected: 
            # [rm.joint_names[idx] for idx in rm._joint_group_indices[group_name]] --> 
            # ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 
            #   'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 
            #   'left_hand_index_0_joint', 'left_hand_index_1_joint', 'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint', 
            #   'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 
            #   right_hand_index_0_joint', 'right_hand_index_1_joint', 'right_hand_middle_0_joint', 'right_hand_middle_1_joint', 'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint'
            # ]
            upper_body_pose = np.concatenate([
                action_43d_or_36d[30:31],  # waist yaw
                action_43d_or_36d[28:30],   # waist roll and pitch
                action_43d_or_36d[14:21],   # left arm 
                # psi0 hands (postprocess_psi0*.py ACTION_SLICES): left = thumb 0:3, middle 3:5,
                # index 5:7 (middle before index); right = thumb 7:10, index 10:12, middle 12:14
                action_43d_or_36d[5:7],    # left hand index
                action_43d_or_36d[3:5],    # left hand middle
                action_43d_or_36d[0:3],    # left hand thumb
                action_43d_or_36d[21:28],   # right arm
                action_43d_or_36d[10:12],   # right hand index
                action_43d_or_36d[12:14],   # right hand middle
                action_43d_or_36d[7:10],    # right hand thumb
            ])

        goal = {
            "target_upper_body_pose": upper_body_pose,
            "navigate_cmd": navigate_cmd,
            "base_height_command": base_height_cmd,
            "target_time": target_time,
            "interpolation_garbage_collection_time": t_now - 2 / control_freq,
            "timestamp": t_now,
        }
        return goal

    def _retract_right_arm(self, upper_body_pose: np.ndarray, proprio: dict, pad_step: int) -> np.ndarray:
        """Replace the right arm targets in upper_body_pose with the RIGHT_WRIST_RETRACT pose for pad_step."""
        rm = self._dwbc_robot_model
        frame = rm.supplemental_info.hand_frame_names["right"]
        if self._retract_start is None:
            q = rm.get_configuration_from_actuated_joints(
                body_actuated_joint_values=proprio["body_q"],
                left_hand_actuated_joint_values=proprio.get("left_hand_q", np.zeros(7)),
                right_hand_actuated_joint_values=proprio.get("right_hand_q", np.zeros(7)),
            )
            q[self._right_arm_indices] = upper_body_pose[self._right_arm_in_upper_body]
            rm.cache_forward_kinematics(q)
            placement = rm.frame_placement(frame)
            self._retract_start = (q, placement.translation.copy(), placement.rotation.copy())
            self._retract_rest = q[self._right_arm_indices].copy()
        q, p0, rot0 = self._retract_start

        prev_step, prev_offset = 0, np.zeros(3)
        offset = np.asarray(RIGHT_WRIST_RETRACT[-1][1])
        for step, key_offset in RIGHT_WRIST_RETRACT:
            if pad_step < step:
                t = (pad_step - prev_step) / (step - prev_step)
                offset = prev_offset + (np.asarray(key_offset) - prev_offset) * t * t * (3 - 2 * t)
                break
            prev_step, prev_offset = step, np.asarray(key_offset)
        q = self._wrist_ik(q, frame, p0 + offset, rot0)
        self._retract_start = (q, p0, rot0)

        upper_body_pose = upper_body_pose.copy()
        upper_body_pose[self._right_arm_in_upper_body] = q[self._right_arm_indices]
        return upper_body_pose

    def _wrist_ik(self, q: np.ndarray, frame: str, position: np.ndarray, rotation: np.ndarray, iters: int = 5) -> np.ndarray:
        """Damped least-squares IK on the right arm joints (finite-difference Jacobian), warm-started at q.

        The target is partly out of reach from the release pose, so the step is regularized toward
        the arm pose at the release; otherwise shoulder/wrist yaw wind up and twist the hand.
        """
        import pinocchio as pin

        rm = self._dwbc_robot_model
        idx = self._right_arm_indices
        q_rest = self._retract_rest

        def error(q):
            rm.cache_forward_kinematics(q)
            m = rm.frame_placement(frame)
            # orientation weighted down: keep the hand roughly level, position matters
            return np.concatenate([position - m.translation, 0.3 * pin.log3(rotation @ m.rotation.T)])

        q = q.copy()
        lower, upper = rm.lower_joint_limits[idx], rm.upper_joint_limits[idx]
        for _ in range(iters):
            e = error(q)
            jac = np.zeros((6, len(idx)))
            for j, i in enumerate(idx):
                dq = q.copy()
                dq[i] += 1e-4
                jac[:, j] = (e - error(dq)) / 1e-4
            step = np.linalg.solve(jac.T @ jac + (1e-4 + 3e-3) * np.eye(len(idx)), jac.T @ e + 3e-3 * (q_rest - q[idx]))
            q[idx] = np.clip(q[idx] + np.clip(step, -0.05, 0.05), lower, upper)
        return q

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def get_stabilize_action(self, observation) -> ActionCmd:
        """Run the WBC pipeline ramping to the WBC default pose — used during stabilization.

        Drives the upper body toward the WBC default configuration (forearms pointing
        forward), matching what teleop does before the operator activates the teleop policy.
        Does NOT read from episode data or advance _data_row_index.
        """
        # Always run the full WBC pipeline (main loop now at 50Hz)
        from decoupled_wbc.control.main.constants import (
            DEFAULT_BASE_HEIGHT,
            DEFAULT_NAV_CMD,
        )
        t_now = self._tick()
        control_freq = self._control_frequency

        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)

        # Use the WBC's own default upper body pose (forearms forward), not the
        # MJCF keyframe defaults (arms down).  On the first step give a 2s ramp
        # window — matching teleop's initial `target_time = t_now + 2.0` — so the
        # arms move smoothly to the default rather than snapping.
        is_first_step = self._cached_target_q is None
        default_upper_body = self._dwbc_robot_model.get_initial_upper_body_pose()
        goal = {
            "target_upper_body_pose": default_upper_body,
            "navigate_cmd": np.asarray(DEFAULT_NAV_CMD),
            "base_height_command": np.atleast_1d(np.asarray(DEFAULT_BASE_HEIGHT)),
            "target_time": t_now + (2.0 if is_first_step else 1 / control_freq),
            "interpolation_garbage_collection_time": t_now - 2 / control_freq,
            "timestamp": t_now,
        }
        self._wbc_policy.set_goal(goal)

        wbc_action = self._wbc_policy.get_action(time=t_now)
        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")

        return ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )

    def _tick(self) -> float:
        """Current simulated time; advances the clock by one control step."""
        t_now = self._sim_time
        self._sim_time += self._control_dt
        return t_now

    def get_action(self, observation, **kwargs) -> ActionCmd:
        """Compute WBC action from recorded teleop commands + live proprioception.

        Data row advancement is managed by the main loop (replay_decoupled_wbc.py).
        This method always runs the full WBC pipeline (main loop now at 50Hz).
        """
        if self._episode_data is None:
            raise RuntimeError("No episode data loaded. Call load_episode() first.")

        # Check if episode data exhausted
        # pad 10 frames with "freeze" action
        if self._data_row_index >= len(self._episode_data) + self._num_pad_frames:
            raise StopIteration("Episode data exhausted")

        # Always run the full WBC pipeline (main loop now at 50Hz)
        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)

        # replay with recorded teleop commands as goals
        if self._data_row_index >= len(self._episode_data):
            # After data exhausted, keep running WBC with the last teleop command as goal
            # (which should be a stable grasp pose for pick-and-place)
            current_row = self._episode_data.iloc[-1]
            is_pad = True
        else:
            current_row = self._episode_data.iloc[self._data_row_index]
            is_pad = False

        t_now = self._tick()
        goal = self._build_wbc_goal(current_row, t_now, is_pad=is_pad)
        if is_pad and self._retract_right_hand:
            pad_step = self._data_row_index - len(self._episode_data)
            goal["target_upper_body_pose"] = self._retract_right_arm(goal["target_upper_body_pose"], proprio, pad_step)
        self._wbc_policy.set_goal(goal)
        self.last_goal = goal

        wbc_action = self._wbc_policy.get_action(time=t_now)

        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(
            wbc_action["q"]
        )
        self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(
            wbc_action["q"], side="left"
        )
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(
            wbc_action["q"], side="right"
        )

        # # Reverse (full config → components)
        # self._cached_target_q = self._wbc_policy.robot_model.get_body_actuated_joints(current_row.action)                                                                                                                                                                                     
        # self._cached_left_hand_q = self._wbc_policy.robot_model.get_hand_actuated_joints(current_row.action, side="left")
        # self._cached_right_hand_q = self._wbc_policy.robot_model.get_hand_actuated_joints(current_row.action, side="right") 

        # if self._data_row_index <= 10:
        self._data_row_index += 1
        return ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )

    def reset_policy(self):
        """Reset the WBC pipeline for a new episode.

        Same as PicoDecoupledAgent.reset_policy but without teleop policy reset.
        """
        self._sim_time = 0.0
        t_now = self._sim_time

        # Reset entire decoupled WBC pipeline (upper and lower body policies)
        self._wbc_policy.reset(init_time=t_now)

        # Clear cached joint targets
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._data_row_index = 0
        self.last_goal = None
        self._teleop_initialized = False
        self._teleop_was_active = False
        self._teleop_activate_time = None
        self._last_teleop_action = {}
        self._t_start = time.monotonic()

    def close(self):
        pass
