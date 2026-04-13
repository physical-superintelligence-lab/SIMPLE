"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Replay agent that feeds recorded teleop commands through the decoupled WBC
pipeline with real physics stepping (no state overwriting).

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import collections
import time

import numpy as np
import pandas as pd

from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic


class ReplayDecoupledAgent:
    """Replays recorded teleop episodes through the decoupled WBC pipeline.

    Instead of live Pico streamer input, teleop commands (upper body targets,
    navigate_command, base_height_command) are read from the recorded dataset.
    The WBC policy re-runs with live sim proprioception, and physics advances
    naturally — objects are NOT set from recorded poses.
    """

    def __init__(self, robot: G1Sonic, sonic_config: dict):
        self.robot = robot
        self.sim_dt = sonic_config["SIMULATE_DT"]

        # --- Decoupled WBC pipeline (no teleop/streamer) ---
        self._init_decoupled_policy(sonic_config)

        # Episode data (set via load_episode)
        self._episode_data: pd.DataFrame | None = None
        self._data_row_index = 0

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
        self._control_frequency = dwbc_config.control_frequency
        self._control_dt = 4 * self.sim_dt  # 0.02 s (50 Hz main loop)
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._t_start = time.monotonic()

    def load_episode(self, episode_data: pd.DataFrame):
        """Load a new episode's data and reset the policy for replay."""
        self._episode_data = episode_data
        self._data_row_index = 0
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

    def _build_wbc_goal(self, row) -> dict:
        """Build WBC goal dict from a recorded dataset row."""
        from decoupled_wbc.control.main.constants import (
            DEFAULT_BASE_HEIGHT,
            DEFAULT_NAV_CMD,
        )

        t_now = time.monotonic()
        control_freq = self._control_frequency

        navigate_cmd = row["teleop.navigate_command"]
        base_height_cmd = row["teleop.base_height_command"]

        target_time = t_now + 1 / control_freq

        action_43d = np.array(row["action"])
        # action_43d = np.array(row["observation.state"])
        upper_body_pose = action_43d[self._upper_body_indices]

        goal = {
            "target_upper_body_pose": upper_body_pose,
            "navigate_cmd": np.asarray(navigate_cmd),
            "base_height_command": np.atleast_1d(np.asarray(base_height_cmd)),
            "target_time": target_time,
            "interpolation_garbage_collection_time": t_now - 2 / control_freq,
            "timestamp": t_now,
        }
        return goal

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
        t_now = time.monotonic()
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

    def get_action(self, observation, **kwargs) -> ActionCmd:
        """Compute WBC action from recorded teleop commands + live proprioception.

        Data row advancement is managed by the main loop (replay_decoupled_wbc.py).
        This method always runs the full WBC pipeline (main loop now at 50Hz).
        """
        if self._episode_data is None:
            raise RuntimeError("No episode data loaded. Call load_episode() first.")

        # Check if episode data exhausted
        if self._data_row_index >= len(self._episode_data):
            raise StopIteration("Episode data exhausted")

        # Always run the full WBC pipeline (main loop now at 50Hz)
        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)

        # replay with recorded teleop commands as goals
        current_row = self._episode_data.iloc[self._data_row_index]
        goal = self._build_wbc_goal(current_row)
        self._wbc_policy.set_goal(goal)

        t_now = time.monotonic()
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
        t_now = time.monotonic()

        # Reset entire decoupled WBC pipeline (upper and lower body policies)
        self._wbc_policy.reset(init_time=t_now)

        # Clear cached joint targets
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._data_row_index = 0
        self._teleop_initialized = False
        self._teleop_was_active = False
        self._teleop_activate_time = None
        self._last_teleop_action = {}
        self._t_start = time.monotonic()

    def close(self):
        pass
