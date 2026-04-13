"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import time
import cv2
import numpy as np

from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic
from simple.teleop.pico.tcp_server import TCPControlServer
from simple.teleop.pico.tcp_video_sender import TCPVideoSender
from simple.teleop.pico.streaming import FrameBuffer, StreamingThread
from .sonic_wbc_agent import SonicWbcAgent


class PicoDecoupledAgent(SonicWbcAgent):
    """Teleoperation agent that uses decoupled WBC policy from decoupled_wbc.

    Instead of forwarding raw Unitree SDK torque commands (like PicoSonicAgent),
    this agent runs the full decoupled whole-body control pipeline internally:
      PicoStreamer -> TeleopPolicy (IK) -> G1DecoupledWholeBodyPolicy -> joint positions

    The PicoStreamer (from decoupled_wbc) reads XRoboToolkit directly — no
    pico_manager or ZMQ needed.  Sim-control signals (drop_robot, reset_env)
    are derived from the Pico controller buttons read by the same streamer.
    """

    def __init__(self, robot: G1Sonic):
        super().__init__(robot)

        self.episodes_saved = 0
        self.num_episodes = 100
        self.image_publish_process = None
        self.sim_dt = self.robot.sonic_config["SIMULATE_DT"]

        # Controlled drop state (same as PicoSonicAgent)
        self._dropping = False
        self._drop_rate = 0.15  # m/s
        self._reset_requested = False

        # Edge-detection state for button combos
        self._drop_btn_last = False
        self._reset_btn_last = False

        # --- Pico VR streaming (camera feed to headset) ---
        self._init_pico_streamer()

        # --- Decoupled WBC pipeline ---
        self._init_decoupled_policy()

    @property
    def reset_requested(self) -> bool:
        """True if a reset was requested via controller buttons. Clears on read."""
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_pico_streamer(self):
        """Set up TCP server for VR camera streaming."""
        self._streaming: StreamingThread | None = None
        self._frame_buffer = FrameBuffer()

        tcp_server = TCPControlServer("0.0.0.0:13579")

        def on_open_camera(camera_req):
            print(f"[PicoDecoupled] OPEN_CAMERA: {camera_req}")
            if self._streaming and self._streaming.is_running():
                print("[PicoDecoupled] Already streaming, ignoring duplicate OPEN_CAMERA")
                return

            fps = camera_req.get("fps") or 60
            width = camera_req.get("width") or 2560
            height = camera_req.get("height") or 720
            bitrate = camera_req.get("bitrate") or 4_000_000
            hevc = bool(camera_req.get("enableMvHevc"))
            ip = camera_req.get("ip")
            port = camera_req.get("port")

            if not ip or not port:
                print("[PicoDecoupled] OPEN_CAMERA missing ip/port, cannot stream")
                return
            try:
                sender = TCPVideoSender(
                    ip=ip, port=port,
                    width=width, height=height, fps=fps,
                    bitrate=bitrate, hevc=hevc,
                )
            except ConnectionRefusedError:
                print(f"[PicoDecoupled] Connection refused to {ip}:{port}")
                return

            self._streaming = StreamingThread(
                frame_buffer=self._frame_buffer,
                fps=fps,
                publishers=[sender],
                on_ended=lambda: tcp_server.close_client(),
            )
            self._streaming.start()

        def on_close_camera():
            print("[PicoDecoupled] CLOSE_CAMERA received")
            if self._streaming:
                self._streaming.stop()
                self._streaming = None
            tcp_server.close_client()

        tcp_server.on_open_camera = on_open_camera
        tcp_server.on_close_camera = on_close_camera
        tcp_server.start()

    def _init_decoupled_policy(self):
        """Initialize the decoupled WBC pipeline (streamer + teleop IK + WBC policy)."""
        from decoupled_wbc.control.robot_model.instantiation.g1 import (
            instantiate_g1_robot_model,
        )
        from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
            instantiate_g1_hand_ik_solver,
        )
        from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
        from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
        from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
        from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig

        sonic_cfg = self.robot.sonic_config
        enable_waist = sonic_cfg.get("enable_waist", False)
        waist_location = "lower_and_upper_body" if enable_waist else "lower_body"

        self._dwbc_robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=sonic_cfg.get("high_elbow_pose", False),
        )

        # Sanity check: the robot_model's actuated joints should match the sim's joint layout
        supp = self._dwbc_robot_model.supplemental_info
        assert supp.body_actuated_joints == self.robot.joint_names[:29] # type:ignore
        assert supp.left_hand_actuated_joints == self.robot.hand_names[:7] # type:ignore
        assert supp.right_hand_actuated_joints == self.robot.hand_names[7:14] # type:ignore

        # Load the decoupled_wbc's own config (g1_29dof_gear_wbc.yaml) via ControlLoopConfig.
        # SIMPLE's sonic_config has VERSION=sonic_model12 which is incompatible with
        # get_wbc_policy (expects gear_wbc). The decoupled_wbc config is self-contained.
        dwbc_config = ControlLoopConfig(
            enable_waist=enable_waist,
            high_elbow_pose=sonic_cfg.get("high_elbow_pose", False),
        )
        wbc_config = dwbc_config.load_wbc_yaml()

        assert wbc_config["SIMULATE_DT"] == self.sim_dt

        # WBC policy (lower body RL + upper body interpolation)
        self._wbc_policy = get_wbc_policy(
            "g1", self._dwbc_robot_model, wbc_config,
            init_time=dwbc_config.upper_body_joint_speed,
        )

        # Teleop policy (Pico streamer + retargeting IK)
        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        retargeting_ik = TeleopRetargetingIK(
            robot_model=self._dwbc_robot_model,
            left_hand_ik_solver=left_hand_ik,
            right_hand_ik_solver=right_hand_ik,
            body_active_joint_groups=["upper_body"],
        )

        body_control_device = sonic_cfg.get("body_control_device", "pico")
        hand_control_device = sonic_cfg.get("hand_control_device", "pico")
        body_streamer_ip = sonic_cfg.get("body_streamer_ip", "192.168.0.1")

        self._teleop_policy = TeleopPolicy(
            robot_model=self._dwbc_robot_model,
            retargeting_ik=retargeting_ik,
            body_control_device=body_control_device,
            hand_control_device=hand_control_device,
            body_streamer_ip=body_streamer_ip,
            activate_keyboard_listener=False,
        )

        # Keep a reference to the underlying PicoStreamer so we can read
        # raw button states for drop_robot / reset_env detection.
        self._pico_streamer = self._teleop_policy.teleop_streamer.body_streamer

        self._teleop_initialized = False
        self._teleop_was_active = False
        self._teleop_activate_time: float | None = None
        self._t_start = time.monotonic()
        self._control_frequency = dwbc_config.control_frequency

        # Seconds over which target_time decays from engage-window to normal 1/freq
        self._arm_engage_smooth_secs = 1.0

        # Control timestep at 50 Hz (main loop runs at this frequency)
        self._control_dt = 4 * self.sim_dt  # 0.02 s
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None

    # ------------------------------------------------------------------
    # Button helpers (read directly from PicoStreamer's XrClient)
    # ------------------------------------------------------------------

    def _poll_pico_buttons(self):
        """Read Pico controller buttons and detect edge-triggered sim commands.

        Matches the pico_manager conventions:
          - right_axis_click          -> drop_robot
          - left_grip + right_grip    -> reset_env
        """
        xr = self._pico_streamer.xr_client

        # --- drop_robot: right joystick click (edge-triggered) ---
        drop_btn = bool(xr.get_button_state_by_name("right_axis_click"))
        if drop_btn and not self._drop_btn_last:
            if (
                self.robot.elastic_band
                and self.robot.elastic_band.enable
                and not self._dropping
            ):
                self._dropping = True
                print("[PicoDecoupled] Controlled drop started (right stick click)")
        self._drop_btn_last = drop_btn

        # --- reset_env: left_grip + right_grip held simultaneously (edge) ---
        left_grip = xr.get_key_value_by_name("left_grip") > 0.5
        right_grip = xr.get_key_value_by_name("right_grip") > 0.5
        reset_btn = left_grip and right_grip
        if reset_btn and not self._reset_btn_last:
            self._reset_requested = True
            print("[PicoDecoupled] Environment reset requested (L-grip + R-grip)")
        self._reset_btn_last = reset_btn

    # ------------------------------------------------------------------
    # Rendering / streaming
    # ------------------------------------------------------------------

    def update_render_caches(self, observation: dict):
        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(observation)
        if self._streaming and self._streaming.is_running():
            self._push_stereo_frame(observation)
        return observation

    def _push_stereo_frame(self, observation: dict) -> None:
        left = observation.get("head_stereo_left")
        right = observation.get("head_stereo_right")
        if left is None or right is None:
            return
        left_bgr = np.ascontiguousarray(left[..., ::-1])
        right_bgr = np.ascontiguousarray(right[..., ::-1])
        # Draw debug text on top-right of each eye
        text = f"{self.episodes_saved}/{self.num_episodes}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 1.0, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = left_bgr.shape[1] - tw - 100
        y = th + 10
        cv2.putText(left_bgr, text, (x, y), font, scale, (0, 255, 0), thickness)
        cv2.putText(right_bgr, text, (x, y), font, scale, (0, 255, 0), thickness)
        stereo = np.concatenate([left_bgr, right_bgr], axis=1)
        self._frame_buffer.put(stereo)

    # ------------------------------------------------------------------
    # Core: build observation for decoupled WBC and compute action
    # ------------------------------------------------------------------

    def _build_wbc_observation(self, sim_obs: dict) -> dict:
        """Convert SIMPLE observation dict to decoupled_wbc observation format.

        The decoupled_wbc G1Env.observe() returns q/dq/ddq/tau_est as full
        configuration vectors (43 dofs: 29 body + 7 left hand + 7 right hand)
        in the robot_model's joint order.  We must replicate that layout using
        ``robot_model.get_configuration_from_actuated_joints()``.
        """
        rm = self._dwbc_robot_model
        obs = {}

        # Build full-configuration vectors (body 29 + hands 7+7 = 43 dofs)
        left_hand_q = sim_obs.get("left_hand_q", np.zeros(7))
        right_hand_q = sim_obs.get("right_hand_q", np.zeros(7))
        left_hand_dq = sim_obs.get("left_hand_dq", np.zeros(7))
        right_hand_dq = sim_obs.get("right_hand_dq", np.zeros(7))

        mjcf_to_natural_order = lambda q: np.concatenate([q[:3], q[5:7], q[3:5]])

        # NOTE: "rm" expects hand joint in NATURAL order: thumb/index/middle
        # and it returns joints in URDF order
        obs["q"] = rm.get_configuration_from_actuated_joints( 
            body_actuated_joint_values=sim_obs["body_q"],
            left_hand_actuated_joint_values=mjcf_to_natural_order(left_hand_q),
            right_hand_actuated_joint_values=mjcf_to_natural_order(right_hand_q),
        )
        obs["dq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_dq"],
            left_hand_actuated_joint_values=mjcf_to_natural_order(left_hand_dq),
            right_hand_actuated_joint_values=mjcf_to_natural_order(right_hand_dq),
        )
        obs["ddq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_ddq", np.zeros(29)),
            left_hand_actuated_joint_values=mjcf_to_natural_order(sim_obs.get("left_hand_ddq", np.zeros(7))),
            right_hand_actuated_joint_values=mjcf_to_natural_order(sim_obs.get("right_hand_ddq", np.zeros(7))),
        )
        obs["tau_est"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_tau_est", np.zeros(29)),
            left_hand_actuated_joint_values=mjcf_to_natural_order(sim_obs.get("left_hand_tau_est", np.zeros(7))),
            right_hand_actuated_joint_values=mjcf_to_natural_order(sim_obs.get("right_hand_tau_est", np.zeros(7))),
        )

        # Floating base
        obs["floating_base_pose"] = sim_obs["floating_base_pose"]
        obs["floating_base_vel"] = sim_obs["floating_base_vel"]
        obs["floating_base_acc"] = sim_obs.get("floating_base_acc", np.zeros(6))

        # Torso IMU
        obs["torso_quat"] = sim_obs.get("secondary_imu_quat", np.array([1, 0, 0, 0]))
        obs["torso_ang_vel"] = sim_obs.get("secondary_imu_vel", np.zeros(6))[3:6] if "secondary_imu_vel" in sim_obs else np.zeros(3)

        # Wrist pose placeholder (used for data export, not policy input)
        obs["wrist_pose"] = sim_obs.get("wrist_pose", np.zeros(14))

        return obs
    
    def get_stabilize_action(self, proprio) -> ActionCmd:
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

    def _run_decoupled_policy(self, sim_obs: dict) -> dict:
        """Run one step of the decoupled WBC pipeline.

        Returns dict with "q" key containing target joint positions.
        """
        t_now = time.monotonic()

        # # Suppress activation until the robot has stabilized
        # if not self.robot.stabilized and self._teleop_policy.is_active:
        #     self._teleop_policy.reset()
        #     self._wbc_policy.reset(init_time=t_now)
        #     print("[PicoDecoupled] Activation ignored — robot not yet stabilized")

        # 1. Get upper body command from teleop policy (reads Pico streamer internally)
        teleop_action = self._teleop_policy.get_action()

        # Cache the raw teleop action for data recording (wrist_pose, navigate_cmd, etc.)
        self._last_teleop_action = teleop_action
        syned_teleop_action_eef = teleop_action["wrist_pose"]

        # Add timing info expected by the WBC policy
        teleop_action["timestamp"] = t_now
        if not self._teleop_initialized:
            teleop_action["target_time"] = t_now + 2.0  # initial pose time
            self._teleop_initialized = True
        else:
            control_freq = self._control_frequency
            teleop_action["target_time"] = t_now + (1 / control_freq)

        # 2. Build observation for the WBC policy
        wbc_obs = self._build_wbc_observation(sim_obs)
        self._wbc_policy.set_observation(wbc_obs)

        # 3. Set goal (upper body target + locomotion commands)
        teleop_just_activated = self._teleop_policy.is_active and not self._teleop_was_active
        self._teleop_was_active = self._teleop_policy.is_active

        if teleop_just_activated:
            self._teleop_activate_time = t_now
            print("[PicoDecoupled] Teleop activated — smoothing arm engagement")

        wbc_goal = {}
        if teleop_action:
            wbc_goal = teleop_action.copy()
            control_freq = self._control_frequency
            wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * (1 / control_freq)

            # After activation, decay target_time from _arm_engage_smooth_secs → 1/freq.
            # This prevents the interpolation policy from chasing the first noisy IK
            # targets at maximum rate, which causes the characteristic arm shake.
            if self._teleop_activate_time is not None:
                elapsed = t_now - self._teleop_activate_time
                remaining = max(0.0, self._arm_engage_smooth_secs - elapsed)
                if remaining > 0.0:
                    wbc_goal["target_time"] = t_now + remaining

            # On the very first step, also seed the goal with the current pose so the
            # interpolation starts from zero positional error.
            if teleop_just_activated and "q" in wbc_goal:
                wbc_goal["q"] = wbc_obs["q"].copy()

        # Debug: trace toggle_policy_action through the pipeline
        if wbc_goal.get("toggle_policy_action"):
            print(f"[PicoDecoupled] toggle_policy_action={wbc_goal['toggle_policy_action']} → sending to WBC policy")

        if wbc_goal:
            # # Binary vx: deadzone → 0, outside deadzone → ±0.35 m/s constant
            # if "navigate_cmd" in wbc_goal:
            #     nav = np.asarray(wbc_goal["navigate_cmd"], dtype=float).copy()
            #     _VX_DEADZONE = 0.1
            #     _VX_SPEED = 0.35
            #     nav[0] = np.sign(nav[0]) * _VX_SPEED if abs(nav[0]) > _VX_DEADZONE else 0.0
            #     wbc_goal["navigate_cmd"] = nav
            self._wbc_policy.set_goal(wbc_goal)

        # 4. Compute whole-body action'
        # NOTE it return joints in URDF order
        wbc_action = self._wbc_policy.get_action(time=t_now)
        wbc_action.update({
            "action_eef": syned_teleop_action_eef
        })
        return wbc_action

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def get_action(self, observation, instruction=None, **kwargs):
        # Run full WBC pipeline every step (main loop now at 50Hz)
        # proprio = self.robot.prepare_obs()
        # for k,v in kwargs["privileged_info"]["proprio"].items(): assert np.all(v == proprio[k])
        proprio = kwargs["privileged_info"]["proprio"]
        if not self.robot.stabilized:
            return self.get_stabilize_action(proprio)
        
        wbc_action = self._run_decoupled_policy(proprio)

        # Read Pico buttons directly for drop_robot / reset_env
        self._poll_pico_buttons()

        # Cache the target joint positions for use in _build_frame() for recording
        # Body: 29 joints in URDF=MJCF=SIMPLE order
        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        # Hands: 7 joints each (driven by trigger/grip via teleop IK)
        # format: thumb/index/middle (SIMPLE order)
        self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")

        # Handle elastic band descent
        if self._dropping and self.robot.elastic_band and self.robot.elastic_band.enable:
            self.robot.elastic_band.length -= self._drop_rate * self._control_dt
            if self.robot.elastic_band.length <= -0.25 and abs(self.robot.pelvis_vz) < 0.05:
                self.robot.elastic_band.enable = False
                self._dropping = False
                print(f"[PicoDecoupled] Robot landed (pelvis Z={self.robot.pelvis_z:.3f} m)")

        # While elastic band is active, only apply band forces (ignore WBC output)
        if (
            self.robot.elastic_band
            and self.robot.elastic_band.enable
            and self.robot.use_floating_root_link
        ):
            return ActionCmd(
                "elastic_band",
                dropping=self._dropping,
                drop_rate=self._drop_rate,
            )

        return ActionCmd( # all synced!
            "decoupled_wbc",
            target_q=self._cached_target_q, # (29,)
            left_hand_q=self._cached_left_hand_q, # (7,)
            right_hand_q=self._cached_right_hand_q, # (7,)
            base_height_command=wbc_action["base_height_command"], # (1,)
            navigate_cmd=wbc_action["navigate_cmd"], # (4,)
            torso_rpy_cmd=wbc_action["torso_rpy_cmd"], # (3,)  can be derived from q using FK
            action_eef=wbc_action["action_eef"], # (14,) teleop IK synced
            obs_tensor=wbc_action["obs_tensor"] # (1,516=86*6)
        )

    def reset_policy(self):
        """Reset the WBC pipeline to initial state for a new episode.

        Resets the upper-body interpolation policy back to the default pose,
        deactivates the teleop policy (upper body tracking stops — the
        teleoperator must press the activation button to re-enable after
        aligning with the default pose), resets the lower-body RL policy's
        observation history, and clears cached joint targets.
        """
        import collections

        t_now = time.monotonic()

        # 1. Reset entire decoupled WBC pipeline (upper and lower body policies)
        self._wbc_policy.reset(init_time=t_now)

        # 2. Reset and deactivate the teleop policy.
        #    This clears retargeting IK, sets is_active=False so the upper
        #    body holds the default pose until the operator re-activates
        #    (via the Pico activation button), which also re-calibrates.
        self._teleop_policy.reset()

        # 3. Clear cached joint targets
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._teleop_initialized = False
        self._teleop_was_active = False
        self._teleop_activate_time = None
        self._last_teleop_action = {}

    def publish_low_state(self, proprio):
        # No Unitree bridge needed — decoupled WBC reads from SIMPLE directly
        pass

    def close(self):
        if hasattr(self, "_teleop_policy"):
            self._teleop_policy.close()
