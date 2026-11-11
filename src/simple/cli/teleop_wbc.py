# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
import typer
from typing_extensions import Annotated, TYPE_CHECKING
import gymnasium as gym
import numpy as np
import simple.envs as _  # import all envs

if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

from simple.agents.pico_wbc_agent import PicoWbcAgent
from simple.cli._decoupled_wbc_recording import (
    create_gr00t_exporter,
    set_ego_view_feature_shape,
    validate_existing_ego_view_feature_shape,
    apply_wbc_token_action,
    WBC_TOKEN_DIM,
)
from simple.cli._sonic_collection_control import inherited_collection_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.robots.g1_sonic import G1Sonic

import json
import os
import shutil
from pathlib import Path
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import time
import tyro
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Episode recording state machine
# ---------------------------------------------------------------------------

class RecordingState(enum.Enum):
    WAITING_FOR_LANDING = "waiting_for_landing"
    RECORDING = "recording"
    EPISODE_DONE = "episode_done"
    # Episode saved; waiting for the OPERATOR to reset (both grips).  The sim
    # must not env.reset() on its own here: only the manager's grip-reset path
    # re-syncs the whole stack (forces PLANNER, re-arms auto_pose, walks the
    # controller through a safety reset that re-inits heading/streamed state).
    # A sim-side reset leaves the manager in POSE with stale state and teleop
    # is dead from episode 2 on.
    WAITING_FOR_RESET = "waiting_for_reset"


_RECORD_MIN_PELVIS_Z = 0.5  # robot counts as "standing" (ready to record) above this


def _reset_episode_step_budget(env, sonic_env) -> None:
    """Zero the gym TimeLimit clock so max_episode_steps bounds the RECORDED
    trajectory, not the ~20 s of pre-record setup (which would otherwise
    truncate the episode on its first frame)."""
    w = env
    seen: set[int] = set()
    while w is not None and id(w) not in seen:
        seen.add(id(w))
        if hasattr(w, "_elapsed_steps"):
            w._elapsed_steps = 0  # gym TimeLimit truncation counter
        w = getattr(w, "env", None)
    if hasattr(sonic_env, "step_count"):
        sonic_env.step_count = 0  # cosmetic; not used for truncation


def _reset_task_success_state(task) -> None:
    """Clear the task's success/lift accumulators at RECORDING start so state
    built up during the pre-record window can't end the episode on frame 1
    (attributes differ by task, so reset defensively whatever exists)."""
    if hasattr(task, "_init_target_height"):
        task._init_target_height = None
    if hasattr(task, "reward"):
        task.reward = 0
    if hasattr(task, "_contact_started"):
        task._contact_started = False
    if hasattr(task, "_success"):
        task._success = False


def _save_episode_env_config(exporter, task, episode_index: int):
    """Write the task state_dict as environment_config into episodes.jsonl."""
    from simple.utils import NumpyArrayEncoder
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    env_conf = task.state_dict()
    lines[episode_index]["environment_config"] = json.dumps(env_conf, cls=NumpyArrayEncoder)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


def _init_exporter(
    save_dir: str,
    task_prompt: str,
    robot_model,
    obj_names: list[str],
    joint_names: list[str],
    ego_view_shape=None,
):
    """Create a Gr00tDataExporter for LeRobot-format recording."""
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

    features = get_dataset_features(robot_model)
    set_ego_view_feature_shape(features, ego_view_shape)
    validate_existing_ego_view_feature_shape(save_dir, ego_view_shape)
    features["observation.state"]["names"] = joint_names # state joint names
    modality_config = get_modality_config(robot_model)

    # WBC action = SONIC 64-dim whole-body token + 14 hand joints (see helper).
    apply_wbc_token_action(features, modality_config, joint_names)

    # Add object poses feature: each object has 7D (pos xyz + quat wxyz)
    num_objects = len(obj_names)
    if num_objects > 0:
        obj_names_flat = []
        for name in obj_names:
            for suffix in ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]:
                obj_names_flat.append(f"{name}.{suffix}")
        features["observation.object_poses"] = {
            "dtype": "float64",
            "shape": (num_objects * 7,),
            "names": obj_names_flat,
        }

    # A run that dies between exporter init and the first saved episode leaves
    # a meta-only stub (info.json with 0 episodes, no episodes.jsonl) that the
    # exporter's resume path can't load — it falls back to a HF Hub fetch of a
    # placeholder repo and crashes.  Treat such a stub as no dataset at all.
    save_root = Path(save_dir)
    if save_root.exists() and not (save_root / "meta" / "episodes.jsonl").exists():
        print(f"[teleop_wbc] removing episode-less dataset stub at {save_root}")
        shutil.rmtree(save_root)

    exporter = create_gr00t_exporter(
        save_root=save_dir,
        fps=50,
        features=features,
        modality_config=modality_config,
        task=task_prompt,
    )
    return exporter


def _build_frame(agent, obj_names: list[str], observation, privileged_info, action):
    """Assemble one recording frame from sim state and the forwarded LowCmd."""
    proprio = privileged_info["proprio"]
    rm = agent.robot.sonic_robot_model

    # Hand cmds arrive in Dex3 motor order; the robot model wants natural order.
    mjcf_to_natural_order = lambda q: np.concatenate([q[:3], q[5:7], q[3:5]])

    # Prefer the tick-synced g1_debug snapshot (token + reconstructed LowCmd
    # from ONE controller state-logger entry — see WbcDebugSubscriber); fall
    # back to the separately-sampled DDS messages when the stream is absent.
    sync = action["wbc_debug"]
    if sync is not None:
        # last_action == motor_cmd[i].q (same formula both sides); hand
        # actions are in HandCmd motor order, same as the DDS path below.
        body_q = sync["q_target"]
        left_hand_q = mjcf_to_natural_order(sync["left_hand_action"])
        right_hand_q = mjcf_to_natural_order(sync["right_hand_action"])
    else:
        # Body targets: 29 values straight out of the DDS message.
        lc = action["low_cmd"]
        body_q = np.array([lc.motor_cmd[i].q for i in range(29)])
        left_hand_q = mjcf_to_natural_order(np.array([action["left_hand_cmd"].motor_cmd[i].q for i in range(7)]))
        right_hand_q = mjcf_to_natural_order(np.array([action["right_hand_cmd"].motor_cmd[i].q for i in range(7)]))

    # 43-DOF joint action — still needed below for action.eef (wrist FK).
    action_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=body_q,
        left_hand_actuated_joint_values=left_hand_q,
        right_hand_actuated_joint_values=right_hand_q,
    )

    # Recorded action = 64-dim whole-body token + the 14 hand targets.  With
    # the g1_debug snapshot, token + body_q + hand targets all come from the
    # same controller state-logger entry (tick-synced at the source, same
    # convention as real-robot recordings); the teleop loop then copies
    # `action` into data_frame before env.step, so token + hands + image +
    # state stay one synchronized frame.  left_hand_q / right_hand_q are the
    # natural-order hand targets already computed above.
    token = sync["token"] if sync is not None else action["token"]
    if token is None:
        # No token on the action yet: record zeros so the shape stays valid.
        if not getattr(agent, "_warned_no_token", False):
            print("[teleop_wbc][WARN] no WBC token on the action — recording zeros(64).")
            agent._warned_no_token = True
        token = np.zeros(WBC_TOKEN_DIM, dtype=np.float64)
    else:
        token = np.asarray(token, dtype=np.float64).ravel()
        assert token.shape == (WBC_TOKEN_DIM,), (
            f"WBC token shape {token.shape}, expected ({WBC_TOKEN_DIM},)")
    wbc_action = np.concatenate([token, left_hand_q, right_hand_q]).astype(np.float64)

    proprio_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=proprio["body_q"],
        left_hand_actuated_joint_values=mjcf_to_natural_order(proprio["left_hand_q"]),
        right_hand_actuated_joint_values=mjcf_to_natural_order(proprio["right_hand_q"]),
    )
    frame = {
        "observation.images.ego_view": observation["head_stereo_left"],
        "observation.state": np.asarray(observation["joint_qpos"], dtype=np.float64),
        "action": wbc_action,
        "observation.base_pose": np.asarray(proprio["floating_base_pose"], dtype=np.float64),
        "observation.base_vel": np.asarray(proprio["floating_base_vel"], dtype=np.float64),
    }

    # Object poses: unchanged from decoupled.
    if obj_names:
        obj_poses = []
        for name in obj_names:
            obj_poses.append(privileged_info[name])
        frame["observation.object_poses"] = np.concatenate(obj_poses).astype(np.float64)

    import pinocchio as pin
    def _wrist_pose_14(q):
        # Fixed-base model: FK is pelvis-relative, matching the dwbc dataset
        # convention.  auto_clip=False — the supplemental shoulder-roll limits
        # exclude 0, so clipping would silently bend recorded wrist poses.
        rm.cache_forward_kinematics(q, auto_clip=False)
        out = []
        for name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
            M = rm.frame_placement(name)
            quat = pin.Quaternion(M.rotation)
            # dataset convention is scalar-first wxyz; coeffs() would be xyzw
            out.append(np.concatenate([M.translation, [quat.w, quat.x, quat.y, quat.z]]))
        return np.concatenate(out)
    frame["observation.eef_state"] = _wrist_pose_14(proprio_q).astype(np.float64)
    frame["action.eef"] = _wrist_pose_14(action_q).astype(np.float64)
    frame["observation.img_state_delta"] = np.array([0.0], dtype=np.float32)

    # navigate/height from the pico_manager's "planner" topic
    # (mode, movement[3] world-frame, facing[3] heading vector, speed, height),
    # mapped onto decoupled's layout [lin_vel_x, lin_vel_y, vyaw, target_yaw].
    from decoupled_wbc.control.main.constants import DEFAULT_BASE_HEIGHT, DEFAULT_NAV_CMD
    planner = agent.latest_planner
    if planner is not None and agent.latest_planner_age < 0.5:
        h = float(np.asarray(planner["height"]).ravel()[0])
        # the manager sends -1.0 as "use default height"
        frame["teleop.base_height_command"] = np.asarray(
            [h if h > 0 else float(DEFAULT_BASE_HEIGHT)], dtype=np.float64)

        facing = np.asarray(planner["facing"], dtype=np.float64).ravel()
        target_yaw = float(np.arctan2(facing[1], facing[0]))
        # manager movement is world-frame: rotate back into the heading frame
        # -> [forward, left] in [0,1], scaled like decoupled's MAX_LINEAR_VEL
        mv = np.asarray(planner["movement"], dtype=np.float64).ravel()
        c, s = np.cos(target_yaw), np.sin(target_yaw)
        fwd = c * mv[0] + s * mv[1]
        left = -s * mv[0] + c * mv[1]
        # slot [2]: yaw-rate demand, from successive heading targets (the
        # decoupled recorder's vyaw_flag analogue), clipped like MAX_ANGULAR_VEL
        prev = getattr(agent, "_rec_prev_target_yaw", None)
        vyaw = 0.0
        if prev is not None:
            dyaw = np.arctan2(np.sin(target_yaw - prev), np.cos(target_yaw - prev))
            vyaw = float(np.clip(dyaw / 0.02, -1.0, 1.0))
        agent._rec_prev_target_yaw = target_yaw
        frame["teleop.navigate_command"] = np.asarray(
            [0.5 * fwd, 0.5 * left, vyaw, target_yaw], dtype=np.float64)
    else:
        # no planner data (or CONFLATE-stale, e.g. POSE mode): record defaults
        frame["teleop.base_height_command"] = np.atleast_1d(
            np.asarray(DEFAULT_BASE_HEIGHT, dtype=np.float64))
        frame["teleop.navigate_command"] = np.asarray(DEFAULT_NAV_CMD, dtype=np.float64)

    return frame

def _load_sonic_config() -> dict:
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyBendPickTeleop-v0",
    target: str | None = None,
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 30000,
    render_hz: Annotated[int, typer.Option()] = 50,
    save_dir: Annotated[str, typer.Option()] = "data/teleop_wbc",
    num_episodes: Annotated[int, typer.Option()] = 100,
    dr_level: Annotated[int, typer.Option()] = 0,
    record: Annotated[bool, typer.Option()] = False,
    success_criteria: Annotated[float, typer.Option()] = 3,
    drop_settle: Annotated[float, typer.Option(help="Seconds the controller must hold a quiet, stiff pose on the crane before auto-drop releases it. The controller needs time after engagement before it is really stabilized.")] = 3.0,
    auto_drop: Annotated[bool, typer.Option(help="Release the crane automatically once the manager is streaming (policy engaged) AND the controller holds with real gains for half a second. On by default because the manager's right-stick click depends on an SDK axis-click method that some builds do not expose, leaving no way to drop at all. --no-auto-drop for the manual click.")] = True,
):
    assert sim_mode in ["mujoco"], f"Invalid sim_mode {sim_mode} for teleop."
    collection_channel = inherited_collection_channel()
    if collection_channel is not None and (not record or num_episodes != 1):
        raise SystemExit(
            "automated SONIC collection requires --record and --num-episodes 1"
        )
    if record:
        try:
            from decoupled_wbc.data.exporter import Gr00tDataExporter  # noqa: F401
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "--record needs the lerobot extra:\n"
                "  uv sync --inexact --group sonic --extra lerobot"
            ) from exc
    sim_cnt = 0

    sonic_config = _load_sonic_config()

    print(f"Creating environment: {env_id}")
    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=render_hz,
        physics_dt=sonic_config["SIMULATE_DT"],
        headless=headless,
        max_episode_steps=max_episode_steps,
        sonic_config=sonic_config,
        target=target,
        dr_level=dr_level,
        success_criteria=success_criteria
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    task = sonic_env.task
    robot = task.robot
    assert sonic_env.spec is not None
    assert isinstance(robot, G1Sonic)

    agent = PicoWbcAgent(robot, auto_drop=auto_drop, drop_settle_s=drop_settle)
    agent.num_episodes = num_episodes

    # --- Recording setup ---
    exporter = None
    rec_state = RecordingState.WAITING_FOR_LANDING
    episodes_saved = 0
    automated_frame_limit: int | None = None
    automated_recorded_frames = 0
    control_decimal =  int(1/ sonic_config["SIMULATE_DT"]/render_hz)
    control_dt = control_decimal * robot.sim_dt  # = 0.02 s (50 Hz)

    def _on_episode_reset():
        """Reset per-episode agent state; the stabilization WBC holds the robot.

        The agent's in-process WBC (get_stabilize_action) balances the robot
        from spawn until robot.stabilized latches and the SONIC controller
        takes over — the crane is disabled, matching teleop_decoupled_wbc.
        """
        agent._dropping = False
        agent._hold_q = None
        agent._catch_ticks = None
        band = robot.elastic_band
        if band is not None:
            # The in-process stabilization WBC (get_stabilize_action) balances
            # the robot from spawn, so the crane is not needed — same as
            # teleop_decoupled_wbc.  Leaving it enabled would only mislabel the
            # status line and block the recording gate (elastic_done).
            band.enable = False
        # Reset the WBC pipeline so the robot starts from the default pose; the
        # teleop policy stays off until the operator presses the activation button.
        agent.reset_policy()

    step_pbar = None  # Progress bar for current recording episode

    from decoupled_wbc.control.utils.telemetry import Telemetry
    telemetry = Telemetry(window_size=100)

    observation, privileged_info = env.reset()
    _on_episode_reset()

    # Read obj_names after first reset so layout is populated by domain randomization.
    obj_names = list(sonic_env.mujoco.mj_objects.keys())

    if record:
        run_save_dir = (
            f"{os.path.abspath(save_dir)}/{sonic_env.spec.id}/level-{dr_level}"
        )
        # Startup FK probe: fail fast (with pinocchio's list of valid frames)
        # if the wrist frame names ever drift from the URDF.
        _rm = robot.sonic_robot_model
        _rm.cache_forward_kinematics(np.zeros(43), auto_clip=False)
        _rm.frame_placement("left_wrist_yaw_link")
        _rm.frame_placement("right_wrist_yaw_link")

        exporter = _init_exporter(
            run_save_dir,
            task.instruction,
            robot.sonic_robot_model,
            obj_names,
            robot.joint_names,
            observation["head_stereo_left"].shape,
        )
        print(f"\n[Record] Exporter initialized, saving to {run_save_dir}")
        print(f"[Record] Ego view shape: {observation['head_stereo_left'].shape}")
        print(f"[Record] Recording {len(obj_names)} objects: {obj_names}")

    try:
        # Sample robot state at the PHYSICS rate, like gear_sonic's own sim
        # loop does, and hand it to the heartbeat thread (which owns the DDS
        # send).  Reading mjData must happen here on the sim thread; the actual
        # publish is off-thread so the physics loop never blocks on the network.
        def _cache_at_physics_rate():
            agent.cache_proprio(robot.prepare_obs())

        sonic_env.substep_callback = _cache_at_physics_rate
        # …and keep state flowing across env.reset(), which otherwise goes
        # silent long enough (scene recompile + renderer rebuild) to trip the
        # controller's 500 ms watchdog and make it exit.
        agent.start_state_heartbeat()

        _status_last = 0.0
        while True:
            step_start = time.monotonic()

            data_frame = {}

            with telemetry.timer("agent.get_action"):
                action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            
            data_frame["observation"] = observation.copy()
            data_frame["action"] = action.copy()
            data_frame["privileged_info"] = privileged_info.copy()

            with telemetry.timer("env.step"):
                observation, reward, terminated, truncated, privileged_info = env.step(action)

            # (state is published from the physics-rate substep hook set up
            # before the loop, not here — see _cache_at_physics_rate)

            # One-line status once a second: which of the startup steps is
            # still missing is otherwise invisible, and every failure so far
            # looked identical from the operator's seat.
            now = time.monotonic()
            if now - _status_last >= 1.0:
                _status_last = now
                band = agent.robot.elastic_band
                on_crane = bool(band and band.enable)
                if not agent.unitree_bridge.cmd_received():
                    ctrl_state = "controller DOWN"
                elif not agent.engaged:
                    ctrl_state = "policy NOT ENGAGED (press A+X)"
                else:
                    ctrl_state = f"CONTROL mode={agent.manager_mode}"
                if not record:
                    rec_txt = "rec off"
                elif rec_state == RecordingState.RECORDING:
                    rec_txt = f"RECORDING ep{episodes_saved + 1}"
                elif rec_state == RecordingState.WAITING_FOR_RESET:
                    rec_txt = f"ep saved ({episodes_saved}) — SQUEEZE BOTH GRIPS to reset"
                else:
                    rec_txt = f"rec waiting ({episodes_saved} saved)"
                # Success outside RECORDING is deliberately ignored (no frames
                # to save / episode already saved) — but say so, or a completed
                # task just spams [Task] success prints with no visible effect.
                if (terminated or truncated) and rec_state != RecordingState.RECORDING:
                    rec_txt += " | task success IGNORED (not recording)"
                if on_crane and ctrl_state.startswith("CONTROL"):
                    next_step = f"-> {agent.settle_progress}, then auto-drop"
                elif on_crane:
                    next_step = "-> engage first (A+X)"
                elif "POSE" in ctrl_state:
                    next_step = "-> your body drives the robot"
                elif "PLANNER" in ctrl_state:
                    next_step = ("-> A+B to leave IDLE, then sticks"
                                 if "IDLE" in ctrl_state else "-> sticks to walk; A+X for POSE")
                else:
                    next_step = "-> teleop"
                print(f"[status] {'CRANE' if on_crane else 'free'} | {ctrl_state} | "
                      f"pelvis_z={sonic_env.mjData.qpos[2]:.2f} | {rec_txt} | "
                      f"ind={agent.rec_indicator or '-'} {next_step}")

            if agent.reset_requested:
                # Discard any in-progress recording
                if exporter is not None and rec_state == RecordingState.RECORDING:
                    print("[Record] Reset requested, discarding in-progress episode")
                    exporter.skip_and_start_new_episode()
                    # Close progress bar for discarded episode
                    if step_pbar is not None:
                        step_pbar.close()
                        step_pbar = None

                observation, privileged_info = env.reset()
                _on_episode_reset()
                obj_names = list(sonic_env.mujoco.mj_objects.keys())
                sim_cnt = 0
                rec_state = RecordingState.WAITING_FOR_LANDING
                print("[TeleopWbc] Environment reset complete")

            with telemetry.timer("update_viewer"):
                if not headless:
                    sonic_env.update_viewer()

            with telemetry.timer("update_reward"):
                sonic_env.update_reward()

            with telemetry.timer("update_render"):
                # Headset overlay: red REC while frames are saved; yellow ARMED
                # from the A+X POSE entry until the record gate opens; gray IDLE
                # after the episode is saved, until the operator resets the scene.
                if exporter is not None and rec_state is RecordingState.RECORDING:
                    indicator = "rec"
                elif (exporter is not None
                        and rec_state is RecordingState.WAITING_FOR_LANDING
                        and "POSE" in agent.manager_mode):
                    indicator = "armed"
                elif exporter is not None and rec_state is RecordingState.WAITING_FOR_RESET:
                    indicator = "idle"
                else:
                    indicator = ""
                if indicator != agent.rec_indicator:
                    # Terminal trace of every overlay transition: if the headset
                    # shows nothing, this line says which input vetoed it.
                    print(f"[TeleopWbc] headset indicator -> {indicator or 'off'} "
                          f"(record={exporter is not None}, rec_state={rec_state.value}, "
                          f"manager_mode={agent.manager_mode})")
                agent.rec_indicator = indicator
                agent.update_render_caches(observation)

            with telemetry.timer("recode_data"):
                # --- Recording logic (runs at 50Hz) ---
                if exporter is not None:
                    started_recording_this_tick = False
                    if rec_state == RecordingState.WAITING_FOR_LANDING:
                        # Check if robot has landed (elastic band done) AND
                        # teleop policy is active (operator has re-aligned and
                        # pressed the activation button)
                        elastic_done = (
                            not agent._dropping
                            and (robot.elastic_band is None or not robot.elastic_band.enable)
                        )
                        teleop_active = agent.is_teleop_active # new logic to check if teleop is active
                        # Ready = stabilization handed over (robot.stabilized
                        # latches during the stabilize phase, before SONIC
                        # forwarding starts) + SONIC holding with real gains +
                        # operator engaged + standing.  Without the stabilized/
                        # holding checks the gate opens while the agent is still
                        # returning stabilize actions, which carry no low_cmd
                        # and crash _build_frame.
                        standing = sonic_env.mjData.qpos[2] > _RECORD_MIN_PELVIS_Z
                        # POSE gate: recording only runs in full-body POSE mode.
                        # After a grip-reset the manager parks in PLANNER (no
                        # auto_pose re-arm) until the operator presses A+X, so
                        # this is the operator's explicit "start episode" signal
                        # for episode 2+ — inferred from the manager's stream
                        # topic (pose vs planner).
                        in_pose = "POSE" in agent.manager_mode
                        if (elastic_done and teleop_active and standing and agent.engaged
                                and robot.stabilized and agent.controller_holding
                                and in_pose):
                            if collection_channel is not None:
                                collection_channel.send(
                                    "ready",
                                    pelvis_z=float(sonic_env.mjData.qpos[2]),
                                    manager_mode=agent.manager_mode,
                                    controller_holding=bool(agent.controller_holding),
                                    camera_shape=list(
                                        observation["head_stereo_left"].shape
                                    ),
                                )
                                start_message = collection_channel.expect(
                                    "start", timeout=60.0
                                )
                                automated_frame_limit = int(
                                    start_message.get("frame_count", 0)
                                )
                                requested_fps = float(
                                    start_message.get("fps", render_hz)
                                )
                                if automated_frame_limit <= 0:
                                    raise RuntimeError(
                                        "automated collection frame_count must be positive"
                                    )
                                if requested_fps != render_hz:
                                    raise RuntimeError(
                                        "automated collection fps does not match render_hz: "
                                        f"{requested_fps} != {render_hz}"
                                    )
                            rec_state = RecordingState.RECORDING
                            started_recording_this_tick = True
                            automated_recorded_frames = 0
                            sim_cnt = 0  # Reset sim counter for new episode
                            # Reset the step budget + task success so the ~20s of
                            # setup doesn't truncate/terminate the recording.
                            _reset_episode_step_budget(env, sonic_env)
                            _reset_task_success_state(sonic_env.task)
                            # terminated/truncated came from THIS iteration's step()
                            # (before the resets above); clear so the transition frame
                            # doesn't end the episode.
                            terminated = False
                            truncated = False
                            # Create progress bar for this recording episode
                            step_pbar = tqdm(desc=f"Recording episode {episodes_saved + 1}", unit="frame",
                                            leave=False, position=1, bar_format="{desc} {n_fmt} {rate_fmt}")
                            print("[Record] Teleop active, starting episode recording")
                            
                    

                    if rec_state == RecordingState.RECORDING:
                        if started_recording_this_tick:
                            # The current data_frame was sampled before the
                            # supervisor released the first source frame.
                            pass
                        elif data_frame["action"]["low_cmd"] is None:
                            # Stabilize/crane actions carry no SONIC LowCmd —
                            # skip the frame instead of crashing; recording
                            # resumes when forwarding resumes.
                            pass
                        else:
                            frame = _build_frame(agent, obj_names, **data_frame)
                            exporter.add_frame(frame)
                            automated_recorded_frames += 1
                            if step_pbar is not None:
                                step_pbar.update(1)

                        if (
                            automated_frame_limit is not None
                            and automated_recorded_frames >= automated_frame_limit
                        ):
                            rec_state = RecordingState.EPISODE_DONE
                        elif automated_frame_limit is None and (terminated or truncated):
                            rec_state = RecordingState.EPISODE_DONE

                    if rec_state == RecordingState.EPISODE_DONE:
                        # Close progress bar for this episode
                        if step_pbar is not None:
                            step_pbar.close()
                            step_pbar = None

                        ep_idx = exporter.episode_buffer["episode_index"]
                        exporter.save_episode()
                        _save_episode_env_config(exporter, task, ep_idx)
                        episodes_saved += 1
                        agent.episodes_saved = episodes_saved
                        print(f"[Record] Episode {episodes_saved} saved")
                        if collection_channel is not None:
                            collection_channel.send(
                                "saved",
                                frames=automated_recorded_frames,
                                episode_index=int(ep_idx),
                                dataset_dir=str(run_save_dir),
                            )
                        if episodes_saved >= num_episodes:
                            print(f"[Record] Reached {num_episodes} episodes, stopping")
                            break
                        # Hand the reset to the operator instead of env.reset()ing
                        # here: only the manager's grip-reset chord re-syncs the
                        # manager (PLANNER + auto_pose re-arm) and the controller
                        # (safety reset, heading re-init) along with the sim.  The
                        # reset then arrives as agent.reset_requested above.
                        rec_state = RecordingState.WAITING_FOR_RESET
                        print("[Record] Squeeze BOTH GRIPS to reset the scene for "
                              "the next episode (manager re-engages POSE automatically)")
                        continue  # skip sleep / increment for this iteration

            elapsed = time.monotonic() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Only show timing when the control loop overran
                telemetry.log_timing_info(context=f"{sim_cnt: >3}: Control loop overran by {-sleep_time:.3f} seconds", threshold=0.001)

            sim_cnt += 1
    except KeyboardInterrupt:
        print("Simulator interrupted by user.")
        if step_pbar is not None:
            step_pbar.close()
            step_pbar = None
    finally:
        if step_pbar is not None:
            step_pbar.close()
        if exporter is not None:
            # Save-on-quit: a free teleop demo never auto-saves (no task success),
            # so bank the in-progress episode on any exit.  The guard skips a
            # double-save after a clean EPISODE_DONE (buffer empty, not RECORDING).
            buf = getattr(exporter, "episode_buffer", None)
            n_frames = buf.get("size", 0) if buf is not None else 0
            if n_frames > 0 and rec_state == RecordingState.RECORDING:
                print(f"[Record] Saving in-progress episode before exit ({n_frames} frames)...")
                try:
                    ep_idx = exporter.episode_buffer["episode_index"]
                    exporter.save_episode()
                    _save_episode_env_config(exporter, task, ep_idx)
                    episodes_saved += 1
                    agent.episodes_saved = episodes_saved
                    print(f"[Record] Episode {episodes_saved} saved (on quit)")
                except Exception as e:
                    print(f"[Record] Failed to save episode on quit: {e}")
            # Delete the fresh 0-frame writers left by the last save (no stray mp4).
            for w in getattr(exporter, "video_writers", {}).values():
                try:
                    w.cancel()
                except Exception:
                    pass
            print(f"[Record] Done. {episodes_saved} episodes saved to {run_save_dir}")
        env.close()
        agent.close()
        if collection_channel is not None:
            collection_channel.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
