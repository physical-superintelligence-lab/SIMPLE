"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Replay recorded teleop demonstrations and record a new dataset with newly rendered images.

Reads a LeRobot dataset recorded by render_decoupled_wbc.py, restores the scene configuration,
replays actions through the WBC pipeline, and records a new dataset with Isaac Sim/MuJoCo
rendered images combined with all proprioceptive/action data.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import os
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import json
import time
from datetime import datetime
import typer
import gymnasium as gym
from typing_extensions import Annotated, TYPE_CHECKING
import torch
import numpy as np
import matplotlib.pyplot as plt
import simple.envs as _  # import all envs
from tqdm import tqdm
import transforms3d as t3d
import mujoco
from scipy.spatial.transform import Rotation as R
import cv2
from PIL import Image
from pathlib import Path
import shutil
if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

from simple.cli.render_decoupled_wbc import _load_episodes, _load_episode_configs
from simple.agents.replay_decoupled_agent import ReplayDecoupledAgent
from simple.robots.g1_sonic import G1Sonic
from simple.utils import NumpyArrayEncoder


def _init_exporter(save_dir: str, task_prompt: str, robot_model, obj_names: list[str], joint_names: list[str]):
    """Create a Gr00tDataExporter for LeRobot-format recording."""
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

    features = get_dataset_features(robot_model)
    features["observation.state"]["names"] = joint_names # state joint names

    modality_config = get_modality_config(robot_model)

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

    exporter = Gr00tDataExporter.create(
        save_root=save_dir,
        fps=50,
        features=features,
        modality_config=modality_config,
        task=task_prompt,
    )
    return exporter


def _save_episode_env_config(exporter, env_conf: dict, episode_index: int):
    """Write the episode's environment_config into episodes.jsonl."""
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    lines[episode_index]["environment_config"] = json.dumps(env_conf, cls=NumpyArrayEncoder)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


def _validate_and_repair_dataset(save_dir: str, data_dir: str):
    """Validate consistency between episodes.jsonl and video files.

    Removes episodes from episodes.jsonl if their corresponding MP4 video is missing.
    Also removes corresponding results from replay.txt.

    Args:
        save_dir: Path to dataset with episodes.jsonl
        data_dir: Path to source data directory with replay.txt

    Returns:
        (num_valid_episodes, num_removed_episodes, episodes_to_skip_indices)
    """
    run_save_dir = Path(save_dir)
    episodes_jsonl = run_save_dir / "meta" / "episodes.jsonl"
    video_dir = run_save_dir / "videos" / "chunk-000" / "observation.images.ego_view"
    replay_txt = Path(data_dir) / "meta" / "replay.txt"

    if not episodes_jsonl.exists():
        return 0, 0, set()

    # Read current episodes
    with open(episodes_jsonl, "r") as f:
        episodes_list = [json.loads(line) for line in f]

    # Check each episode for missing video
    valid_episodes = []
    removed_indices = set()

    for ep_idx, episode_meta in enumerate(episodes_list):
        video_path = video_dir / f"episode_{ep_idx:06d}.mp4"

        if not video_path.exists():
            print(f"[Validate] Episode {ep_idx} in episodes.jsonl but video missing: {video_path}")
            removed_indices.add(ep_idx)
        else:
            valid_episodes.append(episode_meta)

    # Rewrite episodes.jsonl with only valid episodes
    if removed_indices:
        with open(episodes_jsonl, "w") as f:
            for entry in valid_episodes:
                f.write(json.dumps(entry) + "\n")
        print(f"[Validate] Removed {len(removed_indices)} corrupted episodes from episodes.jsonl")

        # Also remove from replay.txt
        if replay_txt.exists():
            with open(replay_txt, "r") as f:
                results_lines = f.readlines()

            valid_results = []
            for line in results_lines:
                parts = line.strip().split(": ")
                if len(parts) == 2:
                    ep_idx = int(parts[0])
                    if ep_idx not in removed_indices:
                        valid_results.append(line)

            with open(replay_txt, "w") as f:
                f.writelines(valid_results)
            print(f"[Validate] Cleaned up {len(removed_indices)} episodes from replay.txt")

    return len(valid_episodes), len(removed_indices), removed_indices


def _build_frame(observation, source_row, registered_features: set = None):
    """Assemble one recording frame by reusing source dataset fields with newly rendered image.

    Args:
        observation: Current observation with newly rendered head_stereo_left image
        source_row: Source dataset row with all other fields
        registered_features: Set of valid feature names (if None, copy all except metadata)

    Returns:
        Frame dict with source data + new image
    """
    # Metadata columns to skip
    metadata_cols = {'index', 'timestamp', 'episode_index', 'frame_index', 'task_index'}

    frame = {}
    for key in source_row.index:
        # Skip metadata columns
        if key in metadata_cols:
            continue

        # Skip extra features not in registered set
        if registered_features and key not in registered_features:
            continue

        # Copy the value, ensuring it's an array
        value = source_row[key]

        # Handle scalar values that should be arrays
        if key in {'teleop.base_height_command', 'observation.img_state_delta'}:
            frame[key] = np.atleast_1d(np.asarray(value))
        else:
            frame[key] = np.asarray(value)

    # Replace with newly rendered image
    frame["observation.images.ego_view"] = observation["head_stereo_left"]

    return frame


def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyBendPick-v1",
    data_dir: Annotated[str, typer.Option()] = "data/render_decoupled_wbc/simple/G1WholebodyBendPick-v1/level-0",
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    webrtc: Annotated[bool, typer.Option()] = True,
    render_hz: Annotated[int, typer.Option()] = 50,
    num_episodes: Annotated[int, typer.Option()] = -1,
    record: Annotated[bool, typer.Option()] = True,
    save_dir: Annotated[str, typer.Option()] = "data/replay_decoupled_wbc",
    success_criteria: Annotated[float, typer.Option()] = 0.5,
    resume: Annotated[bool, typer.Option()] = False
):
    """Physics-based replay of recorded teleop through decoupled WBC pipeline with dataset recording.

    Args:
        resume: If True, resume from last interrupted replay without re-processing already saved episodes.
    """
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    # # Load dataset info
    # info_path = Path(data_dir) / "meta" / "info.json"
    # with open(info_path) as f:
    #     dataset_info = json.load(f)
    # dataset_fps = dataset_info["fps"]

    # Load episodes from parquet
    print(f"Loading dataset from {data_dir} ...")
    episodes = _load_episodes(data_dir)
    total_episodes = len(episodes)
    if num_episodes < 0:
        num_episodes = total_episodes
    num_episodes = min(num_episodes, total_episodes)
    print(f"Loaded {total_episodes} episodes, will replay {num_episodes}")

    # Load per-episode environment configs
    episode_configs = _load_episode_configs(data_dir)
    if episode_configs:
        print(f"Loaded environment configs for {len(episode_configs)} episodes")
    else:
        print("WARNING: No environment_config found in episodes.jsonl")

    # Create environment
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt  # = 0.02 s (50 Hz)

    print(f"Creating environment: {env_id} (sim_mode={sim_mode})")
    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=render_hz,
        physics_dt=sim_dt,
        headless=headless,
        webrtc=webrtc,
        max_episode_steps=30000,
        sonic_config=sonic_config,
        success_criteria=success_criteria
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    task = sonic_env.task
    robot = task.robot
    assert isinstance(robot, G1Sonic)

    # Create agent once (WBC policy persists across episodes)
    agent = ReplayDecoupledAgent(robot, sonic_config)

    # Recording setup
    exporter = None
    episodes_saved = 0
    results = {}
    run_save_dir = None

    # Check for resume - load already saved episodes
    episodes_to_skip = set()
    if resume and record:
        run_save_dir = f"{os.path.abspath(save_dir)}/{sonic_env.spec.id}/level-0"
        run_save_path = Path(run_save_dir)
        episodes_jsonl = run_save_path / "meta" / "episodes.jsonl"

        if episodes_jsonl.exists():
            # Validate dataset consistency before loading
            print("[Resume] Validating dataset consistency...")
            num_valid, num_removed, _ = _validate_and_repair_dataset(run_save_dir, data_dir)

            with open(episodes_jsonl, "r") as f:
                saved_episodes = [json.loads(line) for line in f]
            episodes_saved = len(saved_episodes)
            print(f"[Resume] Found {episodes_saved} valid episodes, will skip and continue")
            if num_removed > 0:
                print(f"[Resume] Removed {num_removed} corrupted episodes during validation")

            # Load results from replay.txt if it exists
            replay_txt = Path(data_dir) / "meta" / "replay.txt"
            if replay_txt.exists():
                with open(replay_txt, "r") as f:
                    for line in f:
                        parts = line.strip().split(": ")
                        if len(parts) == 2:
                            ep_idx = int(parts[0])
                            success = parts[1].lower() == "true"
                            results[ep_idx] = success
                print(f"[Resume] Loaded {len(results)} episode results from replay.txt")
        else:
            print(f"[Resume] No existing dataset found at {run_save_dir}, starting fresh")

    try:
        # Calculate initial progress for resume mode
        initial_progress = len(results) if resume else 0
        pbar = tqdm(
            range(num_episodes),
            desc="Episodes",
            unit="ep",
            position=0,
            leave=True,
            initial=initial_progress
        )
        for ep_idx in pbar:
            if ep_idx not in episodes:
                print(f"Episode {ep_idx} not found, skipping")
                continue

            # Skip if already processed in resume mode
            if resume and ep_idx in results:
                pbar.update(1)
                continue

            ep_data = episodes[ep_idx]

            # Reset environment with the exact scene configuration
            env_conf = episode_configs[ep_idx]
            observation, info = env.reset(options={"state_dict": env_conf})

            # Save env_conf immediately while it's intact (before lighting state gets cleared)
            saved_env_conf = env_conf

            # # Skip elastic band, engage RL policy (same as teleop_decoupled._on_episode_reset)
            # if robot.elastic_band is not None:
            #     robot.elastic_band.enable = False

            # Initialize exporter on first episode
            if record and exporter is None:
                obj_names = list(sonic_env.mujoco.mj_objects.keys())
                if run_save_dir is None:
                    run_save_dir = f"{os.path.abspath(save_dir)}/{sonic_env.spec.id}/level-0"

                # Clean up corrupted dataset only in non-resume mode
                run_save_path = Path(run_save_dir)
                if not resume and run_save_path.exists():
                    print(f"[Record] Removing corrupted dataset at {run_save_dir}")
                    shutil.rmtree(run_save_path)

                exporter = _init_exporter(
                    run_save_dir,
                    task.instruction,
                    agent._dwbc_robot_model,
                    obj_names,
                    robot.joint_names
                )
                print(f"[Record] Exporter initialized, saving to {run_save_dir}")
                print(f"[Record] Recording {len(obj_names)} objects: {obj_names}")

                # Verify exporter state matches saved episodes in resume mode
                if resume and episodes_saved > 0:
                    episodes_jsonl = Path(run_save_dir) / "meta" / "episodes.jsonl"
                    if episodes_jsonl.exists():
                        with open(episodes_jsonl, "r") as f:
                            saved_count = sum(1 for _ in f)
                        if saved_count != episodes_saved:
                            print(f"[WARNING] Episode count mismatch: loaded {episodes_saved} but found {saved_count} in episodes.jsonl")
                            episodes_saved = saved_count
                    print(f"[Record] Resuming from episode {episodes_saved}")

            agent.load_episode(ep_data)
            # TODO engage RL policy immediately
            agent._wbc_policy.lower_body_policy.use_policy_action = True

            sim_cnt = 0
            # --- Wait for robot to stabilize (velocity-based) ---
            stab_pbar = tqdm(desc=f"  Stabilizing (Ep {ep_idx})", unit="step", position=1, leave=False)
            try:
                while not robot.stabilized:
                    step_start = time.monotonic()
                    action = agent.get_stabilize_action(observation)
                    observation, reward, terminated, truncated, info = env.step(action)

                    sonic_env.update_viewer()
                    sonic_env.update_reward()

                    elapsed = time.monotonic() - step_start
                    sleep_time = control_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    stab_pbar.update(1)
                    sim_cnt += 1
            finally:
                stab_pbar.close()

            # === DEBUGGING ===
            mjModel = env.unwrapped.mjModel
            mjData = env.unwrapped.mjData

            # steady root pose
            pelvis_body_id = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            base_pos_steady = mjData.xpos[pelvis_body_id].copy()
            base_quat_steady = t3d.quaternions.mat2quat(mjData.xmat[pelvis_body_id].reshape(3, 3))

            # loaded robot pose
            initial_frame = ep_data.iloc[0]
            base_pos_loaded = initial_frame["observation.base_pose"][:3]
            base_quat_loaded = initial_frame["observation.base_pose"][3:]
            base_mat_loaded = t3d.quaternions.quat2mat(base_quat_loaded)

            # print base pose error
            pos_error = np.linalg.norm(base_pos_steady - base_pos_loaded)
            quat_error = np.arccos(np.clip(np.abs(np.dot(base_quat_steady, base_quat_loaded)), 0, 1))
            print(f"\n{'='*80}")
            print(f"BASE POSE COMPARISON at Episode {ep_idx} Stabilization")
            # print(f"Steady Base Position: {base_pos_steady}, Loaded Base Position: {base_pos_loaded}")
            # print(f"Steady Base Quaternion (wxyz): {base_quat_steady}, Loaded Base Quaternion (wxyz): {base_quat_loaded}")
            print(f"Base Position Error: {pos_error:.6f} m")
            print(f"Base Quaternion Error (geodesic distance): {quat_error:.6f} rad ({np.degrees(quat_error):.2f}°)")
            print(f"{'='*80}")

            loaded_initial_qpos = initial_frame["observation.state"] # observation["joint_qpos"]
            # FK to get loaded head camera pose
            rm = agent._dwbc_robot_model
            loaded_initial_qpos = rm.get_configuration_from_actuated_joints(
                body_actuated_joint_values=loaded_initial_qpos[:29],
                left_hand_actuated_joint_values=loaded_initial_qpos[29:36],
                right_hand_actuated_joint_values=loaded_initial_qpos[36:43],
            )
            rm.cache_forward_kinematics(loaded_initial_qpos)
            # Get torso link pose (relative to robot base)
            torso_placement = rm.frame_placement("torso_link")
            torso_pos_local = torso_placement.translation  # relative to pelvis
            torso_mat_local = torso_placement.rotation
            # Transform torso from robot local frame to world frame using pelvis transform
            pelvis_rot = R.from_matrix(base_mat_loaded)
            torso_pos_fk = base_pos_loaded + pelvis_rot.apply(torso_pos_local)
            torso_mat_fk = base_mat_loaded @ torso_mat_local
            torso_quat_wxyz_fk = t3d.quaternions.mat2quat(torso_mat_fk)
            # Apply camera offset in torso frame
            DEFAULT_HEAD_CAM_POSITION = np.array([0.05366004+0.0039635, 0.01752999, 0.4738702 - 0.044], dtype=np.float32)
            DEFAULT_HEAD_CAM_ORIENTATION = np.array([0.91496, 0.0, 0.40355, 0.0], dtype=np.float32)  # wxyz
            q_isaac_mujoco_mat = np.array([
                [ 0,  0, -1],
                [-1,  0,  0],
                [ 0,  1,  0]
            ])
            q_isaac_mujoco_wxyz = t3d.quaternions.mat2quat(q_isaac_mujoco_mat)  # already wxyz format
            # FK camera pose: torso offset from FK model
            torso_rot_fk = R.from_matrix(torso_mat_fk)
            cam_offset_rotated = torso_rot_fk.apply(DEFAULT_HEAD_CAM_POSITION)
            fk_cam_pos = torso_pos_fk + cam_offset_rotated
            # FK camera orientation: multiply quaternions
            fk_cam_quat_wxyz = t3d.quaternions.qmult(torso_quat_wxyz_fk, DEFAULT_HEAD_CAM_ORIENTATION)
            fk_cam_quat_wxyz = t3d.quaternions.qmult(fk_cam_quat_wxyz, q_isaac_mujoco_wxyz)
            # print(f"   Loaded Position (FK): {fk_cam_pos}")
            # print(f"   Loaded Quaternion (FK): {fk_cam_quat_wxyz}")

            # Compare it with the steady head camera pose
            cam_id = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_CAMERA, "head_stereo_left")
            steady_cam_pos = mjData.cam_xpos[cam_id].copy()
            steady_cam_mat = mjData.cam_xmat[cam_id].reshape(3, 3)
            steady_cam_quat_wxyz = t3d.quaternions.mat2quat(steady_cam_mat)
            
            # Position errors
            print(f"\n{'='*80}")
            pos_error_fk = np.linalg.norm(steady_cam_pos - fk_cam_pos)
            print(f"Head cam Position Error: {pos_error_fk:.6f} m")

            # Quaternion errors (geodesic distance and relative rotation)
            quat_error_fk = np.arccos(np.clip(np.abs(np.dot(steady_cam_quat_wxyz, fk_cam_quat_wxyz)), 0, 1))
            print(f"Head cam Quaternion Error (geodesic): {quat_error_fk:.6f} rad ({np.degrees(quat_error_fk):.2f}°)")

            # Calculate relative rotation in Euler angles
            # Relative rotation: R_rel = R_oracle^T @ R_fk
            steady_cam_mat = t3d.quaternions.quat2mat(steady_cam_quat_wxyz)
            fk_cam_mat = t3d.quaternions.quat2mat(fk_cam_quat_wxyz)
            rel_rot_mat = steady_cam_mat.T @ fk_cam_mat

            # Convert to Euler angles (xyz order)
            rel_euler_rad = t3d.euler.mat2euler(rel_rot_mat, axes='sxyz')
            rel_euler_deg = np.degrees(rel_euler_rad)

            print(f"Relative rotation (Euler angles - sxyz):") 
            print(f"  Roll:  {rel_euler_deg[0]:7.2f}° (around x)")
            print(f"  Pitch: {rel_euler_deg[1]:7.2f}° (around y)")
            print(f"  Yaw:   {rel_euler_deg[2]:7.2f}° (around z)")
            print(f"{'='*80}\n")

            # Image.fromarray(observation["head_stereo_left"]).save(f"{ep_idx}_steady.png")

            # Load and save first frame from episode video
            video_dir = Path(data_dir) / "videos" / "chunk-000" / "observation.images.ego_view"
            video_path = video_dir / f"episode_{ep_idx:06d}.mp4"

            cap = cv2.VideoCapture(str(video_path))
            _, frame = cap.read()
            cap.release()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Image.fromarray(frame_rgb).save(f"{ep_idx}_loaded.png")

            agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1), dtype=torch.float32)
            agent._data_row_index = 0  # Start from beginning of episode data (now synchronized)
            sim_cnt = 0  # Reset sim counter for new episode
            eef_pos_errors = []  # Track EEF position errors for this episode
            max_lower_body_vels = []  # Track max lower body joint velocities for this episode

            # Sim step loop (50Hz) with frame progress bar
            episode_steps = len(ep_data)
            step_pbar = tqdm(
                total=episode_steps,
                desc=f"  Frames (Ep {ep_idx})",
                unit="frame",
                leave=False,
                position=2,
                bar_format="{desc} [{bar}] {n_fmt}/{total_fmt}"
            )

            try:
                while True:
                    step_start = time.monotonic()

                    try:
                        action = agent.get_action(observation)
                    except StopIteration:
                        break

                    # Capture state before step for recording
                    privileged_info = info.copy()
                    observation_before_step = observation.copy()
                    action_before_step = action.copy()

                    observation, reward, terminated, truncated, info = env.step(action)

                    # Track max lower body joint velocity
                    body_dq = info["proprio"]["body_dq"]
                    max_lower_body_vel = np.max(np.abs(body_dq))
                    max_lower_body_vels.append(max_lower_body_vel)

                    # Compare current right hand position with recorded dataset
                    rm = agent._dwbc_robot_model

                    # Current right hand position from forward kinematics
                    obs_q_full = rm.get_configuration_from_actuated_joints(
                        body_actuated_joint_values=observation["joint_qpos"][:29],
                        left_hand_actuated_joint_values=observation["joint_qpos"][29:36],
                        right_hand_actuated_joint_values=observation["joint_qpos"][36:43],
                    )
                    rm.cache_forward_kinematics(obs_q_full)
                    right_hand_frame_name = rm.supplemental_info.hand_frame_names["right"]
                    current_eef_pos = rm.frame_placement(right_hand_frame_name).translation

                    # Recorded right hand pose from dataset using observation.state (43D joint config)
                    if 0 <= agent._data_row_index < len(ep_data):
                        current_row = ep_data.iloc[agent._data_row_index-1] # -1 because data index is incremented after getting action
                        if "observation.state" in current_row:
                            recorded_state = np.array(current_row["observation.state"], dtype=np.float64)
                            # Convert 43D to full configuration
                            recorded_q_full = rm.get_configuration_from_actuated_joints(
                                body_actuated_joint_values=recorded_state[:29],
                                left_hand_actuated_joint_values=recorded_state[29:36],
                                right_hand_actuated_joint_values=recorded_state[36:43],
                            )
                            # Run forward kinematics on recorded configuration
                            rm.cache_forward_kinematics(recorded_q_full)
                            recorded_right_eef_pos = rm.frame_placement(right_hand_frame_name).translation
                            eef_pos_error = np.linalg.norm(current_eef_pos - recorded_right_eef_pos)
                            eef_pos_errors.append(eef_pos_error)

                    sonic_env.update_viewer()
                    sonic_env.update_reward()

                    # Record frame with newly rendered image
                    if exporter is not None and 0 <= agent._data_row_index < len(ep_data):
                        source_row = ep_data.iloc[agent._data_row_index]
                        registered_features = set(exporter.features.keys())
                        frame = _build_frame(observation_before_step, source_row, registered_features)
                        exporter.add_frame(frame)

                    if terminated or truncated:
                        break

                    # Update step progress bar at control rate (50Hz)
                    step_pbar.update(1)

                    # Real-time pacing
                    elapsed = time.monotonic() - step_start
                    sleep_time = control_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    sim_cnt += 1
            finally:
                step_pbar.close()

            # Save episode only if successful
            success = getattr(sonic_env, "_success", False)
            if exporter is not None and success:
                exporter.save_episode()
                _save_episode_env_config(exporter, saved_env_conf, episodes_saved)
                episodes_saved += 1
                print(f"[Record] Episode {episodes_saved} saved")
            elif exporter is not None and not success:
                exporter.skip_and_start_new_episode()
                # Clean up failed episode's MP4 file
                video_dir = Path(exporter.root) / "videos" / "chunk-000" / "observation.images.ego_view"
                mp4_path = video_dir / f"episode_{episodes_saved:06d}.mp4"
                if mp4_path.exists():
                    mp4_path.unlink()
                    print(f"[Record] Episode {ep_idx} failed - deleted partial MP4 at {mp4_path}")
                else:
                    print(f"[Record] Episode {ep_idx} failed - skipping recording")

            # Plot and save EEF position error and lower body velocity chart
            if eef_pos_errors and max_lower_body_vels:
                fig, axes = plt.subplots(2, 1, figsize=(12, 10))

                # Plot 1: EEF position error
                axes[0].plot(eef_pos_errors, linewidth=2, color='tab:blue')
                axes[0].set_xlabel("Step", fontsize=12)
                axes[0].set_ylabel("Right Hand EEF Position Error (m)", fontsize=12)
                axes[0].set_title(f"Episode {ep_idx} - Right Hand EEF Position Error", fontsize=14)
                axes[0].grid(True, alpha=0.3)

                # Plot 2: Max lower body joint velocity
                axes[1].plot(max_lower_body_vels, linewidth=2, color='tab:orange')
                axes[1].set_xlabel("Step", fontsize=12)
                axes[1].set_ylabel("Max Lower Body Joint Velocity (rad/s)", fontsize=12)
                axes[1].set_title(f"Episode {ep_idx} - Max Lower Body Joint Velocity", fontsize=14)
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()

                # Create directory for charts if it doesn't exist
                chart_dir = Path(save_dir) / "eef_error_charts"
                chart_dir.mkdir(parents=True, exist_ok=True)

                # Save chart
                chart_path = chart_dir / f"episode_{ep_idx}_metrics.png"
                plt.savefig(chart_path, dpi=150)
                plt.close()
                # print(f"Saved metrics chart to {chart_path}")

            # Record success and save immediately to prevent data loss
            results[ep_idx] = success

            # Save result immediately to replay.txt
            replay_txt = Path(data_dir) / "meta" / "replay.txt"
            replay_txt.parent.mkdir(parents=True, exist_ok=True)
            with open(replay_txt, "a") as f:
                f.write(f"{ep_idx}: {success}\n")

            completed = len(results)
            sr = sum(results.values()) / completed
            successes = sum(results.values())
            pbar.set_postfix(SR=f"{sr:.1%} ({successes}/{completed})")
            status = "✓" if success else "✗"
            tqdm.write(f"  ep {ep_idx:>3}  {status}  SR={sr:.1%} ({successes}/{completed})")

        # Report statistics
        if results:
            sr = sum(results.values()) / len(results)
            print(f"\n{'='*50}")
            print(f"Success rate: {sr:.1%} ({sum(results.values())}/{len(results)})")
            print(f"Per-episode: {results}")
            print(f"{'='*50}")

            # Rewrite replay.txt with sorted results to ensure consistency
            replay_txt = Path(data_dir) / "meta" / "replay.txt"
            with open(replay_txt, "w") as f:
                for idx, success in sorted(results.items()):
                    f.write(f"{idx}: {success}\n")
            print(f"Results finalized and saved to {replay_txt}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        import traceback
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        if exporter is not None:
            exporter.stop_video_writers()
            print(f"[Record] Done. {episodes_saved} episodes saved to {save_dir}")
        env.close()
        agent.close()


def typer_main():
    typer.run(main)

if __name__ == "__main__":
    typer.run(main)
