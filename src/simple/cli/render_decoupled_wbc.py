"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Replay recorded teleop demonstrations in MuJoCo + Isaac Sim for rendering.

Reads a LeRobot dataset recorded by teleop_decoupled.py, restores the exact
scene configuration (object identities, positions, lighting, etc.) from the
saved environment_config per episode, then sets MuJoCo physics state from the
recorded data each frame and runs Isaac Sim rendering.

No physics stepping is performed — the MuJoCo state is directly overwritten
from the dataset.

When --record is enabled, a new LeRobot dataset is written with Isaac Sim
rendered images combined with all original proprioceptive/action data.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import os
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import json
import time
import typer
import mujoco
import numpy as np
import gymnasium as gym
from pathlib import Path
from typing_extensions import Annotated, TYPE_CHECKING
from tqdm import tqdm
import tyro
import simple.envs as _  # import all envs

if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

def _load_episodes(data_dir: str):
    """Load all episodes from a LeRobot dataset directory."""
    import pyarrow.parquet as pq

    data_path = Path(data_dir) / "data"
    parquet_files = sorted(data_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")

    episodes = {}
    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        for ep_idx in df["episode_index"].unique():
            episodes[int(ep_idx)] = df[df["episode_index"] == ep_idx].reset_index(drop=True)

    return episodes


def _load_episode_configs(data_dir: str):
    """Load environment_config per episode from meta/episodes.jsonl."""
    meta_file = Path(data_dir) / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return {}

    configs = {}
    with open(meta_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            ep_idx = entry.get("episode_index", None)
            env_conf_str = entry.get("environment_config", None)
            if ep_idx is not None and env_conf_str is not None:
                configs[int(ep_idx)] = json.loads(env_conf_str)
    return configs


def _init_replay_exporter(save_dir: str, fps: int, task_prompt: str, obj_names: list[str], joint_names: list[str]):
    """Create a Gr00tDataExporter for recording replayed Isaac Sim data."""
    from decoupled_wbc.control.robot_model.instantiation.g1 import (
        instantiate_g1_robot_model,
    )
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

    robot_model = instantiate_g1_robot_model()
    features = get_dataset_features(robot_model)
    features["observation.state"]["names"] = joint_names # state joint names
    modality_config = get_modality_config(robot_model)

    # # Add torso RPY command feature (3D: roll, pitch, yaw)
    # features["observation.torso_rpy_command"] = {
    #     "dtype": "float64",
    #     "shape": (3,),
    #     "names": ["roll", "pitch", "yaw"],
    # }

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
        fps=fps,
        features=features,
        modality_config=modality_config,
        task=task_prompt,
    )
    return exporter


def _build_replay_frame(row, isaac_image, source_features: dict):
    """Build a recording frame from a source parquet row and an Isaac-rendered image."""
    frame = {
        "observation.images.ego_view": isaac_image,
        "observation.state": np.asarray(row["observation.state"], dtype=np.float64),
        "observation.eef_state": np.asarray(row["observation.eef_state"], dtype=np.float64),
        "action": np.asarray(row["action"], dtype=np.float64),
        "action.eef": np.asarray(row["action.eef"], dtype=np.float64),
        "observation.img_state_delta": np.atleast_1d(
            np.asarray(row["observation.img_state_delta"], dtype=np.float32)
        ),
        "teleop.navigate_command": np.asarray(
            row["teleop.navigate_command"], dtype=np.float64
        ),
        "teleop.base_height_command": np.atleast_1d(
            np.asarray(row["teleop.base_height_command"], dtype=np.float64)
        ),
    }

    # Conditionally include features that may not be in all datasets
    if "observation.base_pose" in source_features:
        frame["observation.base_pose"] = np.asarray(
            row["observation.base_pose"], dtype=np.float64
        )
    if "observation.base_vel" in source_features:
        frame["observation.base_vel"] = np.asarray(
            row["observation.base_vel"], dtype=np.float64
        )
    # if "observation.torso_rpy_command" in source_features:
    #     frame["observation.torso_rpy_command"] = np.asarray(
    #         row["observation.torso_rpy_command"], dtype=np.float64
    #     )
    if "observation.object_poses" in source_features:
        frame["observation.object_poses"] = np.asarray(
            row["observation.object_poses"], dtype=np.float64
        )

    return frame


def _save_replay_episode_env_config(exporter, env_conf: dict, episode_index: int):
    """Write the episode's environment_config into episodes.jsonl."""
    from simple.utils import NumpyArrayEncoder
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    lines[episode_index]["environment_config"] = json.dumps(env_conf, cls=NumpyArrayEncoder)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")

def _load_episode_tasks(data_dir: str):
    """Load task name per episode from meta/tasks.jsonl."""                                                                                                                        
    tasks = {}                                                                                                 
    with open(Path(data_dir) / "meta" / "tasks.jsonl", "r") as f:
        for line in f:             
            entry = json.loads(line)                                                                                                                                      
            tasks[int(entry["task_index"])] = entry["task"]
    return tasks

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyBendPick-v1",
    data_dir: Annotated[str, typer.Option()] = "data/teleop_decoupled_wbc/simple/G1WholebodyBendPick-v1/level-0",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    webrtc: Annotated[bool, typer.Option()] = True,
    render_hz: Annotated[int, typer.Option()] = 30,
    num_episodes: Annotated[int, typer.Option()] = -1,
    record: Annotated[bool, typer.Option()] = False,
    save_dir: Annotated[str, typer.Option()] = "data/render_decoupled_wbc",
    dr_level: Annotated[int, typer.Option()] = 0,
):
    """Replay recorded teleop dataset with Isaac Sim rendering."""
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    # Default save_dir: append _isaac to data_dir
    if record and not save_dir:
        save_dir = f"{data_dir}_isaac"

    # Load dataset info
    info_path = Path(data_dir) / "meta" / "info.json"
    with open(info_path) as f:
        dataset_info = json.load(f)
    dataset_fps = dataset_info["fps"]

    # Load episodes from parquet
    print(f"Loading dataset from {data_dir} ...")
    episodes = _load_episodes(data_dir)
    total_episodes = len(episodes)
    if num_episodes < 0:
        num_episodes = total_episodes
    num_episodes = min(num_episodes, total_episodes)
    print(f"Loaded {total_episodes} episodes, will replay {num_episodes}")

    # Load per-episode environment configs (object identities, poses, scene, etc.)
    episode_configs = _load_episode_configs(data_dir)
    if episode_configs:
        print(f"Loaded environment configs for {len(episode_configs)} episodes")
    else:
        print("WARNING: No environment_config found in episodes.jsonl — "
              "scene will NOT match recorded episodes (objects may differ)")

    # Determine features available
    features = dataset_info["features"]
    has_base_pose = "observation.base_pose" in features
    has_base_vel = "observation.base_vel" in features
    has_object_poses = "observation.object_poses" in features
    print(f"Features: base_pose={has_base_pose}, base_vel={has_base_vel}, object_poses={has_object_poses}")

    # Create environment with Isaac Sim rendering
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"

    print(f"Creating environment: {env_id} (sim_mode={sim_mode})")
    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=render_hz,
        physics_dt=sonic_config["SIMULATE_DT"],
        headless=headless,
        webrtc=webrtc,
        max_episode_steps=1000,
        sonic_config=sonic_config,
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    mujoco_sim = sonic_env.mujoco
    isaac_sim = sonic_env.isaac
    robot = sonic_env.task.robot

    frame_dt = 1.0 / dataset_fps

    # --- Recording setup ---
    exporter = None
    episodes_saved = 0

    # # read replay results 
    # replay_results = {}
    # with open(f"{data_dir}/meta/replay.txt", "r") as f:
    #     replay_results = f.read().strip().splitlines()
    #     for line in replay_results:
    #         eps_idx, result = line.split(": ")
    #         replay_results[int(eps_idx)] = eval(result)

    # print(f"Loaded replay results for {len(replay_results)} episodes")

    try:
        for ep_idx in tqdm(range(num_episodes), desc="Episodes", unit="episode"):
            # if ep_idx not in episodes or not replay_results[ep_idx]:
            #     print(f"Episode {ep_idx} not found or failed, skipping")
            #     continue

            ep_data = episodes[ep_idx]
            num_frames = len(ep_data)

            # --- Reset environment with the exact scene configuration ---
            env_conf = episode_configs.get(ep_idx, None)
            if env_conf is not None:
                obs, info = env.reset(options={"state_dict": env_conf})
            else:
                obs, info = env.reset()

            obj_names_labels = list(sonic_env.mujoco.mj_objects.keys())
            obj_names = list(sonic_env.task.layout.actors[i].asset.name.replace(" ","_") for i in obj_names_labels)
            num_objects = len(obj_names_labels)

            # Init exporter after first reset so obj_names are available
            if record and exporter is None:
                task_prompt = _load_episode_tasks(data_dir)[0]
                exporter = _init_replay_exporter(
                    f"{os.path.abspath(save_dir)}/{sonic_env.spec.id}/level-{dr_level}", 
                    dataset_fps, task_prompt, obj_names_labels,
                    robot.joint_names
                )
                print(f"[Record] Exporter initialized, saving to {save_dir}")
                print(f"[Record] Recording {len(obj_names_labels)} objects: {obj_names_labels}")

            # Dataset joint names for mapping observation.state → MuJoCo joints
            # dataset_joint_names = features["observation.state"]["names"]
            dataset_joint_names = dataset_info["features"]["observation.state"]["names"]

            for frame_idx in tqdm(range(num_frames), desc=f"Frames (Ep {ep_idx})", leave=False, unit="frame"):
                if not record:
                    step_start = time.monotonic()
                row = ep_data.iloc[frame_idx]

                # --- Set robot base pose ---
                if has_base_pose:
                    base_pose = np.array(row["observation.base_pose"])
                    mujoco_sim.mjData.qpos[:7] = base_pose

                # --- Set robot base velocity ---
                if has_base_vel:
                    base_vel = np.array(row["observation.base_vel"])
                    mujoco_sim.mjData.qvel[:6] = base_vel

                # --- Set robot joint positions from observation.state ---
                obs_state = np.array(row["observation.state"])
                for jname, jval in zip(dataset_joint_names, obs_state):
                    if jname in mujoco_sim.joints:
                        mujoco_sim.joints[jname].qpos = jval

                # --- Set object poses ---
                # Note: If env_conf was used for reset, object poses are already correct.
                # Only set object poses from dataset if env_conf was NOT used.
                if has_object_poses and num_objects > 0:
                    obj_poses_flat = np.array(row["observation.object_poses"])
                    obj_positions = []
                    obj_orientations = []
                    for i in range(num_objects):
                        pose_7d = obj_poses_flat[i * 7 : (i + 1) * 7]
                        obj_positions.append(pose_7d[:3])
                        obj_orientations.append(pose_7d[3:])
                    mujoco_sim.set_object_poses(obj_names, obj_positions, obj_orientations)

                # --- Recompute MuJoCo derived quantities (xpos, xquat, etc.) ---
                mujoco.mj_forward(mujoco_sim.mjModel, mujoco_sim.mjData)

                # --- Sync to Isaac Sim and render ---
                if isaac_sim is not None:
                    isaac_sim.step(mujoco_sim)

                # --- Record frame ---
                if exporter is not None and isaac_sim is not None:
                    rendered = isaac_sim.render()
                    isaac_image = rendered["head_stereo_left"]
                    frame = _build_replay_frame(row, isaac_image, features)
                    exporter.add_frame(frame)

                # Pace to dataset fps (skip when recording for speed)
                if not record:
                    elapsed = time.monotonic() - step_start
                    sleep_time = frame_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            # --- Save episode ---
            if exporter is not None:
                exporter.save_episode()
                if env_conf is not None:
                    _save_replay_episode_env_config(exporter, env_conf, episodes_saved) 
                episodes_saved += 1
                print(f"[Record] Episode {episodes_saved} saved")

            print(f"[Replay] Episode {ep_idx} done")

        print(f"\n[Replay] All {num_episodes} episodes replayed")
    except Exception as e:
        print(f"[Replay] Error: {e}")
        raise
    finally:
        if exporter is not None:
            exporter.stop_video_writers()
            print(f"[Record] Done. {episodes_saved} episodes saved to {save_dir}")
        env.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
