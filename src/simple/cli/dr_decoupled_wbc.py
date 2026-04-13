"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

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
from typing_extensions import Annotated, Any, TYPE_CHECKING
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
import tyro
if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv
    from simple.core.task import Task

from simple.agents.pico_decoupled_agent import PicoDecoupledAgent
from simple.cli.render_decoupled_wbc import _load_episodes, _load_episode_configs
from simple.robots.g1_sonic import G1Sonic
from simple.utils import NumpyArrayEncoder

def _make_sonic_config() -> dict[str, Any]:
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config

def _init_exporter(save_dir: str, task_prompt: str, robot_model, obj_names: list[str], joint_names: list[str]):
    """Create a Gr00tDataExporter for LeRobot-format recording."""
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

    features = get_dataset_features(robot_model)
    features["observation.state"]["names"] = joint_names

    modality_config = get_modality_config(robot_model)

    # Add object poses feature
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

def _save_episode_env_config(exporter, env_config: dict, episode_index: int):
    """Write the episode's environment_config into episodes.jsonl."""
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    lines[episode_index]["environment_config"] = json.dumps(env_config, cls=NumpyArrayEncoder)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyXMovePick-v0",
    data_dir: Annotated[str, typer.Option()] = "data/replay_decoupled_wbc/simple/G1WholebodyXMovePick-v0/level-0",
    dr_level: Annotated[int, typer.Option()] = 0,
    num_episodes: Annotated[int, typer.Option()] = -1,
    render_hz: Annotated[int, typer.Option()] = 50,
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 600,
    success_criteria: Annotated[float, typer.Option()] = 0.7,
    save_dir: Annotated[str, typer.Option()] = "data/evals",
):
    """Create minimal evaluation dataset with first frame and environment_config."""
    sonic_config = _make_sonic_config()
    print(f"Creating environment: {env_id}")
    env = gym.make(
        env_id,
        sim_mode="mujoco_isaac",
        render_hz=render_hz,
        headless=headless,
        max_episode_steps=max_episode_steps,
        success_criteria=success_criteria,
        sonic_config=sonic_config,
        # dr_level=dr_level,
    )
    sonic_env:SonicLocoManipEnv = env.unwrapped  # type: ignore
    task: Task = env.unwrapped.task  # type: ignore
    robot = task.robot
    print(f"DR level: {dr_level}")

    agent = PicoDecoupledAgent(task.robot)  # type: ignore
    agent.num_episodes = num_episodes
    agent._wbc_policy.lower_body_policy.use_policy_action = True

    # Load episodes from parquet
    print(f"Loading dataset from {data_dir} ...")
    episodes = _load_episodes(data_dir)
    total_episodes = len(episodes)
    if num_episodes < 0:
        num_episodes = total_episodes
    num_episodes = min(num_episodes, total_episodes)

    # Load per-episode environment configs
    episode_configs = _load_episode_configs(data_dir)

    # Initialize exporter
    exporter = None
    episodes_saved = 0
    obj_names = []
    run_save_dir = f"{os.path.abspath(save_dir)}/{sonic_env.spec.id}/dr-level-{dr_level}"
    run_save_path = Path(run_save_dir)

    # Clean up existing dataset
    if run_save_path.exists():
        print(f"Removing existing dataset at {run_save_dir}")
        shutil.rmtree(run_save_path)

    try:
        for ep_idx in tqdm(range(num_episodes), desc="Episodes", unit="episode"):
            if ep_idx not in episodes:
                print(f"Episode {ep_idx} not found, skipping")
                continue

            # Reset environment with the exact scene configuration
            env_conf = episode_configs[ep_idx]
            observation, privileged_info = env.reset(options={"state_dict": env_conf, "dr_level": dr_level})

            # Initialize exporter on first episode
            if exporter is None:
                obj_names = list(sonic_env.mujoco.mj_objects.keys())
                exporter = _init_exporter(
                    run_save_dir,
                    task.instruction,
                    agent._dwbc_robot_model,
                    obj_names,
                    robot.joint_names
                )
                print(f"Exporter initialized, saving to {run_save_dir}")
                print(f"Recording {len(obj_names)} objects: {obj_names}")

            # Build frame dict from observation (first frame only)
            # Use observation["joint_qpos"] which contains all 43D joints (body 29 + hands 7+7)
            base_vel_3d = observation.get("base_lin_vel", np.zeros(3))
            base_ang_vel_3d = observation.get("base_ang_vel", np.zeros(3))
            base_vel_6d = np.concatenate([base_vel_3d, base_ang_vel_3d]) if len(base_vel_3d) == 3 else base_vel_3d

            frame = {
                "observation.images.ego_view": observation["head_stereo_left"],
                "observation.state": np.asarray(observation["joint_qpos"], dtype=np.float64),
                "observation.base_pose": np.asarray(observation.get("base_pose_quat", observation.get("floating_base_pose", np.zeros(7))), dtype=np.float64),
                "observation.base_vel": np.asarray(base_vel_6d, dtype=np.float64),
                "observation.eef_state": np.zeros(14, dtype=np.float64),  # right hand EEF (7D pose + 7D vel)
                "observation.img_state_delta": np.zeros(1, dtype=np.float32),  # placeholder - float32!
                "teleop.navigate_command": np.zeros(4, dtype=np.float64),  # vx, vy, vyaw, jump
                "teleop.base_height_command": np.zeros(1, dtype=np.float64),  # height command
                "action": np.zeros(43, dtype=np.float64),  # no action in eval dataset
                "action.eef": np.zeros(14, dtype=np.float64),  # EEF action placeholder
            }

            # Add object poses if available, otherwise default
            num_objects = len(obj_names)
            if "object_poses" in observation:
                frame["observation.object_poses"] = np.asarray(
                    observation["object_poses"], dtype=np.float64
                )
            elif num_objects > 0:
                # Default zeros for each object (7D: pos xyz + quat wxyz)
                frame["observation.object_poses"] = np.zeros(num_objects * 7, dtype=np.float64)

            # Add frame to exporter
            exporter.add_frame(frame)

            # Save episode and write environment config
            exporter.save_episode()
            # update env config
            env_config = task.state_dict()
            _save_episode_env_config(exporter, env_config, episodes_saved)
            episodes_saved += 1
            print(f"[Record] Episode {episodes_saved} saved")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        import traceback
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        if exporter is not None:
            exporter.stop_video_writers()
            print(f"[Record] Done. {episodes_saved} episodes saved to {run_save_dir}")
        env.close()
        agent.close()

def typer_main():
    typer.run(main)

if __name__ == "__main__":
    typer.run(main)