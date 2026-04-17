"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import gymnasium as gym
import simple.envs # import all envs
import numpy as np
from simple.envs.wrappers import VideoRecorder
from simple.agents.keyboard_agent import KeyboardAgent
import os
import json
import typer
from typing_extensions import Annotated
from copy import deepcopy


def load_episodes_configs(episodes_jsonl_path: str) -> list[dict]:
    """Load and parse environment configs from an episodes.jsonl file."""
    configs = []
    with open(episodes_jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            if "environment_config" in entry:
                env_config = json.loads(entry["environment_config"])
                configs.append(env_config)
    return configs


def swap_target_in_config(env_config: dict, new_target_uid: str) -> dict:
    """Swap the target object with a distractor in the environment config."""
    config = deepcopy(env_config)
    dr_state = config["dr_state_dict"]

    old_target = dr_state["target"]
    old_target_uid = old_target["uid"]

    distractors = dr_state["distractors"]
    if new_target_uid not in distractors:
        raise ValueError(
            f"Distractor UID {new_target_uid} not found. "
            f"Available: {list(distractors.keys())}"
        )

    new_target = distractors[new_target_uid]
    dr_state["target"] = new_target
    del distractors[new_target_uid]
    distractors[old_target_uid] = old_target
    return config


def list_all_objects(env_config: dict) -> list[dict]:
    """List all objects (target + distractors) in an environment config."""
    dr_state = env_config["dr_state_dict"]
    objects = []

    from simple.assets.graspnet import GraspNet_1B_Object_Names

    def _resolve_name(entry, uid):
        name = entry.get("name")
        if name:
            return name
        int_uid = int(uid) if isinstance(uid, (str, int)) else uid
        return GraspNet_1B_Object_Names.get(int_uid, str(uid))

    target = dr_state["target"]
    objects.append({
        "uid": target["uid"],
        "name": _resolve_name(target, target["uid"]),
        "role": "target",
    })

    for uid, dist in dr_state["distractors"].items():
        objects.append({
            "uid": uid,
            "name": _resolve_name(dist, uid),
            "role": "distractor",
        })
    return objects


def interactive_select_object(env_config: dict, episode_idx: int) -> dict | str | None:
    """Interactively prompt the user to select which object to grasp."""
    objects = list_all_objects(env_config)
    print(f"\n{'='*50}")
    print(f"  Episode {episode_idx} - Select object to grasp")
    print(f"{'='*50}")
    for i, obj in enumerate(objects):
        marker = " (current target)" if obj["role"] == "target" else ""
        print(f"  [{i}] {obj['name']} (uid={obj['uid']}){marker}")
    print(f"  [s] Skip this episode")
    print(f"  [q] Quit")
    print(f"{'='*50}")

    while True:
        choice = input("Choose object index (or s/q): ").strip().lower()
        if choice == "q":
            return None
        if choice == "s":
            return "skip"
        try:
            idx = int(choice)
            if 0 <= idx < len(objects):
                selected = objects[idx]
                if selected["role"] == "target":
                    return deepcopy(env_config)
                else:
                    return swap_target_in_config(env_config, selected["uid"])
            else:
                print(f"  Invalid index. Enter 0-{len(objects)-1}, s, or q.")
        except ValueError:
            print(f"  Invalid input. Enter 0-{len(objects)-1}, s, or q.")


def main(
    env_id: str = "simple/FrankaTabletopGraspMP-v0",
    task: str = "franka_tabletop_grasp_mp",
    robot_uid: str = "franka_fr3",
    controller_uid: str = "pd_joint_pos",
    target_object: str = "graspnet1b:63",
    scene_uid: str = "hssd:scene1",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 50000,
    save_dir: Annotated[str, typer.Option()] = "./output",
    replay_episodes: Annotated[str, typer.Option(help="Path to episodes.jsonl to replay scenes from")] = "",
    episode_index: Annotated[int, typer.Option(help="Index of episode to load from episodes.jsonl (default: 0)")] = 0,
    interactive: Annotated[bool, typer.Option(help="Interactively select which object to grasp")] = False,
):
    # Load env_config from episodes.jsonl if provided
    state_dict = None
    if replay_episodes and os.path.exists(replay_episodes):
        episodes_configs = load_episodes_configs(replay_episodes)
        if not episodes_configs:
            print(f"[KeyboardCtrl] No valid episodes found in {replay_episodes}")
        else:
            print(f"[KeyboardCtrl] Loaded {len(episodes_configs)} episodes from {replay_episodes}")
            if interactive:
                result = interactive_select_object(episodes_configs[episode_index % len(episodes_configs)], episode_index)
                if result is None or result == "skip":
                    print("[KeyboardCtrl] User cancelled.")
                    return
                state_dict = result
            else:
                state_dict = deepcopy(episodes_configs[episode_index % len(episodes_configs)])
            assert isinstance(state_dict, dict)
            target_name = state_dict["dr_state_dict"]["target"].get("name", "unknown")
            print(f"[KeyboardCtrl] Loading episode {episode_index}: target={target_name}")

    env = gym.make(
        env_id,
        task=task,
        robot_uid=robot_uid,
        controller_uid=controller_uid, 
        target_object=target_object,
        scene_uid=scene_uid,
        sim_mode=sim_mode,
        headless=headless,
        max_episode_steps=max_episode_steps,
    )

    output_eval_dir = f"{save_dir}/keyboard_ctrl"
    os.makedirs(output_eval_dir, exist_ok=True)
    env = VideoRecorder(env=env, video_folder=f"{output_eval_dir}", write_png=False)

    observation, info = env.reset(options={"state_dict": state_dict})
    agent = KeyboardAgent(env.unwrapped.task.robot) # type: ignore

    episode_over = False
    step = 0
    while not episode_over:
        sim_app = env.unwrapped.simulation_app # type: ignore
        if sim_app is not None:
            sim_app.update()
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated
        if step >= max_episode_steps - 1:
            print(f"[KeyboardCtrl] Reached max episode steps ({max_episode_steps}). Ending episode.")
            break        
        step+=1
    env.close()


if __name__ == "__main__":
    typer.run(main)