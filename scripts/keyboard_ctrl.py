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
import typer
from typing_extensions import Annotated

def main(
    env_id: str = "simple/FrankaTabletopGrasp-v0",
    task: str = "franka_tabletop_grasp",
    robot_uid: str = "franka_fr3",
    controller_uid: str = "pd_joint_pos",
    target_object: str = "graspnet1b:63",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 50,
    save_dir: Annotated[str, typer.Option()] = "./output",
):
    env = gym.make(
        env_id,
        task=task,
        robot_uid=robot_uid,
        controller_uid=controller_uid, 
        target_object=target_object,
        sim_mode=sim_mode,
        headless=headless,
        max_episode_steps=max_episode_steps,
    )

    output_eval_dir = f"{save_dir}/keyboard_ctrl"
    os.makedirs(output_eval_dir, exist_ok=True)
    env = VideoRecorder(env=env, video_folder=f"{output_eval_dir}", write_png=False)

    observation, info = env.reset()
    agent = KeyboardAgent(env.unwrapped.task.robot)

    episode_over = False
    while not episode_over:
        env.unwrapped.simulation_app.update() # type: ignore
        action = agent.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f'terminated: {terminated}, truncated: {truncated}')
        episode_over = terminated        

    env.close()
if __name__ == "__main__":
    typer.run(main)