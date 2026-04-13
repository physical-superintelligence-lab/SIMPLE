"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

# from simple.args import args
import os
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for offscreen rendering
import gymnasium as gym
import simple.envs # import all envs
import numpy as np
from simple.envs.wrappers import VideoRecorder
import typer
import json
from typing_extensions import Annotated
from simple.core.action import ActionCmd

PRERESET_TASK_STATE = "examples/demo_task_state_dict.json"

def main(
    env_id: str = "simple/FrankaTabletopGrasp-v0",
    task: str = "franka_tabletop_grasp",
    robot_uid: str = "franka_fr3",
    controller_uid: str = "pd_joint_pos",
    scene_uid: str = "hssd:scene3",
    target_object: str = "graspnet1b:63",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = True,  # Must be True for WebRTC
    webrtc: Annotated[bool, typer.Option()] = True,  # Enable WebRTC streaming
    max_episode_steps: Annotated[int, typer.Option()] = 6000,
    save_dir: Annotated[str, typer.Option()] = "data/output",
):
    env = gym.make(
        env_id,
        task=task,
        robot_uid=robot_uid,
        controller_uid=controller_uid, 
        target_object=target_object,
        scene_uid=scene_uid,
        sim_mode=sim_mode,
        headless=headless,
        webrtc=webrtc,
        max_episode_steps=max_episode_steps,
    )

    output_eval_dir = f"{save_dir}/test_env"
    os.makedirs(output_eval_dir, exist_ok=True)
    env = VideoRecorder(env=env, video_folder=f"{output_eval_dir}", write_png=True)

    if os.path.exists(PRERESET_TASK_STATE):
        # load task state dict if exists
        state_dict = json.load(open(PRERESET_TASK_STATE, "r"))
    else:
        # randomly reset the episode
        state_dict = None
    #for debug
    state_dict = None
    observation, info = env.reset(options={"state_dict": state_dict})
    episode_over = False
    while not episode_over:
        # sample random actions 
        action = env.unwrapped.task.robot.random_action()

        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
    env.close()

   

if __name__ == "__main__":
    typer.run(main)