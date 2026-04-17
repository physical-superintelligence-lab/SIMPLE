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
import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
from simple.envs.wrappers import VideoRecorder
import typer
import json
from typing_extensions import Annotated
from simple.core.action import ActionCmd

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/FrankaTabletopGrasp-v0",
    scene_uid: Annotated[str, typer.Option()] = "hssd:scene1",
    target_object: str | None = None,
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    webrtc: Annotated[bool, typer.Option()] = True,  # Enable WebRTC streaming
    max_episode_steps: Annotated[int, typer.Option()] = 30000,
    render_hz: Annotated[int, typer.Option()] = 30, # FIXME
    data_format: Annotated[str, typer.Option()] = "lerobot",
    save_dir: Annotated[str, typer.Option()] = "data/datagen",
    num_episodes: Annotated[int, typer.Option()] = 100,
    shard_size: Annotated[int, typer.Option()] = 100,
    dr_level: Annotated[int, typer.Option()] = 0,
    plan_batch_size: Annotated[int, typer.Option()] = 1,
    ignore_target_collision: Annotated[bool, typer.Option()] = False,
    debug: Annotated[bool, typer.Option()] = False,
    easy_motion_gen: Annotated[bool, typer.Option()] = False,
    eval: Annotated[bool, typer.Option(help="Only generate env configs")] = False,
):
    make_kwargs = dict(
        scene_uid=scene_uid,
        target_object=target_object,
        sim_mode=sim_mode,
        headless=headless,
        webrtc=webrtc,
        max_episode_steps=max_episode_steps,
        render_hz=render_hz,
        dr_level=dr_level,
    )
    if "Sonic" in env_id or "Teleop" in env_id:
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        sonic_config = SimLoopConfig().load_wbc_yaml()
        sonic_config["ENV_NAME"] = "simple"
        make_kwargs["sonic_config"] = sonic_config
    env = gym.make(env_id, **make_kwargs)
    task = env.unwrapped.task  # type: ignore

    output_eval_dir = f"{save_dir}/test_env"
    os.makedirs(output_eval_dir, exist_ok=True)
    env = VideoRecorder(env=env, video_folder=f"{output_eval_dir}", write_png=True)


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