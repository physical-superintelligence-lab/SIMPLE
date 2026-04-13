"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import typer
from typing_extensions import Annotated, TYPE_CHECKING
import gymnasium as gym
import simple.envs as _# import all envs

if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

from simple.agents.pico_sonic_agent import PicoSonicAgent
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
import time
import tyro

PRERESET_TASK_STATE = "" #"examples/demo_task_state_dict.json" # 



def main(
    env_id: Annotated[str, typer.Argument()] = "simple/FrankaTabletopGrasp-v0",
    scene_uid: Annotated[str, typer.Option()] = "hssd:scene1",
    target_object: str | None = None,
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
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
):  
    assert sim_mode in ["mujoco"], f"Invalid sim_mode {sim_mode} for teleop."
    sim_cnt = 0

    # adapted from run_sim_loop.main
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[]) # FIXME
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple" # self.task.label

    print(f"Creating environment: {env_id}")
    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=render_hz,
        headless=headless,
        max_episode_steps=max_episode_steps,
        sonic_config=sonic_config
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped # type: ignore
    task = sonic_env.task
    robot = task.robot

    agent = PicoSonicAgent(robot) # teleop g1 using sonic + pico

    observation, info = env.reset()
    try:
        # adapted from base_sim.start
        while (
            (sonic_env.mujoco.viewer and sonic_env.mujoco.viewer.is_running())
            or (sonic_env.mujoco.viewer is None)
        ):
            step_start = time.monotonic()

            action = agent.get_action(observation, instruction=task.instruction)
            observation, reward, terminated, truncated, info = env.step(action)

            if "proprio" in info:
                agent.publish_low_state(info["proprio"])

            if agent._reset_requested:
                agent._reset_requested = False
                observation, info = env.reset()
                sim_cnt = 0
                print("[Teleop] Environment reset complete")

            if sim_cnt % int(robot.viewer_dt / robot.sim_dt) == 0:
                sonic_env.update_viewer()

            if sim_cnt % int(robot.reward_dt / robot.sim_dt) == 0:
                sonic_env.update_reward()

            if sim_cnt % int(robot.image_dt / robot.sim_dt) == 0:
                agent.update_render_caches(observation)

            # Simple rate limiter (replaces ROS rate)
            elapsed = time.monotonic() - step_start
            sleep_time = robot.sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            sim_cnt += 1
    except KeyboardInterrupt:
        print("Simulator interrupted by user.")
    finally:
        env.close()
        agent.close()
    
def typer_main():
    typer.run(main)

if __name__ == "__main__":
    typer.run(main)