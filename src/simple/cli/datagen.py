# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import typer
from typing import Optional
from typing_extensions import Annotated
import gymnasium as gym
from tqdm import tqdm
import simple.envs as _# import all envs
from simple.mp.curobo import CuRoboPlanner
from simple.agents.mp import MotionPlannerAgent
from simple.envs.lerobot import LerobotRecorder
from simple.envs.psi0_recorder import Psi0Recorder
from simple.envs.wrappers import StandStabilizationWrapper
import traceback
import sys
import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
import json
import shutil
from pathlib import Path
PRERESET_TASK_STATE = "" #"examples/demo_task_state_dict.json" # 

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/FrankaTabletopGrasp-v0",
    scene_uid: Annotated[str, typer.Option()] = "hssd:scene1",
    target_object: str | None = None,
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    webrtc: Annotated[bool, typer.Option()] = True,  # Enable WebRTC streaming
    max_episode_steps: Annotated[int, typer.Option()] = 1000,
    render_hz: Annotated[Optional[int], typer.Option()] = None, # default: 50 for psi0 (the psi0 data fps), else 30
    # lerobot: raw LeRobot recording (LerobotRecorder); psi0: G1 wholebody (AMO) data in the psi0 training
    # format (states/action 32/36-D, as SIMPLE-decoupled-wbc), replayable with simple.cli.replay
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
    assert data_format in ("lerobot", "psi0"), f"unsupported --data-format {data_format!r} (lerobot | psi0)"
    if render_hz is None:
        render_hz = 50 if data_format == "psi0" else 30
    # create environment
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

    # create motion planner
    render_hz = task.metadata["render_hz"]
    planner = CuRoboPlanner(
        robot=task.robot,
        plan_dt=0.01,
        plan_batch_size = 1,
        easy_motion_gen = easy_motion_gen,
        ignore_target_collisions = ignore_target_collision,
    )
    # create the data generation agent with a solver for the task
    mp_agent = MotionPlannerAgent(task, planner, debug=debug, plan_batch_size=plan_batch_size)

    # create data recorder wrapper, depending on data format
    # in this example we only support lerobot format
    if data_format == "lerobot":
        env = LerobotRecorder(env=env, agent=mp_agent, shard_size=shard_size, root_dir=save_dir, debug=debug) # +"."+env.unwrapped.task.uid
    else:
        # stand warm-up inside reset (as eval and replay do), not recorded: frame 0 is the stabilized robot
        env = Psi0Recorder(env=StandStabilizationWrapper(env), agent=mp_agent, root_dir=save_dir)
        print(f"[Datagen] psi0 dataset: {env.root_dir} ({env.writer.num_episodes} existing episodes)")
    recorder = env
    
    if os.path.exists(PRERESET_TASK_STATE):
        # load task state dict if exists
        state_dict = json.load(open(PRERESET_TASK_STATE, "r"))
    else:
        # randomly reset the episode
        state_dict = None
    
    observation = None
    info = None
    success_count = 0

    def execute_action_sequence(stage: str):
        nonlocal observation, info, success_count
        while True:
            try:
                action = mp_agent.get_action(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if terminated:
                        success_count += 1
                    return "episode_end"
            except StopIteration:
                if stage == "phase":
                    return "phase_done"
                print("Motion plan exhausted before episode end.")
                recorder.clear_episode_buffer()
                return "episode_end"
            except Exception as e:
                print(f"Error during action execution: {e}", file=sys.stderr)
                recorder.clear_episode_buffer()
                traceback.print_exc()
                if sys.gettrace() is not None:
                    raise
                return "episode_end"
    
    while success_count < num_episodes:
        state_dict = None
        observation, info = env.reset(options={"state_dict": state_dict})
        
        phase = 1
        while True:
            try:
                state = mp_agent.synthesize()
                if state is False:
                    # env.env_cfgs.pop()
                    break
                if state == "phase_break":
                    print(f"[Datagen] Executing phase {phase} before continuing...")
                    result = execute_action_sequence("phase")
                    if result == "phase_done":
                        print(f"[Datagen] Phase {phase} completed, continuing to phase {phase + 1}...")
                        phase += 1
                        continue
                    else:
                        break
                else:
                    execute_action_sequence("final")
                    break
            except Exception as e:
                print(f"Error during episode execution: {e}", file=sys.stderr)
                traceback.print_exc()
                print("Motion planning failed during synthesis.")
                # raise e
                env.close()
                exit(1)

        mp_agent.reset()
        """ # write the dict into a pickle file
        import pickle
        with open(f"control_{success_count}_debug_ctrl_err.pkl", "wb") as f:
            pickle.dump(task.robot._debug_ctrl_err, f) # type: ignore
            print("write out control error dict")
        break """


    # pbar.close()
    # close the environment and finalize data saving
    env.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
