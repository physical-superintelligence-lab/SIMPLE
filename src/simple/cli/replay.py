# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
import sys
import typer
from typing import Optional
from typing_extensions import Annotated
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
import json
import pickle
import shutil
from pathlib import Path

import simple.envs as _ # import all envs
from simple.cli._replay_checks import require_episode_task
from simple.envs.wrappers import StandStabilizationWrapper, VideoRecorder
from simple.agents.replay_agent import ReplayAgent
import itertools
spinner = itertools.cycle(['|', '/', '-', '\\'])

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/FrankaTabletopGrasp-v0",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 300,
    render_hz: Annotated[Optional[int], typer.Option()] = None, # default: the dataset fps (30 for non-lerobot data)
    data_format: Annotated[str, typer.Option()] = "lerobot",
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    replay_dir: Annotated[str, typer.Option("--replay-dir", "--save-dir")] = "data/replays", # --save-dir: docker.sh sharding
    num_episodes: Annotated[int, typer.Option()] = 50,
    episode_start: Annotated[int, typer.Option()] = 0, # replay from episode_start to episode_start + num_episodes
    success_criteria: Annotated[float, typer.Option()] = 0.9,
    video_keys: Annotated[str, typer.Option()] = "head_stereo_left,head_stereo_right", # comma-separated cameras to save
    record: Annotated[bool, typer.Option()] = False, # save episodes as datagen.py does (LerobotRecorder)
    record_dir: Annotated[Optional[str], typer.Option()] = None, # default: <replay-dir>/record (per docker.sh shard)
    save_failed: Annotated[bool, typer.Option()] = True, # --record also saves failed episodes; replay.txt lists outcomes
    stop_on_success: Annotated[bool, typer.Option()] = True, # --no-stop-on-success: replay every action (full-length video, e.g. to re-render a dataset)
):
    os.makedirs(replay_dir, exist_ok=True)

    upsample_factor = 1 # default
    # Load dataset
    if data_format == "lerobot":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from simple.datasets.lerobot import get_episode_lerobot as get_episode
        dataset = LeRobotDataset(repo_id=env_id, root=data_dir, video_backend="pyav")
        num_episodes = min(num_episodes, dataset.num_episodes - episode_start)
        dataset_fps = dataset.meta.fps
        if render_hz is None:
            render_hz = dataset_fps
        if render_hz != dataset_fps:
            print("=" * 50)
            print(f"Warning: overriding dataset render_hz {dataset_fps} with user-specified value {render_hz}. This may cause control gap.")
            print("=" * 50)
            assert render_hz % dataset_fps == 0, f"render_hz {render_hz} must be a multiple of the dataset fps {dataset_fps}"
            upsample_factor = render_hz // dataset_fps
            
        print(f"loaded {num_episodes} episodes.")

    elif data_format == "rlds" or data_format == "rlds-legacy":
        render_hz = render_hz or 30
        import tensorflow_datasets as tfds
        from simple.datasets.rlds import get_episode_rlds as get_episode
        builder = tfds.builder_from_directory(data_dir)
        dataset = builder.as_dataset(split='train')
        num_episodes = min(num_episodes, int(dataset.cardinality().numpy()) - episode_start)
        print(f"loaded {num_episodes} episodes.")
        
    elif data_format == "rlds_numpy":
        data_path = Path(data_dir)
        assert data_path.exists(), f"Data path {data_path} does not exist."
        dataset = sorted(list(data_path.glob("*episode_*.pkl")))
        num_episodes = min(num_episodes, len(dataset) - episode_start)
        render_hz = 3 # hardcoded for old vlt-grasp dataset

        from simple.datasets.rlds import convert_env_config_to_state_dict
        def get_episode(dataset, idx, *_,**__):
            with open(dataset[idx], "rb") as f:
                episode = pickle.load(f)
            env_cfg = json.loads(episode["task"]['environment_config'][0].decode("utf-8"))
            task_uuid = episode["task"]["uuid"][0].decode("utf-8")
            
            state_dict = convert_env_config_to_state_dict(env_cfg)
            return state_dict, episode#, task_id

    # Create environment
    grasp_env = gym.make(
        env_id,
        render_hz=render_hz,
        sim_mode=sim_mode,
        max_episode_steps=max_episode_steps,
        headless=headless,
        success_criteria=success_criteria,
    )
    # Stand-warmup is now done inside env.reset for Humanoid robots so the agent
    # sees a stabilized robot from step 0. No-op for non-Humanoid.
    grasp_env = StandStabilizationWrapper(grasp_env)
    task = grasp_env.unwrapped.task # type: ignore
    # Render only the head cameras: the layout (and so the sim's cameras), the observations and the
    # recorded features all come from task.sensor_cfgs.
    task.sensor_cfgs = {k: v for k, v in task.sensor_cfgs.items() if k.startswith("head")}
    grasp_env.unwrapped.observation_space = task.observation_space # type: ignore
    recorder = None
    if record:
        # Record like datagen.py (LerobotRecorder), so scripts/postprocess_psi0.py turns the recording into
        # Psi0 data; saving is decided per episode (auto_save=False): by the replay's final success, or always with --save-failed.
        from simple.envs.lerobot import LerobotRecorder
        save_dir = record_dir or f"{replay_dir}/record"
        run_save_dir = f"{os.path.abspath(save_dir)}/{env_id}/level-{task.dr.level}"
        if os.path.exists(run_save_dir):
            shutil.rmtree(run_save_dir)
        recorder = LerobotRecorder(env=grasp_env, root_dir=save_dir, auto_save=False)
    # Loop through episodes
    stats = defaultdict(bool)
    for idx in tqdm(range(episode_start, episode_start + num_episodes)):

        if data_format == "lerobot": # the replay needs no images
            env_conf, episode = get_episode(dataset, idx, data_format, load_videos=False) # type: ignore
        else:
            env_conf, episode = get_episode(dataset, idx, data_format) # type: ignore
        require_episode_task(env_conf, task, idx, data_dir, env_id)
        task_id = f"episode_{idx}"

        # Wrap a video recorder for visualization
        env = VideoRecorder(env=recorder or grasp_env, video_folder=f"{replay_dir}", name_prefix=task_id, framerate=render_hz, write_png=False,
                            video_keys=video_keys.split(","))

        # if env_conf["dr_state_dict"]["scene"]["uid"] != 'hssd:scene1' and env_conf["dr_state_dict"]["scene"]["uid"] != 'hssd:scene3':
        #     continue
        # Reset environment with episode-specific config
        observation, info = env.reset(options={"state_dict": env_conf})
        instruction = task.instruction
        j = 0

        # Create replay agent
        replay_agent = ReplayAgent(episode, task.robot, data_format=data_format, upsample_factor=upsample_factor)

        with tqdm(total=0, bar_format='{desc}', ncols=80, file=sys.stdout) as t:
            # Run the episode
            episode_over = False
            while not episode_over:
                try:
                    action = replay_agent.get_action(observation, info=info, instruction=instruction)
                    observation, reward, terminated, truncated, info = env.step(action)
                    j += 1
                    if terminated and stop_on_success: # task success; stop here (env._success is re-evaluated every step)
                        episode_over = True
                        print(f"Episode succeeded at step {j}.")
                except StopIteration: # ReplayAgent's queue is exhausted (PrimitiveAgent turns IndexError into StopIteration)
                    episode_over = True
                    print(f"Episode completed: all actions executed.")

                t.set_description_str(f"Replaying {next(spinner)}")
                sys.stdout.flush()
            
        is_success = grasp_env.unwrapped._success # type: ignore
        stats[task_id] = is_success
        # Save per-episode video
        if isinstance(env, VideoRecorder):
            env.release()
        if recorder is not None:
            recorder.finish_episode(is_success or save_failed)
            with open(Path(recorder.root_dir) / "meta" / "replay.txt", "a") as f:
                f.write(f"{idx}: {is_success}\n")

        """ # write the dict into a pickle file
        import pickle
        with open(f"replay_{idx}_debug_ctrl_err.pkl", "wb") as f:
            pickle.dump(task.robot._debug_ctrl_err, f) # type: ignore
            print("write out control error dict")
        break """

    # Log replay results
    sr = sum(stats.values()) / max(len(stats), 1) # no episode in range (e.g. an empty docker.sh shard)
    tqdm.write(f"Success rate: {sr}")
    with open(f"{replay_dir}/replay_stats.txt", "a+") as f:
        for task_id, result in stats.items():
            f.write(f"{task_id}: {result} \n")
        f.write(f"success rate: {sr:.2f} \n")
        f.write("================\n")
    
    if recorder is not None:
        tqdm.write(f"[Record] {recorder.episode_index} episodes saved to {recorder.root_dir}")

    # Finally close the environment
    grasp_env.close()

def typer_main():
    typer.run(main)

if __name__ == "__main__":
    typer.run(main)
