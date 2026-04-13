import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
import sys
import typer
from typing_extensions import Annotated
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
import json
import pickle
from pathlib import Path

import simple.envs as _ # import all envs
from simple.envs.wrappers import VideoRecorder
from simple.agents.replay_agent import ReplayAgent
import itertools
spinner = itertools.cycle(['|', '/', '-', '\\'])

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/FrankaTabletopGrasp-v0",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 300,
    render_hz: Annotated[int, typer.Option()] = 30, # to be overwritten by dataset
    data_format: Annotated[str, typer.Option()] = "lerobot",
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    replay_dir: Annotated[str, typer.Option()] = "data/replays",
    num_episodes: Annotated[int, typer.Option()] = 50,
    success_criteria: Annotated[float, typer.Option()] = 0.9,
    is_postprocess: Annotated[bool, typer.Option()] = False,
):
    os.makedirs(replay_dir, exist_ok=True)

    upsample_factor = 1 # default
    # Load dataset
    if data_format == "lerobot":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from simple.datasets.lerobot import get_episode_lerobot as get_episode
        dataset = LeRobotDataset(repo_id=env_id,root=data_dir)
        num_episodes = min(num_episodes, dataset.num_episodes)
        dataset_fps = dataset.meta.fps
        if render_hz != dataset_fps:
            print("=" * 50)
            print(f"Warning: overriding dataset render_hz {dataset_fps} with user-specified value {render_hz}. This may cause control gap.")
            print("=" * 50)
            upsample_factor = render_hz // dataset_fps
            
        print(f"loaded {num_episodes} episodes.")

    elif data_format == "rlds" or data_format == "rlds-legacy":
        import tensorflow_datasets as tfds
        from simple.datasets.rlds import get_episode_rlds as get_episode
        builder = tfds.builder_from_directory(data_dir)
        dataset = builder.as_dataset(split='train')
        num_episodes = dataset.cardinality().numpy()
        print(f"loaded {num_episodes} episodes.")
        
    elif data_format == "rlds_numpy":
        data_path = Path(data_dir)
        assert data_path.exists(), f"Data path {data_path} does not exist."
        dataset = sorted(list(data_path.glob("*episode_*.pkl")))
        num_episodes = min(num_episodes, len(dataset))
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
    task = grasp_env.unwrapped.task # type: ignore
    # Loop through episodes
    stats = defaultdict(bool)
    for idx in tqdm(range(num_episodes)):
        
        env_conf, episode = get_episode(dataset, idx, data_format) # type: ignore
        assert env_conf["uid"] == task.uid, f"Environment and dataset mismatch: {env_conf['uid']} vs {task.uid}"
        task_id = f"episode_{idx}"

        # Wrap a video recorder for visualization
        env = VideoRecorder(env=grasp_env, video_folder=f"{replay_dir}", name_prefix=task_id, framerate=render_hz, write_png=False)

        # if env_conf["dr_state_dict"]["scene"]["uid"] != 'hssd:scene1' and env_conf["dr_state_dict"]["scene"]["uid"] != 'hssd:scene3':
        #     continue
        # Reset environment with episode-specific config
        observation, info = env.reset(options={"state_dict": env_conf})
        instruction = task.instruction
        j = 0

        # Create replay agent
        replay_agent = ReplayAgent(episode, task.robot, data_format=data_format, upsample_factor=upsample_factor,is_postprocess=is_postprocess)

        with tqdm(total=0, bar_format='{desc}', ncols=80, file=sys.stdout) as t:
            # Run the episode
            episode_over = False
            last_action = None
            while not episode_over:
                try:
                    action = replay_agent.get_action(observation, info=info, instruction=instruction)
                    observation, reward, terminated, truncated, info = env.step(action)
                    last_action = action
                    j += 1
                except StopIteration:
                    episode_over = True
                    for _ in range(2): # hack for env to settle down
                        observation, reward, terminated, truncated, info = env.step(last_action)
                    print(f"Episode completed: all actions executed.")
        
                except IndexError:
                    episode_over = True
                    print("Episode finished.")

                t.set_description_str(f"Replaying {next(spinner)}")
                sys.stdout.flush()
            
        is_success = grasp_env.unwrapped._success # type: ignore
        stats[task_id] = is_success
        # Save per-episode video
        if isinstance(env, VideoRecorder):
            env.release()

        """ # write the dict into a pickle file
        import pickle
        with open(f"replay_{idx}_debug_ctrl_err.pkl", "wb") as f:
            pickle.dump(task.robot._debug_ctrl_err, f) # type: ignore
            print("write out control error dict")
        break """

    # Log replay results
    sr = sum(stats.values())/len(stats)
    tqdm.write(f"Success rate: {sr}")
    with open(f"{replay_dir}/replay_stats.txt", "a+") as f:
        for task_id, result in stats.items():
            f.write(f"{task_id}: {result} \n")
        f.write(f"success rate: {sr:.2f} \n")
        f.write("================\n")
    
    # Finally close the environment
    grasp_env.close()

def typer_main():
    typer.run(main)

if __name__ == "__main__":
    typer.run(main)