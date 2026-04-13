# Replay Recorded Data

import all envs
```
...
import simple.envs as _ # import all envs
...
```

pass in `env-id` and `data-dir`

```
def main(
    env_id: Annotated[str, typer.Argument()] = "simple/FrankaTabletopGrasp-v0",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 150,
    render_hz: Annotated[int, typer.Option()] = 30,
    data_format: Annotated[str, typer.Option()] = "lerobot",
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    replay_dir: Annotated[str, typer.Option()] = "data/replays",
    num_episodes: Annotated[int, typer.Option()] = 1,
):
    os.makedirs(replay_dir, exist_ok=True)
```

Load dataset

```
    dataset = LeRobotDataset(repo_id=env_id,root=data_dir)
    num_episodes = min(num_episodes, dataset.num_episodes)
```

Create environment
```
    grasp_env = gym.make(
        env_id,
        sim_mode=sim_mode,
        max_episode_steps=max_episode_steps, 
        headless=headless,
    )
    task = grasp_env.unwrapped.task # type: ignore
```

Loop through episodes
```
    stats = defaultdict(bool)
    for idx in tqdm(range(num_episodes)):
```

Extract episode data
```
        from_idx = int(dataset.episode_data_index["from"][idx].item())
        to_idx = int(dataset.episode_data_index["to"][idx].item())
        episode = [dataset[i] for i in range(from_idx, to_idx)]
```

Wrap a video recorder for visualization
```
        env = VideoRecorder(env=grasp_env, video_folder=f"{replay_dir}", framerate=render_hz, write_png=False)
```

Reset environment with episode-specific config
```
        task_id = f"episode_{idx}"
        env_conf = json.loads(dataset.meta.episodes[idx]['environment_config'])
        assert env_conf["uid"] == task.uid, f"Environment and dataset mismatch: {env_conf['uid']} vs {task.uid}"
        observation, info = env.reset(options={"state_dict": env_conf})
        instruction = task.instruction
```

Create replay agent
```
        replay_agent = ReplayAgent(episode, task.robot, data_format=data_format)
```

Run the episode
```
        episode_over = False
        while not episode_over:
            try:
                action = replay_agent.get_action(observation, info, instruction=instruction)
                observation, reward, terminated, truncated, info = env.step(action)
            except IndexError:
                episode_over = True
                print("Episode finished.")
```    
Save per-episode video
```
        is_success = grasp_env.unwrapped._success # type: ignore
        stats[task_id] = is_success
        # Save per-episode video
        if isinstance(env, VideoRecorder):
            env.release()
```

Log replay results
```
    sr = sum(stats.values())/len(stats)
    tqdm.write(f"Success rate: {sr}")
    with open(f"{replay_dir}/replay_stats.txt", "a+") as f:
        for task_id, result in stats.items():
            f.write(f"{task_id}: {result} \n")
        f.write(f"success rate: {sr:.2f} \n")
        f.write("================\n")
```

Finally close the environment
```
    grasp_env.close()
```

Main entrance of the script
```
if __name__ == "__main__":
    typer.run(main)
```