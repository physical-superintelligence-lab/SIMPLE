# Run an Environment
Script Usage:
```bash
python scripts/test_env.py --help
```
Run with default parameters:
```bash
python scripts/test_env.py
```
By default, a video result will be recoded under `./data/output/test_env`.
You can open this folder to check the simulation results.

If you want to video be played using a `libx264` compatitable player. e.g., `VSCode`, please install `ffmpeg`
```bash
sudo apt-get install ffmpeg
```
> Tips: You might install `ffmpeg` to have video generated in `libx264` format, so that the video can be directly previewed in `VSCode`.

## Detailed Explanations

Common imports:

```python
import os
# Must be set before MuJoCo is imported, so that offscreen rendering uses EGL
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym

# Import all the built-in environments (this is what registers the env ids)
import simple.envs

# Import a wrapper for recording simulation videos
from simple.envs.wrappers import VideoRecorder
```

Create a [Gym-style](https://gymnasium.farama.org/) environment:
```python
# Create a built-in environment
env = gym.make(
    "simple/FrankaTabletopGraspMP-v0",
    task="franka_tabletop_grasp_mp",
    robot_uid="franka_fr3",
    controller_uid="pd_joint_pos",
    target_object="graspnet1b:63",
    scene_uid="hssd:scene3",
    sim_mode="mujoco_isaac",
    headless=True,
    max_episode_steps=6000,
)

# Wrap it to dump a video (and per-frame PNGs) of every episode
env = VideoRecorder(env=env, video_folder="data/output/test_env", write_png=True)
```

There are a few import parameters here:

+ `"simple/FrankaTabletopGraspMP-v0"`, the first positional argument, is the env id. All the built-in env ids can be listed by running

    ```bash
     python scripts/list_env.py
    ```

+ `task`=`franka_tabletop_grasp_mp`, pass the task uid here. Each env id already registers a default task uid, so you only need to pass this when you want to override it.

+ `robot_uid`=`franka_fr3`, pass the robot uid here. Currently registered: `franka_fr3`, `aloha`, `vega_1`, `g1`, `g1_inspire`, `g1_wholebody`, `g1_inspire_wholebody`, `g1_sonic`.

+ `controller_uid`=`pd_joint_pos`, choose the controller method, currently supported `pd_joint_pos`, `pd_delta_eef` ... [TODO]

+ `max_episode_steps`=`6000`. Maximum steps allowed for each episode.

+ `headless`=`[True|False]`. If set to false, `IsaacSim` 's GUI will show.

+ `sim_mode`=`mujoco_isaac`. Available choices: `[mujoco|isaac|mujoco_isaac]`

+ `target_object`=`graspnet1b:63`, This is a task-specific parameter. In this case the target object's asset uid to grasp.

+ `scene_uid`=`hssd:scene3`, the background scene to load.


Main loop
```python
observation, info = env.reset()

episode_over = False
# When the Isaac GUI is up (headless=False) its event loop has to be pumped by
# hand, otherwise the window is frozen and its close button does nothing.
sim_app = env.unwrapped.simulation_app if not env.unwrapped.headless else None
while not episode_over:
    if sim_app is not None:
        if not sim_app.is_running():
            print("Isaac viewer closed; ending episode.")
            break
        sim_app.update()

    # sample a random action -- env.step() takes an ActionCmd, not a raw array,
    # so use the robot helper instead of env.action_space.sample()
    action = env.unwrapped.task.robot.random_action()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
```
