# Run an Environment
Script Usage:
```
python scripts/test_env.py --help
```
Run with default parameters:
```
python scripts/test_env.py
```
By default, a video result will be recoded under `./output/test`.
You can open this folder to check the simulation results.

If you want to video be played using a `libx264` compatitable player. e.g., `VSCode`, please install `ffmpeg`
```
sudo apt-get install ffmpeg
```
> Tips: You might install `ffmpeg` to have video generated in `libx264` format, so that the video can be directly previewed in `VSCode`.

## Detailed Explanations

Common imports:

```
# Include this line to parse command line args
from simple.args import args

# Import all the built-in environments
import simple.envs 

# Import a wrapper for recording simulation videos
from simple.envs.wrappers import VideoRecorder
```

Create a [Gym-style](https://gymnasium.farama.org/) environment:
```
# Create a built-in environment
env = gym.make(
    "simple/FrankaTabletopGrasp-v0",
    task="franka_tabletop_grasp",
    robot_uid="franka_fr3",
    controller_uid="pd_joint_pos", 
    target_object="graspnet1b:63",
    sim_mode=args.sim_mode,
    max_episode_steps=args.max_episode_steps, 
    headless=args.headless,
)
```

There are a few import parameters here:

+ `task`=`simple/FrankaTabletopGrasp-v0`, pass the task uid here. All the built-in tasks can be listed by running
  
    ```
     python scripts/list_env.py
    ```

+ `robot_uid`=`franka_fr3`, pass the robot uid here. We currently support Aloha, Franka panda, research3, a finger extended panda ... [TODO]

+ `controller_uid`=`pd_joint_pos`, choose the controller method, currently supported `pd_joint_pos`, `pd_delta_eef` ... [TODO]

+ `max_episode_steps`=`args.max_episode_steps`. Maximum steps allowed for each episode.

+ `headless`=`[True|False]`. If set to false, `IsaacSim` 's GUI will show.

+ `sim_mode`=`args.sim_mode`. Available choices: `[mujoco|isaac|mujoco_isaac]`

+ `target_object`=`graspnet1b:63`, This is a task-specific parameter. In this case the target object's [asset uid]() to grasp. 


Main loop
```
observation, info = env.reset()

frames = []
episode_over = False
while not episode_over:
    # sample random actions 
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
```