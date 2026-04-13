# Data Collection

Import all built-in environments
```
import typer
from typing_extensions import Annotated
import gymnasium as gym
import simple.envs as _# import all envs
from simple.mp.curobo import CuRoboPlanner
from simple.agents.mp import MotionPlannerAgent
from simple.tasks.aloha_tabletop_grasp import AlohaTabletopGraspTask
# suppress typer trachback
import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
```

Pass-in params
```
def main(
    env_id: Annotated[str, typer.Argument()] = "simple/FrankaTabletopGrasp-v0",
    scene_uid: Annotated[str, typer.Option()] = "hssd:scene1",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 200,
    render_hz: Annotated[int, typer.Option()] = 30,
    data_format: Annotated[str, typer.Option()] = "lerobot",
    save_dir: Annotated[str, typer.Option()] = "data/datagen",
    shard_size: Annotated[int, typer.Option()] = 100,
):
````

create environment
```
    env = gym.make(
        env_id,
        scene_uid=scene_uid,
        sim_mode=sim_mode,
        headless=headless,
        max_episode_steps=max_episode_steps,
        render_hz=render_hz,
    )
    task: AlohaTabletopGraspTask = env.unwrapped.task  # type: ignore
```

create motion planner
```
    render_hz = task.metadata["render_hz"]
    planner = CuRoboPlanner(
        robot=task.robot,
        plan_dt=1.0/render_hz,
        plan_batch_size=40,
    )
    # create motion-planner based agent to solve the task
    mp_agent = MotionPlannerAgent(task, planner)
```

create data recorder wrapper, depending on data format

in this example we only support lerobot format

```    
    if data_format == "lerobot":
        from simple.envs.lerobot import LerobotRecorder
        env = LerobotRecorder(env=env, agent=mp_agent, shard_size=shard_size, root_dir=save_dir)
    else:
        raise NotImplementedError
```    
reset the episode, plan the motion for the episode
```
    observation, info = env.reset()
    mp_agent.plan()
```

loop through the episode
```
    episode_over = False
    while not episode_over:
        try:
            action = mp_agent.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
        except Exception as e:
            print(f"Error during episode execution: {e}")
            break
```

close the environment and finalize data saving
```
    env.close()
```
