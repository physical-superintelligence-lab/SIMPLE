import typer
from typing_extensions import Annotated
import gymnasium as gym
import simple.envs as _# import all envs
from simple.mp.curobo import CuRoboPlanner
from simple.envs.wrappers import VideoRecorder
from simple.agents.vlm_agent import ActiveVLMAgent
from simple.tasks.aloha_tabletop_find_n_grasp_mp import AlohaTabletopFindNGraspTask
from simple.tasks.vega_tabletop_find_n_grasp_mp import VegaTabletopFindNGraspTask
from simple.utils import dump_json
import itertools
from tqdm import tqdm
from PIL import Image

import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
import sys
import json

# PRERESET_TASK_STATE = "examples/demo_task_state_dict.json"
spinner = itertools.cycle(['|', '/', '-', '\\'])

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/AlohaTabletopFindNGraspTask-v0",
    task_state_dict: Annotated[str, typer.Option()] = "examples/task1_state_dict.json",
    scene_uid: Annotated[str, typer.Option()] = "hssd:scene3",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 1000,
    render_hz: Annotated[int, typer.Option()] = 30,
    save_dir: Annotated[str, typer.Option()] = "data/examples",
):
    # read task state dict (Scene and object configurations)
    if os.path.exists(task_state_dict):
        print ("Loading existing task state dict from ", task_state_dict)
        # load task state dict if exists
        state_dict = json.load(open(task_state_dict, "r"))
        scene_uid = state_dict["dr_state_dict"]["scene"]["uid"]
        render_hz = state_dict["metadata"]["render_hz"]
    else:
        # randomly reset the episode
        state_dict = None
    

    # create environment
    env = gym.make(
        env_id,
        scene_uid=scene_uid,
        sim_mode=sim_mode,
        headless=headless,
        max_episode_steps=max_episode_steps,
        render_hz=render_hz,
    )
    task = env.unwrapped.task  # type: ignore

    # create motion planner
    render_hz = task.metadata["render_hz"]
    planner = CuRoboPlanner(
        robot=task.robot,
        plan_dt=1.0/render_hz,
        plan_batch_size=40,
    )
    # create motion-planner based agent to solve the task
    vlm_agent = ActiveVLMAgent(task, planner, 
        os.environ.get("OPENAI_API_KEY", ""),
        "0", 
        max_corrections=5
    )

    # Wrap a video recorder for visualization
    env = VideoRecorder(env=env, video_folder=f"{save_dir}", framerate=render_hz, write_png=False)

    # print( "task_state_dict:", state_dict[task.dr_state_dict["distractors"]])
    # print(state_dict)
    # exit()
    observation, info = env.reset(options={"state_dict": state_dict})

    vlm_agent.reset()
    step_idx = 0
    dir = None

       # dump the task state dict if not exists
    if not os.path.exists(task_state_dict):
        state_dict = task.state_dict()
        dump_json(state_dict, open(task_state_dict, "w"), indent=4)

    # loop through the episode
    with tqdm(total=0, bar_format='{desc}', ncols=80, file=sys.stdout) as t:
        episode_over = False

        camera_spinner = itertools.cycle(['rotate_left', 'rotate_right', 'rotate_up', 'rotate_down', 'roll_clockwise', 'roll_counterclockwise'])
        while not episode_over:
            if step_idx == 0:
                Image.fromarray(observation["wrist"]).save(f"cam_init.png")

            try:
                action = vlm_agent.get_action(observation, info)
            except StopIteration:
                # whenever runs outs of action, we queue the next camera movements
                if dir is not None:
                    # we save the resulting camera image after the last camera movement
                    Image.fromarray(observation["wrist"]).save(f"cam_{dir}_at_{step_idx}.png")
                
                dir = next(camera_spinner)
                # vlm_agent.move_camera(dir, distance=1.0, step=0.01)
                vlm_agent.move_to_object("0", info)
                action = vlm_agent.get_action(observation, info)

            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            t.set_description_str(f"Processing {next(spinner)}")
            sys.stdout.flush()
            step_idx += 1

    # close the environment and finalize data saving
    env.close()

if __name__ == "__main__":
    typer.run(main)