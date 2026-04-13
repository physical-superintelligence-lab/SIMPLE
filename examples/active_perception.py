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
import imageio
import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
import sys
import json
from dotenv import load_dotenv

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/AlohaTabletopFindNGraspTask-v0",
    task_state_dict: Annotated[str, typer.Option()] = "examples/task1_state_dict.json", # "examples/task2_state_dict.json" for another scene
    scene_uid: Annotated[str, typer.Option()] = "hssd:scene3",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = True,   # use False for non-headless mode
    max_episode_steps: Annotated[int, typer.Option()] = 1000,
    render_hz: Annotated[int, typer.Option()] = 30,
    save_dir: Annotated[str, typer.Option()] = "data/examples",
    max_steps: Annotated[int, typer.Option()] = 20,
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

    # Wrap a video recorder for visualization
    env = VideoRecorder(env=env, video_folder=f"{save_dir}", framerate=render_hz, write_png=False)
    
    # reset the episode
    observation, info = env.reset(options={"state_dict": state_dict})
    side_image = observation["side_left"]
    side_image = Image.fromarray(side_image)
    side_image.save(f"{save_dir}/side_image.png")

    wrist_image = observation["wrist"]

    # Get user input for target object and API key
    target_object = input("Enter the target object: ")
    api_key = input("Enter your OpenAI API key or press ENTER to load key from .env: ")
    if api_key == "":
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            print("Got the api_key from .env")
        except:
            print("Error loading OPENAI_API_KEY from .env")
    
    vlm_agent = ActiveVLMAgent(task, planner, api_key, target_object, max_corrections=5)
    vlm_agent.reset()

    # dump the task state dict if not exists
    if not os.path.exists(task_state_dict):
        state_dict = task.state_dict()
        dump_json(state_dict, open(task_state_dict, "w"), indent=4)

    # loop through the episode
    found_target = False
    wrist_images = [wrist_image]
    step = 0
    while not found_target and step < max_steps:
        try:    # get the action in the queue if exists
            action = vlm_agent.get_action(observation, info)
        except StopIteration:
            # get wrist image
            wrist_image = observation["wrist"]
            # append the wrist image to the list for video recording
            wrist_images.append(wrist_image)
            prev_fail = False
            while True:
                step += 1
                direction, distance = vlm_agent.get_movement_command(wrist_image, prev_fail)
                if direction == "done":
                    found_target = True
                    break
                prev_fail = not vlm_agent.move_camera(direction, distance)
                # check if the suggested movement is valid, prev_fail = True => invalid movement
                if prev_fail:
                    print("❌ Previous movement failed! Trying again...")
                    continue
                action = vlm_agent.get_action(observation, info)
                break
            # step += 1
            if found_target:
                break
        # # Can visualize the wrist images every action to better understand the robot's behavior
        # # get wrist image
        # wrist_image = observation["wrist"]
        # # append the wrist image to the list for video recording
        # wrist_images.append(wrist_image)

        observation, reward, terminated, truncated, info = env.step(action)

    if found_target:
        print("Found target!")
    else:
        print("Did not find target!")
    
    with imageio.get_writer(f"{save_dir}/wrist_images.mp4", fps=10) as writer:
        for wrist_image in wrist_images:
            writer.append_data(wrist_image)


if __name__ == "__main__":
    load_dotenv()
    typer.run(main)