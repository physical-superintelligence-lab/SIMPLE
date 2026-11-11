# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# from simple.args import args
import os
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for offscreen rendering
import gymnasium as gym
import simple.envs # import all envs
import numpy as np
from simple.envs.wrappers import VideoRecorder
import typer
import json
from typing import Optional
from typing_extensions import Annotated
from simple.core.action import ActionCmd

PRERESET_TASK_STATE = "examples/demo_task_state_dict.json"


def look_at_robot(robot_pose, distance: float, height: float = 0.8):
    """Point Isaac's GUI viewport at the robot from `distance` metres in front of it."""
    from isaacsim.core.utils.viewports import set_camera_view

    position = np.asarray(robot_pose.position, dtype=float)
    w, x, y, z = robot_pose.quaternion
    # The robot base's +x axis, flattened onto the ground plane.
    forward = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 0.0])
    forward /= np.linalg.norm(forward)
    eye = position + distance * forward + [0.0, 0.0, height]
    target = position + 0.3 * forward + [0.0, 0.0, 0.5]
    set_camera_view(eye=eye, target=target)
    print(f"viewer camera: eye={eye.round(2)} target={target.round(2)}")

def main(
    env_id: str = "simple/FrankaTabletopGraspMP-v0",
    task: str = "franka_tabletop_grasp_mp",
    robot_uid: str = "franka_fr3",
    controller_uid: str = "pd_joint_pos",
    scene_uid: str = "hssd:scene3",
    target_object: str = "graspnet1b:63",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = True,
    webrtc: Annotated[
        Optional[bool], typer.Option(help="WebRTC streaming. Default: on when headless.")
    ] = None,
    max_episode_steps: Annotated[int, typer.Option()] = 6000,
    save_dir: Annotated[str, typer.Option()] = "data/output",
    record: Annotated[bool, typer.Option(help="Write a video under SAVE_DIR/test_env.")] = False,
    viewer_distance: Annotated[
        float, typer.Option(help="GUI camera distance in front of the robot, in metres.")
    ] = 1.5,
):
    if webrtc is None:
        webrtc = headless
    print(f"headless={headless} webrtc={webrtc} DISPLAY={os.environ.get('DISPLAY')!r}")

    env = gym.make(
        env_id,
        task=task,
        robot_uid=robot_uid,
        controller_uid=controller_uid, 
        target_object=target_object,
        scene_uid=scene_uid,
        sim_mode=sim_mode,
        headless=headless,
        webrtc=webrtc,
        max_episode_steps=max_episode_steps,
    )

    if record:
        output_eval_dir = f"{save_dir}/test_env"
        os.makedirs(output_eval_dir, exist_ok=True)
        env = VideoRecorder(env=env, video_folder=f"{output_eval_dir}", write_png=True)

    if os.path.exists(PRERESET_TASK_STATE):
        # load task state dict if exists
        state_dict = json.load(open(PRERESET_TASK_STATE, "r"))
    else:
        # randomly reset the episode
        state_dict = None
    #for debug
    state_dict = None
    observation, info = env.reset(options={"state_dict": state_dict})
    episode_over = False
    sim_app = env.unwrapped.simulation_app if not headless else None
    if sim_app is not None:
        look_at_robot(env.unwrapped.task.layout.actors["robot"].pose, viewer_distance)
    while not episode_over:
        # Pump the Kit event loop.  Without this the viewer never processes its
        # own events, so the window is frozen and its close button does nothing
        # (Ctrl+C still works -- that signal is handled here, not by Kit).
        if sim_app is not None:
            if not sim_app.is_running():
                print("Isaac viewer closed; ending episode.")
                break
            sim_app.update()

        # sample random actions 
        action = env.unwrapped.task.robot.random_action()

        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
    env.close()

   

if __name__ == "__main__":
    typer.run(main)
