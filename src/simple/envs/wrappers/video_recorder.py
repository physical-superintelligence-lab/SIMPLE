# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import os
import shutil
import cv2
import numpy as np
import gymnasium as gym
from simple.envs.video_writer import VideoWriter
from datetime import datetime

class VideoRecorder(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        video_folder:str = "video",
        framerate:int = 10,
        # camera: List[str] = ["mujoco", "front_left", "wrist"],
        name_prefix:str|None = None,
        write_png:bool = False,
        video_keys: list[str] | None = None, # image observations to record; None records all
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, video_folder=video_folder, name_prefix=name_prefix, write_png=write_png 
        )
        gym.Wrapper.__init__(self, env)

        os.makedirs(video_folder, exist_ok=True)
        
        # self._elapsed_steps = None
        self.work_dir = video_folder
        
        if name_prefix is None:
            now = datetime.now()
            name_prefix = now.isoformat().replace(":", "-").replace(".", "-")
            if self.unwrapped.__module__.startswith("simple.envs"):
                self.sim_mode = self.unwrapped.sim_mode # type: ignore
                name_prefix = f"{self.unwrapped.task.uid}_{name_prefix}" # type: ignore

        self.name_prefix = name_prefix
        self.write_png = write_png
        self.video_keys = video_keys
        self.framerate = framerate
        self.video_writers = {}
        self._is_released = False

    def reset(self, **kwargs):
        observations, info = super().reset(**kwargs)

        if kwargs.get("options") is not None:
            if  kwargs["options"].get("task_id") is not None:
                self.name_prefix = kwargs["options"]["task_id"] # overwrite name prefix with task_id
                video_folder = f"{self.work_dir}/{self.name_prefix}"
                if os.path.exists(video_folder):
                    shutil.rmtree(video_folder, ignore_errors=True)
                    print(f"Overwriting existing videos at {video_folder} folder")

                os.makedirs(video_folder, exist_ok=True)

        self._init_writers(observations)
        return observations, info

    def _init_writers(self, observation):
        """Open fresh video writers and seed them with `observation` as frame 0.

        Discards any previously buffered frames — the existing mp4 files (if any)
        are deleted by `VideoWriter.__init__`. Call directly to skip warmup or
        stabilization frames: run the env for the warmup steps, then call
        `_init_writers(observation)` so the saved video starts at this point.
        """
        for video_writer in self.video_writers.values():
            try:
                video_writer.video_writer.release()
            except Exception:
                pass
        self.video_writers = {}
        for key, subspace in self.unwrapped.observation_space.items():
            if self.video_keys is not None and key not in self.video_keys:
                continue
            if len(subspace.shape) == 3 and subspace.shape[-1] == 3: # only record image observations
                self.video_writers[key] = VideoWriter(
                    f"{self.work_dir}/{self.name_prefix}/{key}.mp4",
                    self.framerate,
                    subspace.shape[:2][::-1],
                    write_png=self.write_png,
                )
                self.video_writers[key].write(observation[key])
        self._is_released = False

    def render(self):
        pass

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        for key, video_writer in self.video_writers.items():
            video_writer.write(observation[key])

        return observation, reward, terminated, truncated, info
    
    def release(self):
        if not self._is_released:
            for video_writer in self.video_writers.values():
                video_writer.release(self.unwrapped._success) # type: ignore
            self._is_released = True

    def close(self):
        """Closes the wrapper then the video recorder."""
        self.release()
        super().close()
