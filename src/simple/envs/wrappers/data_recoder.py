import envlogger
import os
import shutil
import gymnasium as gym
from simple.envs.video_writer import VideoWriter
import numpy as np
from enum import IntEnum
class TaskState(IntEnum):
    """ ActionConverterMujoco.convert_plan_to_action """
    rest = 0
    approach = 1
    grasp = 2
    lift = 3
    wait = 4
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1

class DataRecorder(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        datarocord_env: envlogger.EnvLogger,
        video_folder:str = "video",
        write_png:bool = False,
        pickup_env=None,
        render_hz: int = 10,
    ):
        # gym.utils.RecordConstructorArgs.__init__(
        #     self, data_env=datarocord_env,video_folder=video_folder, write_png=write_png, task_id=task_id
        # )
        gym.Wrapper.__init__(self, env)
        os.makedirs(video_folder, exist_ok=True)
        self.env = env
        self.data_env = datarocord_env
        self.work_dir = video_folder
        self.write_png = write_png
        self.pickup_env = pickup_env
        self.render_hz = render_hz
    def update(self, task_id, env_config):
        self.task_id = task_id
        self.env_config = env_config
    def reset(self, **kwargs):
        observations, info = super().reset(**kwargs)
        
        self.isaac_video_writer_left = VideoWriter(f"{self.work_dir}/videos/{self.task_id}_left.mp4", 10, [640,360], write_png=self.write_png)
        self.isaac_video_writer_right = VideoWriter(f"{self.work_dir}/videos/{self.task_id}_right.mp4", 10, [640, 360], write_png=self.write_png)
        self.isaac_video_writer_wrist = VideoWriter(f"{self.work_dir}/videos/{self.task_id}_wrist.mp4", 10, [480, 270], write_png=self.write_png)
        self.isaac_video_writer_side = VideoWriter(f"{self.work_dir}/videos/{self.task_id}_side.mp4", 10, [640, 360], write_png=self.write_png)

        self.isaac_video_writer_left.write(observations["front_stereo_left"])
        self.isaac_video_writer_right.write(observations["front_stereo_right"])
        self.isaac_video_writer_wrist.write(observations["wrist"])
        self.isaac_video_writer_side.write(observations["side_left"])


        scene_info=self.env.unwrapped.task.layout.scene_info
        self.env_config["scene_info"] = scene_info
        self.pickup_env.set_episode_meta(env_cfg=self.env_config, render_hz=self.render_hz, uuid_str=self.task_id)
        time_step = self.data_env.reset()

        return observations, info, time_step
    
    def render(self):
        pass

    def step(self, action,state):
        observations, reward, terminated, truncated, info = super().step(action)

        self.isaac_video_writer_left.write(observations["front_stereo_left"])
        self.isaac_video_writer_right.write(observations["front_stereo_right"])
        self.isaac_video_writer_wrist.write(observations["wrist"])
        self.isaac_video_writer_side.write(observations["side_left"])

        step={"state":np.int32(int(TaskState[state]))}
        gripper_close = GRIPPER_CLOSE if TaskState[state] in [TaskState.grasp, TaskState.lift, TaskState.wait] else GRIPPER_OPEN
        step={"state":np.int32(int(TaskState[state]))
              ,"gripper":[gripper_close]}
        self.pickup_env.cheat(step)
        timestep = self.data_env.step(step)

        return observations, reward, terminated, truncated, info, timestep


    def release(self):
        self.isaac_video_writer_left.release()
        self.isaac_video_writer_right.release()
        self.isaac_video_writer_wrist.release()
        self.isaac_video_writer_side.release()
        
    def close(self):
        # self.isaac_video_writer_left.release()
        # self.isaac_video_writer_right.release()
        # self.isaac_video_writer_wrist.release()
        # self.isaac_video_writer_side.release()
        self.data_env.close()
        super().close()
