from __future__ import annotations
import os
import json
import numpy as np
import shutil
import gymnasium as gym
import transforms3d as t3d
from typing import Optional, TYPE_CHECKING
# import decord


from lerobot.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
from simple.utils import NumpyArrayEncoder
if TYPE_CHECKING:
    from simple.agents.base_agent import BaseAgent as Agent

from simple.agents.base_agent import TaskState
from simple.robots.protocols import DualArm, Wholebody

# # TODO Femove 
# def get_eef_pose(room):
#     robot_world_pos, robot_world_quat = room.robot.get_world_pose()
#     mat_world_robot = np.eye(4)
#     mat_world_robot[:3, 3] = robot_world_pos
#     mat_world_robot[:3, :3] = t3d.quaternions.quat2mat(robot_world_quat)

#     # robot_eef_mat = self.__get_local_pose(isaac_room.robot_eef_xform, mat_world_robot)
#     world_pos, world_ori = room.robot_eef_xform.get_world_pose()
#     mat_world = np.eye(4)
#     mat_world[:3, 3] = world_pos
#     mat_world[:3, :3] = t3d.quaternions.quat2mat(world_ori)
#     robot_eef_mat = np.linalg.inv(mat_world_robot) @ mat_world
#     robot_eef_pos = robot_eef_mat[:3, 3]
#     robot_eef_quat = t3d.quaternions.mat2quat(robot_eef_mat[:3, :3])
#     # joint_positions = room.robot.get_joint_positions()    
#     return np.concatenate([robot_eef_pos, robot_eef_quat]).astype(np.float32)

GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1

from gymnasium import spaces

def _from_gym_action_space(space: spaces.Space) -> dict:
    if isinstance(space, spaces.Dict):
        action_dim = 0
        for subspace in space.spaces.values():
            if subspace.shape is None:
                raise ValueError("Expected all action subspaces to have concrete shapes")
            dim = 1
            for size in subspace.shape:
                dim *= int(size)
            action_dim += dim

        return {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [key for key in space.keys()],
        }
    elif isinstance(space, spaces.Box):
        if space.shape is None:
            raise ValueError("Expected action space to have a concrete shape")
        return {
            "dtype": space.dtype.name,
            "shape": space.shape,
            "names": ["action"],
        }
    else:
        raise ValueError(f"Unsupported action space type: {type(space)}")

class LerobotRecorder(gym.Wrapper, gym.utils.RecordConstructorArgs):
    
    def __init__(
        self,
        env: gym.Env,
        root_dir:str = "data/datagen",
        shard_size: int = 100,
        agent: Optional[Agent] = None,
        debug: bool = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, root_dir=root_dir, shard_size=shard_size
        )
        gym.Wrapper.__init__(self, env)

        task_name = env.unwrapped.spec.id
        dr_level = env.unwrapped.task.dr.level
        dataset_root_dir = f"{os.path.abspath(root_dir)}/{task_name}/level-{dr_level}"
        
        if debug and os.path.exists(dataset_root_dir):
            shutil.rmtree(dataset_root_dir)
        
        self.task = env.unwrapped.task
        self.robot = env.unwrapped.task.robot
        action_space = self.robot.controller.action_space
        
        if isinstance(action_space, gym.spaces.Dict):
            action_shape = sum([s.shape[0] for s in action_space.values()])
        else:
            action_shape = action_space.shape[0]
        
        if hasattr(self.robot.controller, "eef_action_space"):
            eef_action_space = self.robot.controller.eef_action_space
            if isinstance(eef_action_space, gym.spaces.Dict):
                eef_action_shape = sum([s.shape[0] for s in eef_action_space.values()])
            else:
                eef_action_shape = eef_action_space.shape[0]
        else:
            eef_action_shape = 1
            eef_action_space = None    

        from simple.robots.protocols import Humanoid, Wholebody
        features_dict = {
            "action": _from_gym_action_space(self.task.action_space)
        }

        for key, space in self.task.observation_space.items():
            if len(space.shape) == 3 and space.shape[2] == 3: # image
                features_dict[f"observation.rgb_{key}"] = {
                    "dtype": "video",
                    "shape": space.shape,
                    "names": ["height", "width", "channels"],
                }
            elif len(space.shape) == 1: # vector
                features_dict[f"observation.{key}"] = {
                    "dtype": space.dtype.name,
                    "shape": space.shape,
                    "names": [key],
                }

        if isinstance(self.task.robot, Wholebody):
            features_dict["observation.amo_policy_obs_prop"] = {
                "dtype": "float32",
                "shape": (3 + 2 + 2 + 23 * 3 + 2 + 15,),
                "names": ["amo_policy_obs_prop"],
            }
            features_dict["observation.amo_policy_output_torque"] = {
                "dtype": "float32",
                "shape": (15,),
                "names": ["amo_policy_output_torque"],
            }
            features_dict["observation.amo_policy_command"] = {
                "dtype": "float32",
                "shape": (9,),
                "names": ["amo_policy_command"],
            }
            features_dict["observation.amo_policy_rpy"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": ["amo_policy_rpy"],
            }
            features_dict["observation.amo_policy_turning_flag"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["amo_policy_turning_flag"],
            }
            features_dict["observation.amo_policy_target_yaw"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["amo_policy_target_yaw"],
            }

        self.dataset = LeRobotDataset.create(
            repo_id=task_name,
            root=dataset_root_dir,
            fps=self.task.metadata["render_hz"],
            features=features_dict  
        )

        self.env_cfgs = []
        self.agent = agent
        self.root_dir = dataset_root_dir
        self.episode_index = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.dataset.clear_episode_buffer()
        self.agent.reset()
        self.env_conf = self.task.state_dict() 
        # self.env_cfgs.append(env_conf)
        return obs, info
        
    def clear_episode_buffer(self):
        if self.dataset.episode_buffer is None: 
            return

        for cam_key in self.dataset.meta.camera_keys:
            img_dir = self.dataset._get_image_file_path(
                episode_index=self.dataset.episode_buffer["episode_index"], image_key=cam_key, frame_index=0
            ).parent
            if img_dir.is_dir():
                shutil.rmtree(img_dir)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        gripper_close = GRIPPER_OPEN

        frame = {}
        for key, value in obs.items():
            if value.ndim == 3:  # image HWC
                frame[f"observation.rgb_{key}"] = value
            elif value.ndim == 1:
                frame[f"observation.{key}"] = value

        action_1d = []
        
        from simple.robots.protocols import Humanoid
        if isinstance(self.robot, Humanoid):
            #FOR G1
            for jname in self.robot.joint_names:
                # jvalue = action.parameters["target_qpos"][jname]
                action_value = self.robot.actuators[jname].ctrl.item()
                action_1d.append(action_value)
            if isinstance(self.robot, Wholebody):
                pd_target = self.robot.pd_target.copy()
                action_1d[:15] = pd_target # lower-body joint targets from amo policy
                action_1d[-14:] = list(self.robot.hand_target_qpos().values()) # hand joint targets

        else:
            seen_grippers = set()
            for jname, jvalue in action.parameters["target_qpos"].items():
                if "franka" in self.robot.uid and "finger" in jname: # skip finger joints for franka
                    continue
                if "finger" in jname:
                    gripper_side = jname.split('_')[0]
                    if gripper_side not in seen_grippers:
                        action_1d.append(jvalue)
                        seen_grippers.add(gripper_side)
                    continue
                action_1d.append(jvalue)  
                
        frame["action"] = np.array(action_1d, dtype=np.float32)
        
        # FOR ALOHA
        if isinstance(self.robot, DualArm) and hasattr(self.robot.controller, "eef_action_space"):
            gripper_close = [GRIPPER_OPEN, GRIPPER_OPEN]
            if action.type == "close_eef" or action.parameters.get("eef_state") == "close_eef":
                hand_uid = action.parameters.get("hand_uid")
                if hand_uid is not None:
                    if "left" in hand_uid:
                        gripper_close[0] = GRIPPER_CLOSE
                    if "right" in hand_uid:
                        gripper_close[1] = GRIPPER_CLOSE
            elif action.type == "open_eef" or action.parameters.get("eef_state") == "open_eef":
                hand_uid = action.parameters.get("hand_uid")
                if hand_uid is not None:
                    if "left" in hand_uid:
                        gripper_close[0] = GRIPPER_OPEN
                    if "right" in hand_uid:
                        gripper_close[1] = GRIPPER_OPEN
        else:
            gripper_close = [GRIPPER_OPEN]
            if action.type == "close_eef" or action.parameters.get("eef_state") == "close_eef":
                gripper_close = [GRIPPER_CLOSE]


        #FOR wholebody G1
        if isinstance(self.robot, Wholebody):
            frame["observation.amo_policy_obs_prop"] = self.robot.amo_policy.obs_prop.astype(np.float32)
            frame["observation.amo_policy_output_torque"] = self.robot.mjdata.ctrl[:15].astype(np.float32)
            frame["observation.amo_policy_command"] = self.robot.amo_policy.obs_command.astype(np.float32)
            frame["observation.amo_policy_rpy"] = self.robot.amo_policy.obs_rpy.astype(np.float32)
            frame["observation.amo_policy_turning_flag"] = self.robot.amo_policy.obs_turning_flag.astype(np.float32)
            frame["observation.amo_policy_target_yaw"] = self.robot.amo_policy.obs_target_yaw.astype(np.float32)

        self.dataset.add_frame(frame, task=self.unwrapped.task.instruction)

        if terminated or truncated:
            if float(reward) > 0.9:
                # self.env_cfgs.append(self.env_conf)
                self.dataset.save_episode()
                self.write_env_config(self.env_conf, self.episode_index)
                self.episode_index += 1
                self.dataset.clear_episode_buffer()
            else:
                self.dataset.clear_episode_buffer()
                for cam_key in self.dataset.meta.camera_keys:
                    img_dir = self.dataset._get_image_file_path(
                        episode_index=self.dataset.episode_buffer["episode_index"], image_key=cam_key, frame_index=0
                    ).parent
                    if img_dir.is_dir():
                        shutil.rmtree(img_dir)
                self.env_cfgs.pop() if len(self.env_cfgs) > 0 else None

        return obs, reward, terminated, truncated, info

    def close(self):      
        super().close()

    def write_env_config(self, env_conf, episode_index):
        epsideo_meta_file = f"{self.root_dir}/meta/episodes.jsonl"
        if os.path.exists(epsideo_meta_file):
            with open(epsideo_meta_file, "r") as f:
                lines = [json.loads(line) for line in f]
        lines[episode_index]["environment_config"] = json.dumps(env_conf, cls=NumpyArrayEncoder)
        with open(epsideo_meta_file, "w") as f:
            for entry in lines:
                f.write(json.dumps(entry) + "\n")
                
    
