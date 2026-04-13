"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
import sys

import typer
from typing_extensions import Annotated
# from simple.args import args
import copy
import uuid
import pickle
import envlogger
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mujoco
from simple.tasks.registry import TaskRegistry
from simple.engines.mujoco import MujocoSimulator
from simple.agents.primitive_agent import PrimitiveAgent
from simple.envs.video_writer import VideoWriter
from simple.core.action import ActionCmd

import tensorflow as tf
import tensorflow_datasets as tfds
from envlogger.backends import tfds_backend_writer
import transforms3d as t3d

from enum import IntEnum
class TaskState(IntEnum):
    """ ActionConverterMujoco.convert_plan_to_action """
    rest = 0
    approach = 1
    grasp = 2
    lift = 3
    wait = 4


import json
import numpy as np
from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        return JSONEncoder.default(self, obj)

import dm_env
from dm_env import specs



class PickupControlEnv(dm_env.Environment):
    """ an dm_env for controlling only """
    def __init__(self, physics: MujocoSimulator):
        # assert len(physics.get_rooms()) == 1, "parallelization not supported"
        # self.room: MujocoRoom = physics.get_rooms()[0]
        # self.physics: MujocoEngine = physics
        self.mujoco_env = physics
        
        self.target_object_name = None
        self.episode_meta = {}
        
        # is_single_arm = self.mujoco_env.task.robot.is_single_arm
        # coefficient = 1 if is_single_arm else 2
        self.joint_num = self.mujoco_env.task.robot.dof #+ 2 * coefficient

    def reset(self) -> dm_env.TimeStep:
        """Returns the first `TimeStep` of a new episode."""
        return dm_env.restart(self._observation())
    def step(self, action: dict) -> dm_env.TimeStep:
        """Updates the environment according to the action."""
        if action["state"] == int(TaskState.wait): 
            assert self.target_object_name is not None, "please call set_episode_meta() first"
            success = check_success(self.target_object_name, self.mujoco_env)
            self.episode_meta["invalid"] = not success
            reward = 1. if success else 0.
            return dm_env.termination(reward=reward, observation=self._observation())

        return dm_env.transition(reward=0., observation=self._observation(), discount=1.)

    def observation_spec(self):
        """Returns the observation spec."""
        return {
            "robot_qpos": specs.Array(shape=(self.joint_num,), dtype=np.float32), 
            "object_poses": specs.Array(shape=(10, 7), dtype=np.float32) # maximum 10 objects
        }
    
    def action_spec(self):
        return {
            'state': specs.Array(shape=(), dtype=np.int32)
        }
    
    def _observation(self) -> np.ndarray:
        qpos = self.mujoco_env.mjData.qpos.copy()#qpos consists of robot_qpos and object_poses
        
        robot_qpos = qpos[:self.joint_num].astype(np.float32)
        object_poses = np.zeros((10, 7), dtype=np.float32)
        N = (len(qpos) - self.joint_num) // 7
        object_poses[:N] = qpos[self.joint_num:].reshape((-1,7)) # [N,7]
        return {
            "robot_qpos": robot_qpos,
            "object_poses": object_poses
        }
    
    def set_episode_meta(self, env_cfg, render_hz, uuid_str):
        meta = {
            "environment_config": json.dumps(env_cfg, cls=NumpyArrayEncoder),
            "render_hz": np.int64(render_hz),
            "uuid": uuid_str,
            "invalid": True,
        }
        self.target_object_name = env_cfg["dr_state_dict"]["target"]["label"]
        self.episode_meta = meta

    def set_episode_invalid(self, invalid):
        self.episode_meta["invalid"] = invalid
    
def make_ds_config(robot):
    # is_single_arm = robot.is_single_arm
    # coefficient = 1 if is_single_arm else 2
    joint_num = robot.dof #+ 2 * coefficient

    return tfds.rlds.rlds_base.DatasetConfig(
        name='graspnet_pickup',
        observation_info=tfds.features.FeaturesDict({
            "robot_qpos": tfds.features.Tensor(shape=(joint_num,), dtype=np.float32),
            "object_poses": tfds.features.Tensor(shape=(10, 7), dtype=np.float32)
        }),
        action_info=tfds.features.FeaturesDict({
            "state": tfds.features.Tensor(shape=(), dtype=np.int32),
        }),
        reward_info=tfds.features.Tensor(shape=(), dtype=tf.float64),
        discount_info=tfds.features.Tensor(shape=(), dtype=tf.float64),
        episode_metadata_info=tfds.features.FeaturesDict({
            'invalid': tfds.features.Tensor(shape=(), dtype=tf.bool),
            'render_hz': tfds.features.Tensor(shape=(), dtype=np.int64),
            'environment_config': tfds.features.Tensor(shape=(), dtype=tf.string),
            'uuid': tfds.features.Tensor(shape=(), dtype=tf.string),
        }),
    )

def get_episode_metadata(_timestep, _unused_action, pickup_env: PickupControlEnv):
    return pickup_env.episode_meta
def safe_name(name):
    return name if isinstance(name, str) else ""

#TODO need to change
def check_success(target_object_name: str, mujoco_env: MujocoSimulator): # physics_room: MujocoRoom
    """ check if target object was lifted above the table by Franka Panda """
    mj_physics_data = mujoco_env.mjData
    mj_physics_model = mujoco_env.mjModel
    contacted_table_target = False
    contacted_panda_target = False
    all_contacts = []
    for i_contact in range(mj_physics_data.ncon):
        g1=mj_physics_model.geom(mj_physics_data.contact[i_contact].geom1)
        g2=mj_physics_model.geom(mj_physics_data.contact[i_contact].geom2)
        body1=mj_physics_model.body(g1.bodyid).name
        body2=mj_physics_model.body(g2.bodyid).name

        # all_contacts.append((name_geom_1, name_geom_2))
        if "table" in body1 and target_object_name in body2 or \
            "table" in body2 and target_object_name in body1: # or the other way around
            contacted_table_target = True
        robot=mujoco_env.task.robot
        # if robot.uid=="aloha":
        #     string="aloha"
        # elif "franka" in robot.uid:
        #     string="panda"

        string="finger"
        if string in body1 and target_object_name in body2 or \
            string in body2 and target_object_name in body1:
            contacted_panda_target = True
    success = not contacted_table_target and contacted_panda_target
    return success

# def convert_env_config_to_state_dict (env_config):
#     state_dict={
#         "uid": "franka_tabletop_grasp",
#         "dr_state_dict":{}
#     }
#     # state_dict["state_dict"]["uid"] = "franka_tabletop_grasp"
#     dr_state_dict = state_dict['dr_state_dict']
#     # language instruction
#     dr_state_dict["language"] = "Pick up {}."

#     #for target
#     target_info = dr_state_dict["target"]={}
#     target_info['res_id'] = 'graspnet1b'
#     target_info['uid'] = int(env_config['target_info']['id'])

#     #for distractors
#     distractors_info = dr_state_dict["distractors"]={}

#     for obj_info in env_config["object_info"]:
#         obj_id = obj_info["id"]
#         res_id = "graspnet1b"

#         is_target = obj_info["bTarget"]
#         if not is_target:
#             distractors_info[obj_id] = {
#                 "res_id": res_id,
#                 "uid": int(obj_id)
#             }

#     #for spatial
#     spatials_info = dr_state_dict["spatial"]={}
#     for obj_info in env_config["object_info"]:
#         obj_id = str(obj_info["id"])
#         position = obj_info["position"]
#         quaternion = obj_info["orientation"]
#         spatials_info[obj_id] = {
#             "position": position,
#             "quaternion": quaternion
#         }

#     return state_dict

class ControlAgent(PrimitiveAgent):
    
    def __init__(self, robot):
        super().__init__(robot)
    
    def load_plan(self, plan_data):
        approach_traj = plan_data[0][0]
        lift_traj = plan_data[0][1]
        
        # 1. rest phase
        for _ in range(3): # repeat to ensure gripper is open
            self.queue_action(ActionCmd("open_eef", state=TaskState.rest))
        
        # 2. Approach phase (open gripper)
        for qpos in approach_traj:
            if isinstance(qpos, np.ndarray):
                joint_names = list(self.robot.joints.keys())[:len(qpos)]
                target_qpos = dict(zip(joint_names, qpos))
            else:
                target_qpos = qpos
            
            # self.queue_move_qpos_with_eef(target_qpos, "open_eef")
            self.queue_action(ActionCmd(
                "move_qpos", 
                target_qpos=target_qpos,
                state=TaskState.approach
            ))
        
        # 3. Grasp (close gripper)
        for _ in range(3): # repeat to ensure gripper is closed
            # self.queue_close_gripper()
            self.queue_action(ActionCmd("close_eef", state=TaskState.grasp))
        
        # 4. Lift phase (gripper closed)
        for qpos in lift_traj:
            if isinstance(qpos, np.ndarray):
                joint_names = list(self.robot.joints.keys())[:len(qpos)]
                target_qpos = dict(zip(joint_names, qpos))
            else:
                target_qpos = qpos
            
            # self.queue_move_qpos_with_eef(target_qpos, "close_eef")
            self.queue_action(ActionCmd(
                "move_qpos", 
                target_qpos=target_qpos,
                state=TaskState.lift
            ))
        
        # 5. Wait phase (gripper closed)
        self.queue_action(ActionCmd("close_eef", state=TaskState.wait))

def main(
    task: Annotated[str, typer.Option()] = "franka_tabletop_grasp",
    save_label: Annotated[str, typer.Option()] = "plan",
    save_dir: Annotated[str, typer.Option()] = "data/output",
    render_hz: Annotated[int, typer.Option()] = 30,
    dr_level: Annotated[int, typer.Option()] = 3,
    num_shards: Annotated[int, typer.Option()] = 0,
    shard_idx: Annotated[int, typer.Option()] = -1,
    target_id: Annotated[int, typer.Option()] = 63,
):
    save_video = True
    split_group = "train"
    max_episodes_per_shard = 1000
    num_cameras = 1

    episode_save_path = f"{save_dir}/episodes/lv{dr_level}/{target_id}"

    #auto sharding
    if num_shards > 0:
        total_episodes = 37312 # FIXME
        assert shard_idx >= 0 
        take = total_episodes // num_shards
        skip = shard_idx * (total_episodes // num_shards)
        print(f"skip:{skip}")
        print(f"take:{take}")
        episode_save_path = f"{episode_save_path}-skip-{skip}-take-{take}"
        num_iters = take
        save_label = f"{save_label}-skip-{skip}-take-{take}"

    # if os.path.exists(episode_save_path) and len(os.listdir(episode_save_path)) > 0:
    #     raise RuntimeError("Exit now: risk of ovewriting! target folder is not empty!")
    # else:
    #     os.makedirs(episode_save_path, exist_ok=True)

    if not os.path.exists(episode_save_path):
        os.makedirs(episode_save_path, exist_ok=True)

    #task
    # target_object = f"graspnet1b:{target_id:02d}"
    # task = TaskRegistry.make(task, target_object=target_object, render_hz=render_hz, dr_level=dr_level)
    task = TaskRegistry.make(task, render_hz=render_hz, dr_level=dr_level)
    #mujoco
    mujoco=MujocoSimulator(task)
    
    #robot
    robot=task.robot
    control_agent = ControlAgent(robot)

    # create our backend writer for env_logger
    backend_writer = tfds_backend_writer.TFDSBackendWriter(
        data_directory=episode_save_path,
        split_name=split_group,
        max_episodes_per_file=max_episodes_per_shard,
        ds_config=make_ds_config(robot=robot))

    """ physics_room = physics_engine.get_rooms()[0] """
    pickup_env = PickupControlEnv(mujoco)
    env = envlogger.EnvLogger(
        pickup_env,
        backend=backend_writer,
        episode_fn=get_episode_metadata,
    )
    pathlist = sorted(Path(f"{save_dir}/plan/lv{dr_level}/{target_id}").glob('**/*.pkl'))
    stat = [0, 0] # success, total
    pbar = tqdm(pathlist, desc="Iterate plans (0/0)")

    for fpath in pbar:
        with fpath.open("rb") as f:
            plan = pickle.load(f)
        
        # init_robot_pos = plan[0][0][0] #first frame of approach traj
        state_dict = plan[1]

        task.reset(options={"state_dict": state_dict})
        mujoco.update_layout()

        task_id = f"{fpath.parent.name}_{fpath.stem}" # format: {object index}_r{round index}
        uuid_str = f"{task_id}_{str(uuid.uuid4())}"

        control_agent.reset()
        control_agent.load_plan(plan)

        if save_video:
            mujoco_video_writer = VideoWriter(f"{episode_save_path}/videos/{task_id}.mp4", 10, [640, 360], write_png=False)
        
        pickup_env.set_episode_meta(state_dict, render_hz, uuid_str) #TODO

        
        timestep = env.reset() # reset episode
        while not timestep.last() and len(control_agent._action_queue) != 0:
            action_cmd = control_agent.get_action(None)
            
            task_state = action_cmd.parameters.get('state')
            
            mujoco.apply_action(action_cmd)
            frame_mujoco = mujoco.step()[mujoco.default_camera_name]

            if save_video:
                mujoco_video_writer.write(frame_mujoco)
            
            timestep = env.step({
                "state": np.int32(task_state)
            })  
        stat[0] += (1 if timestep.reward > 0 else 0)
        stat[1] += 1
        pbar.set_description(f"Iterate plans ({stat[0]}/{stat[1]})")
        if save_video:
            mujoco_video_writer.release(timestep.reward > 0)
        mujoco.close()
        # break
    env.close()



    





if __name__ == "__main__":
    typer.run(main)


