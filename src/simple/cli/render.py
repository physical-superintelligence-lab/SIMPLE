"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from simple.args import args
from simple.engines.isaac_app import create_simulation_app
import typer
from typing_extensions import Annotated
import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = create_simulation_app(SimulationApp, headless=False)

import cv2
import json
from tqdm import tqdm
import transforms3d as t3d
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import envlogger
from envlogger.backends import tfds_backend_writer
import dm_env
from dm_env import specs

tf.config.set_visible_devices([], "GPU")


from simple.cli.control import TaskState,NumpyArrayEncoder
from simple.tasks.registry import TaskRegistry
from simple.engines.mujoco import MujocoSimulator
from simple.engines.isaacsim import IsaacSimSimulator
from simple.envs.video_writer import VideoWriter
from simple.datasets.rlds import convert_env_config_to_state_dict

def water_mark_idx(img, idx):
    h, w = (100,100) # img.shape[:2]
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100,100)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 2
    lineType               = 2

    cv2.putText(img, f'{idx}', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    return img

def quaternion_distance(q1, q2):
    """
    Calculate the angular distance between two quaternions.

    Parameters:
        q1 (array-like): First quaternion [w, x, y, z].
        q2 (array-like): Second quaternion [w, x, y, z].

    Returns:
        float: Angular distance in radians.
    """
    # Normalize the quaternions to ensure they represent valid rotations
    q1 = np.array(q1) / np.linalg.norm(q1)
    q2 = np.array(q2) / np.linalg.norm(q2)

    # Compute the dot product of the quaternions
    dot_product = np.dot(q1, q2)

    # Clamp the dot product to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # The angular distance is 2 * arccos(|dot_product|)
    angular_distance = 2 * np.arccos(np.abs(dot_product))

    return angular_distance

GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1

class PickupRenderEnv(dm_env.Environment):
    def __init__(self, room: IsaacSimSimulator, camera_resolution: list, save_depth:bool):
        self.room = room
        self.camera_resolution = camera_resolution
        self.save_depth = save_depth
        self._cheat = None
        self._global_idx = 0
        # is_single_arm = self.room.task.robot.is_single_arm
        # if is_single_arm:
        #     coefficient = 1
        # else:
        #     coefficient = 2
        self.joint_num = self.room.task.robot.dof #+2*coefficient

    def reset(self) -> dm_env.TimeStep:
        """Returns the first `TimeStep` of a new episode."""
        
        return dm_env.restart(self._observation())

    def step(self, _action) -> dm_env.TimeStep:
        """Updates the environment according to the action."""
        assert self._cheat is not None, "we rely on cheating"

        reward = self._cheat["reward"].numpy().item()
        discount = self._cheat["discount"].numpy().item()
        if self._cheat["is_last"].numpy():
            return dm_env.termination(reward=reward, observation=self._observation())
    
        return dm_env.transition(reward=reward, observation=self._observation(), discount=discount)

    def observation_spec(self):
        """ an ideal observation spec but was overriden by make_ds_config when wrapped in env_logger """
        return {
            "rgb_base": specs.Array(shape=(*self.camera_resolution, 3), dtype=np.uint8),
            "rgb_wrist": specs.Array(shape=(*self.camera_resolution, 3), dtype=np.uint8),
            "proprio": specs.Array(shape=(self.joint_num,), dtype=np.float32),
            "hindsight_task_state": specs.Array(shape=(), dtype=np.int32), # hindsight_task_state
        }

    def action_spec(self):
        """ an ideal action spec, but it was overriden by make_ds_config when wrapped in env_logger """
        return {
            # 'joint_position': specs.BoundedArray(shape=(9,), dtype=np.float32, minimum=-2*np.pi, maximum=2*np.pi),
            'state': specs.Array(shape=(), dtype=np.int32)

        }
    
    def cheat(self, step):
        """ we already knows what happened """
        self._cheat = step 

    def set_episode_meta(self, env_cfg, render_hz, uuid_str):
        self.episode_meta = {
            "environment_config": json.dumps(env_cfg, cls=NumpyArrayEncoder),
            "render_hz": render_hz,
            "uuid": uuid_str,
            # "score": "",
            "invalid": False # should all be success here
        }

    def _observation(self) -> np.ndarray:
        LEFT_CAM_ONLY = False
        """ copy from data_writer._get_observation() """ 
        ret = {}
        cameras = self.room.cameras
        for name, camera in cameras.items():
            if "left" not in name and LEFT_CAM_ONLY:
                continue

            frame = camera.get_current_frame()
            raw_rgb = frame['rgba'][..., :3].astype(np.uint8)

            """ # DEBUG add water mark
            raw_rgb = water_mark_idx(raw_rgb, self._global_idx) """
            ret[f"rgb_{camera.name}"] = raw_rgb
            if self.save_depth:
                raw_depth = np.expand_dims(frame["distance_to_image_plane"].astype(np.float32), axis=-1)
                ret[f'depth_{camera.name}'] = raw_depth

        self._global_idx += 1
        ret['proprio_joint_positions'] = self.room.robot.get_joint_positions()#aloha 16 franka 9
        ret['proprio_eef_pose'] = get_eef_pose(self.room)
        ret['hindsight_task_state'] = self._cheat["action"]["state"].numpy() if self._cheat is not None else int(TaskState.rest) # careful: intent of next action
        return ret


def make_ds_config(robot):
    # is_single_arm = robot.is_single_arm
    # if is_single_arm:
    #     coefficient = 1
    # else:
    #     coefficient = 2
    joint_num = robot.dof #+2*coefficient
    LEFT_CAM_ONLY = False
    features = {
        "rgb_front_stereo_left": tfds.features.Image(shape=(360, 640, 3), dtype=np.uint8, encoding_format='jpeg'), # FIXME
        "proprio_eef_pose": tfds.features.Tensor(shape=(7,), dtype=np.float32), # pos(3) + quat(4)
        "proprio_joint_positions": tfds.features.Tensor(shape=(joint_num,), dtype=np.float32),
        'hindsight_task_state': tfds.features.Tensor(shape=(), dtype=np.int64), # hindsight task state
    }
    if not LEFT_CAM_ONLY:
        features.update({
            "rgb_front_stereo_right": tfds.features.Image(shape=(360, 640, 3), dtype=np.uint8, encoding_format='jpeg'),
            "rgb_wrist": tfds.features.Image(shape=(270, 480, 3), dtype=np.uint8, encoding_format='jpeg'),
            "rgb_side_left": tfds.features.Image(shape=(360, 640, 3), dtype=np.uint8, encoding_format='jpeg'),
            'rgb_wrist_left':tfds.features.Image(shape=(270, 480, 3), dtype=np.uint8, encoding_format='jpeg'),
        })
    """ if save_depth:
        features.update({
            **{f'depth{view_id}': tfds.features.Image(shape=(*camera_resolution, 1), dtype=np.float32) for view_id in range(num_cameras)},
        }) """

    observation_info=tfds.features.FeaturesDict(features)
    return tfds.rlds.rlds_base.DatasetConfig(
        'FrankaGraspEnv',
        observation_info=observation_info,
        action_info=tfds.features.FeaturesDict({

            'gripper': tfds.features.Tensor(shape=(1,), dtype=np.float32) # gripper open:0/close:1
        }),
        episode_metadata_info=tfds.features.FeaturesDict({
            'invalid': tfds.features.Tensor(shape=(), dtype=tf.bool),
            'render_hz': tfds.features.Tensor(shape=(), dtype=np.int64),
            # the environment config is a complex dict, so we store it as a raw string
            # use literal_eval to convert it back to dict when needed
            'environment_config': tfds.features.Tensor(shape=(), dtype=tf.string),
            'uuid': tfds.features.Tensor(shape=(), dtype=tf.string),
            # 'score': tfds.features.Tensor(shape=(), dtype=tf.string),
        }),
        reward_info=np.float64,
        discount_info=np.float64,
    )

def get_episode_metadata(_timestep, _unused_action, pickup_env: PickupRenderEnv):
    return pickup_env.episode_meta

def get_eef_pose(room: IsaacSimSimulator):
    robot_world_pos, robot_world_quat = room.robot.get_world_pose()
    mat_world_robot = np.eye(4)
    mat_world_robot[:3, 3] = robot_world_pos
    mat_world_robot[:3, :3] = t3d.quaternions.quat2mat(robot_world_quat)

    # robot_eef_mat = self.__get_local_pose(isaac_room.robot_eef_xform, mat_world_robot)
    world_pos, world_ori = room.robot_eef_xform.get_world_pose()
    mat_world = np.eye(4)
    mat_world[:3, 3] = world_pos
    mat_world[:3, :3] = t3d.quaternions.quat2mat(world_ori)
    robot_eef_mat = np.linalg.inv(mat_world_robot) @ mat_world
    robot_eef_pos = robot_eef_mat[:3, 3]
    robot_eef_quat = t3d.quaternions.mat2quat(robot_eef_mat[:3, :3])
    # joint_positions = room.robot.get_joint_positions()    
    return np.concatenate([robot_eef_pos, robot_eef_quat]).astype(np.float32)




def main(    task: Annotated[str, typer.Option()] = "franka_tabletop_grasp",
    save_label: Annotated[str, typer.Option()] = "plan",
    save_dir: Annotated[str, typer.Option()] = "data/output",
    render_hz: Annotated[int, typer.Option()] = 30,
    dr_level: Annotated[int, typer.Option()] = 0,
    num_shards: Annotated[int, typer.Option()] = 0,
    shard_idx: Annotated[int, typer.Option()] = -1,
    shard: Annotated[int, typer.Option()] = 100,
    sub_num_shards: Annotated[int, typer.Option()] = 0,
    sub_shard_idx: Annotated[int, typer.Option()] = -1,
    target_id: Annotated[int, typer.Option()] = 63,
    save_depth: Annotated[bool, typer.Option()] = False,
    split: Annotated[str, typer.Option()] = "train",
    
    ):
    save_video = True
    save_label=save_label
    save_dir=save_dir
    max_episodes_per_shard = shard 
    camera_resolution = [256, 256]
    MINIMUM_WAYPONIT_DISTANCE = 0.07 # in joint space
    episode_save_path = f"{save_dir}/episodes/lv{dr_level}/{target_id}"
    render_save_path = f"{save_dir}/renders/lv{dr_level}/{target_id}"

    if num_shards > 0:
        total_episodes = 37312 # FIXME
        assert shard_idx >= 0 
        take = total_episodes // num_shards
        skip = shard_idx * (total_episodes // num_shards)
        # args.skip = skip
        # args.task = take
        print(f"skip:{skip}")
        print(f"take:{take}")
        # dataset = dataset.skip(skip).take(take)
        render_save_path = f"{render_save_path}-frontstereo+wrist+side-randall-skip-{skip}-take-{take}"
        episode_save_path = f"{episode_save_path}-skip-{skip}-take-{take}"
        num_iters = take
        subfolder = f"{subfolder}-skip-{skip}-take-{take}"

    if sub_num_shards > 0:
        sub_take = take // sub_num_shards
        sub_skip = sub_shard_idx * (take // sub_num_shards)
        render_save_path = f"{render_save_path}-sub-{sub_shard_idx}"
    else:
        sub_take = 0
    
    assert os.path.exists(episode_save_path), f"episode does not exist: {episode_save_path}"
    os.makedirs(render_save_path, exist_ok=True)
    #add task
    # task=task
    target_object = f"graspnet1b:{target_id:02d}"
    task = TaskRegistry.make(task, target_object=target_object, render_hz=render_hz*10, dr_level=dr_level)

    mujoco=MujocoSimulator(task)
    #IsaacSim
    isaac=IsaacSimSimulator(task)
    builder=tfds.builder_from_directories([episode_save_path])
    dataset=builder.as_dataset(split="all")
    if sub_take > 0:
        dataset = dataset.skip(sub_skip).take(sub_take)

    valid_episodes = dataset.filter(lambda episode: not episode["invalid"])
    length_dataset = valid_episodes.reduce(0, lambda x,_: x+1).numpy()
    print("total valid episodes to render: ", length_dataset)
    robot = task.robot
    backend_writer = tfds_backend_writer.TFDSBackendWriter(
        data_directory=render_save_path,
        split_name=split,
        max_episodes_per_file=max_episodes_per_shard,
        ds_config=make_ds_config(robot=robot))
    pickup_env = PickupRenderEnv(isaac, camera_resolution, save_depth)
    with envlogger.EnvLogger(
        pickup_env,
        backend=backend_writer,
        episode_fn=get_episode_metadata,
    ) as env:
        for episode in tqdm(valid_episodes, total=length_dataset, desc="Looping episodes"):
            # env._environment._cheat = None
            state_dict = json.loads(episode['environment_config'].numpy().decode("utf-8"))
            task.reset(options={"state_dict": state_dict})
 
            target_info = state_dict['dr_state_dict']['target']
            distractors_info = state_dict['dr_state_dict']['distractors']
            object_info = {**{target_info["uid"]: target_info}, **distractors_info}
            obj_names = [obj["label"] for obj in object_info.values()]
            
            task_id = episode["uuid"].numpy().decode("utf-8")
            render_hz = episode["render_hz"].numpy()
            num_objects = len(object_info)
            object_id = int(state_dict['dr_state_dict']['target']["uid"])
            task_id = f"{object_id:02d}_{task_id[task_id.index('_p')+1:]}"

            pickup_env.set_episode_meta(state_dict, render_hz, task_id)

            # update mujoco and isaac
            mujoco.update_layout()
            isaac.update_layout()

            isaac_video_writer_left = VideoWriter(f"{render_save_path}/videos/{task_id}_left.mp4", 10, [640, 360], write_png=False)
            isaac_video_writer_right = VideoWriter(f"{render_save_path}/videos/{task_id}_right.mp4", 10, [640, 360], write_png=False)
            isaac_video_writer_wrist = VideoWriter(f"{render_save_path}/videos/{task_id}_wrist.mp4", 10, [480, 270], write_png=False)
            isaac_video_writer_side = VideoWriter(f"{render_save_path}/videos/{task_id}_side.mp4", 10, [640, 360], write_png=False)


            episode_as_list = list(episode["steps"])
            close_gripper_issued = False # see detailed comments below
            close_gripper_observed = False
            close_gripper_frame_idx = -1
            distance_to_last_frame = 0.
            last_action = None
            last_qpos = None

            for step_idx, step in enumerate(tqdm(episode_as_list, desc="Rendering episode", leave=False)):
                # action_joint_positions = step["action"]["joint_positions"].numpy().astype(np.float32)
                # action_eef_pose = step["action"]["eef_pose"].numpy().astype(np.float32)
                action_state = TaskState(step["action"]["state"])
                joint_state = step["observation"]["robot_qpos"].numpy()
                object_poses = step["observation"]["object_poses"].numpy()[:num_objects]
                obj_positions, obj_orientations = np.split(object_poses, [3,], axis=1)
                gripper_close = GRIPPER_CLOSE if action_state in [TaskState.grasp, TaskState.lift, TaskState.wait] else GRIPPER_OPEN

                # """ filtering the trajectory
                #     1. skip static frames
                #     2. skip grasping process
                # """
                # keep_this_frame = False
                # if step_idx == 0:
                #     # alway keep first frame
                #     keep_this_frame = True
                # elif gripper_close == GRIPPER_OPEN and not close_gripper_issued:
                #     # remove very slow frames before closing the gripper 
                #     distance_to_last_frame = np.linalg.norm(last_qpos - joint_state) 
                #     keep_this_frame = distance_to_last_frame > MINIMUM_WAYPONIT_DISTANCE
                # else:
                #     # case: step_idx > 0 and gripper is closed
                #     if not close_gripper_issued:
                #         # keep the first frame to close the gripper
                #         keep_this_frame = True #
                #         close_gripper_issued = True
                #     else: 
                #         # after gripper closed, keep only moving frames, so static grasping frames are removed too
                #         # (actually we keep one static grasping frame)
                #         if not close_gripper_observed:
                #             close_gripper_observed = True
                #             close_gripper_frame_idx = step_idx
                #             keep_this_frame = False
                #         else:
                #             # keep one single frame of closing gripper
                #             # and remove static grasping & lifting frames
                #             distance_to_last_frame = np.linalg.norm(last_qpos - joint_state) 
                #             keep_this_frame = (step_idx == close_gripper_frame_idx + args.render_hz-1) or \
                #                             (distance_to_last_frame > MINIMUM_WAYPONIT_DISTANCE)

                # if not keep_this_frame:
                #     continue

                mujoco.set_robot_qpos(joint_state)
                mujoco.set_object_poses(obj_names, obj_positions, obj_orientations)

                
                mujoco_frame = mujoco.step(render=True)
                frame_issac = isaac.step(mujoco)



                isaac_video_writer_left.write(frame_issac["front_stereo_left"])
                isaac_video_writer_right.write(frame_issac["front_stereo_right"])
                isaac_video_writer_wrist.write(frame_issac["wrist"])
                isaac_video_writer_side.write(frame_issac["side_left"])

                if step_idx == 0:
                    env.reset()
                else:
                    pickup_env.cheat(step)
                    env.step(last_action)
                
                last_action = {"gripper": [gripper_close]} # 1: close, 0: open
                last_qpos = joint_state

                # keep GUI running for debugging purpose
                # if not sim_cfg["headless"]:
                #     while simulation_app.is_running():
                #         simulation_app.update()

            isaac_video_writer_left.release()
            isaac_video_writer_right.release()
            isaac_video_writer_wrist.release()
            isaac_video_writer_side.release()
            # break
    simulation_app.close()
    

def typer_main():
    typer.run(main)

if __name__ == "__main__":
    typer.run(main)
