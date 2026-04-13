"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict,Optional
if TYPE_CHECKING:
    from simple.core.randomizer import RandomizerCfg
from simple.core.task import Task
from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.object import Object
from simple.core.actor import Actor, ObjectActor
from simple.core.layout import Layout
from simple.core.robot import Robot
from simple.core.scene import Scene
# from simple.dr import TargetDR, SpatialDR, DistractorDR, CameraDR, TabletopSceneDR, SceneDR, MaterialDR, LightingDR
from simple.dr import *
from simple.dr.manager import DRManager, TabletopGraspDRManager # , LayoutManager

from simple.dr.types import Box
from simple.robots.registry import RobotRegistry
from simple.tasks.registry import TaskRegistry
from simple.core.actor import ActorReigstry
from simple.robots.protocols import Controllable
from simple.sensors import StereoCameraCfg, SensorCfg, CameraCfg
from copy import deepcopy
from gymnasium import spaces
import numpy as np
from typing import Any
import math
from simple.assets import AssetManager


_LIFT_HEIGHT = 0.05
@TaskRegistry.register("g1_inspire_tabletop_grasp_mp")
class G1InspireTabletopGraspMP(Task):
    uid:str = "g1_inspire_tabletop_grasp_mp"
    label: str = "G1 Inspire Tabletop Grasp Task"
    description: str = "A task where the G1 Inspire robot must grasp target object on a tabletop."

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "dr_level": 0,
        "version": 1.0,
    }

    robot_cfg: dict[str, Any] = dict(
        uid="g1_inspire",
        # controller_uid="pd_delta_ee_pose", 
        # position=[0, 1, 0], 
        # # position=[0.0, 0, 0.0],
        # orientation=[0.7071,0,0,-0.7071], #  w,x,y,z #TODO 
        # orientation=[1.0,0.0,0.0,0.0], #  w,x,y,z #TODO
    )

    sensor_cfgs: dict[str, SensorCfg] = dict(
        front_stereo = StereoCameraCfg(
            uid="Realsense_D415",
            mount="eye_on_base",
            width=640,
            height=360,
            focal_length=1.88,
            fov=np.deg2rad(71.28), 
            near=0.2,
            far=5,
            baseline=0.055,
            pose=dict( 
                # https://en.wikipedia.org/wiki/Spherical_coordinate_system
                # robot is at the origin, camera is orbiting around the robot
                distance=2.5, # radial distance r
                polar=np.deg2rad(60),  # polar angle θ
                azimuth=np.deg2rad(0),  # azimuthal angle φ
            )
        ),
        wrist = CameraCfg(
            uid="Logitech_C930e",
            mount="eye_in_hand",
            width=480,
            height=270,
            focal_length=0.94,
            fov=np.deg2rad(90),
            near=0.1,
            far=5,
            pose=dict(
                position=[0.06, 0.0, 0.05],
                # quaternion=, # FIXME np.array(self.robot.wrist_camera_orientation)
            )
        ),
        wrist_left = CameraCfg(
            uid="Logitech_C930e",
            mount="eye_in_hand",
            width=480,
            height=270,
            focal_length=0.94,
            fov=np.deg2rad(90),
            near=0.1,
            far=5,
            pose=dict(
                position=[0.0, 0.0, 0.05],
            )
        ),
        head_stereo = StereoCameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            width=640,
            height=360,
            focal_length=1.93,
            fov=np.deg2rad(110),
            near=0.2,
            far=5,
            baseline=0.05,
            pose=dict(
                position=[0.0, 0.0, 0.0],
            )
        ),
        side_left = CameraCfg(
            uid="Realsense_D415_mono",
            mount="eye_on_base",
            width=640,
            height=360,
            focal_length=1.88,
            fov=np.deg2rad(71.28),
            near=0.2,
            far=5,
            pose=dict(
                position=[0.35, 0.9, 0.675],
                eulers=np.deg2rad([0, 30, -90]),
            )
        ),
    )


    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language = LanguageDRCfg(
            instructions = [
                "Pick up {}.",
            ]
        ),

        target = TargetDRCfg(
            asset_id="graspnet1b:1" # e.g., "primitive:cube"
        ),

        distractors = DistractorDRCfg(
            res_id="graspnet1b", 
            number_of_distractors=3,
            allow_duplicates=False,
            exclude=["1"]  # Exclude the target object 
        ),

        spatial = SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-0.8, -0.0], high=[-0.8, -0.0]),
            target_region=Box(low=[-0.4, 0.0], high=[-0.4, 0.0]),
            distractors_region=Box(low=[0.3, -0.3], high=[-0.4, 0.3]),
            target_stable_indices = [0],
            target_rotate_z = Box(low=-0.15, high=0.15),
        ),

        camera = CameraDRCfg(
            cam_id="franka_camera",
            # position=[0.5, 0, 0.5], # TODO
            # orientation=[0, 0, 0], # TODO define a range
        ),

        scene = TabletopSceneDRCfg(
            # asset_id="primitive:table",
            # scene_mode="random", # fixed, random
            # table_size=Box(low=[1.4, 1.4, 0.1], high=[1.8, 1.8, 0.2]),  # Table dimensions
            # table_position=Box(low=[0.2, -0.25], high=[0.6, 0.25]),
            # table_height=Box(low=0.0, high=0.0),
            # rotation_z=Box(low=0, high=3.14),  # Rotation around the Z-axis
            table_height=Box(low=0.0, high=0.0),
            room_choices=["hssd:scene0"],
            scene_manager="hssd"
        ),

        lighting = LightingDRCfg(
            light_mode="random", # fixed, random
            light_num=(2,3),
            light_color_temperature=Box(low=6001, high=8001),  # I was not joking :)
            light_intensity=Box(low=5e4, high=5e4),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 2.1),
            light_spacing=Box((1., 1.), (2.5, 2.5)),
            light_position=Box((-1.1, -1.1, 1.3), (1.1, 1.1, 1.5)),
            light_eulers=Box((0,0,-0.5*math.pi), (0,0,0.5*math.pi))
        ),

        material = MaterialDRCfg(
            material_mode="rand_all", # fixed, rand_all, rand_tableground, rand_objects
        )
    )

    def __init__(
        self, 
        robot_uid: str = "g1_inspire",
        scene_uid: str | Scene = "hssd:scene3",
        target_object: str | Object = "graspnet1b:1",
        controller_uid: str = "pd_joint_pos", # pd_joint_vel, pd_ee_pose, pd_delta_ee_pose
        split: str = "train",  # train, val, test
        render_hz: int | None = None,
        dr_level: int = 0,
        physics_dt: float = 0.002, 
        *args,
        **kwargs
    ):
       # lazy init instance variables
        self._instruction = None
        self._target = None
        self._layout = None
        self._init_target_height = None
        
        self.robot_cfg.update(dict(
            uid=robot_uid,
            # controller_uid=controller_uid,
        ))

        self._robot=RobotRegistry.make(**self.robot_cfg)

        # domain randomization confs
        if scene_uid is not None:
            assert isinstance(self.dr_cfgs["scene"], TabletopSceneDRCfg)
            self.dr_cfgs["scene"].room_choices = [scene_uid] # type:ignore

        if target_object is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target_object # type:ignore

        drmgr = TabletopGraspDRManager(level=dr_level, **self.dr_cfgs)
        super().__init__(
            dr=drmgr,
            split=split, 
            render_hz=render_hz, 
            dr_level=dr_level, 
            physics_dt=physics_dt, 
            *args, 
            **kwargs
        )

    @property
    def layout(self) -> Layout:
        """Returns the layout of the task."""
        assert self._layout is not None, "call reset() first"
        return self._layout
    
    @property
    def instruction(self) -> str:
        assert self._instruction is not None, "call reset() first"
        return self._instruction
    
    @property
    def target(self) -> Actor:
        assert self._target is not None, "call reset() first"
        return self._target
    
    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            # "agent": spaces.Box(-np.pi, np.pi, shape=(self.robot.dof,), dtype=np.float32), # FIXME handle gripper
            "joint_qpos": spaces.Box(-np.pi, np.pi, shape=(self.robot.dof,), dtype=np.float32),
            # "eef_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),#TODO
            # "mujoco": spaces.Box(0, 255, shape=(360, 640, 3), dtype=np.uint8),
            # "front_stereo_left": spaces.Box(0, 255, shape=(360, 640, 3), dtype=np.uint8),
            # "front_stereo_right": spaces.Box(0, 255, shape=(360, 640, 3), dtype=np.uint8),
            # "wrist": spaces.Box(0, 255, shape=(270, 480, 3), dtype=np.uint8),
            # "wrist_left": spaces.Box(0, 255, shape=(270, 480, 3), dtype=np.uint8),
            # "side_left": spaces.Box(0, 255, shape=(360, 640, 3), dtype=np.uint8),
        }
        if isinstance(default_obs, spaces.Dict):
            obs.update(dict(default_obs))
        return spaces.Dict(obs)
    
    def reset(self, seed:int|None=None, options: Optional[dict[str, Any]] = None) -> None:
        """Resets the task state."""       
        super().reset(seed, options)
        split = self.metadata.get("split", "train")
        self._target = self.layout.actors.get("target")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        language_template = lang_dr(split)
        self._instruction = language_template.format(self._target.asset.name) # type: ignore
        self._init_target_height = None

    def clone_layout(self, options: dict[str, Any]) -> None:
        """Replicates an environment layout from given options."""
        self._layout = self.dr.replicate_env(self.robot, self.sensor_cfgs, deepcopy(options), self.split)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({}) # TODO
        return state_dict
    
    # def load_state(self, options: Dict[str, Any]) -> None:
    #     return super().load_state(options)

    def check_success(self,  info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info)
        return reward >= 1.0
    
    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        target_obj_height = info["target"][2]
        # print("===="*10)
        # print(f"target_obj_height {target_obj_height} ")
        # print("===="*10)
        if self._init_target_height is None:
            self._init_target_height = target_obj_height
        """Compute reward based on info dict."""
        reward = np.clip((target_obj_height - self._init_target_height) / _LIFT_HEIGHT, 0, 1)
        return reward

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        asset_manager = AssetManager.get("graspnet1b")

        # for asset in asset_manager:
        #     yield ObjectActor(asset=asset) # ActorReigstry.make("object", asset)#

        return [ObjectActor(asset=asset) for asset in asset_manager]

    def decompose(self):
        from simple.datagen.subtask_spec import (
            OpenGripperSpec,
            # MoveEEFToPoseSpec,
            CloseGripperSpec,
            GraspObjectSpec,
            LiftSpec
        )

        return [
            OpenGripperSpec("init", grasp_type="bodex", hand_uid="inspire_right"),
            GraspObjectSpec("approach", target_uid=self.target.uid, pregrasp=False,
                            grasp_type="bodex", hand_uid="inspire_right", lock_links=["waist_roll_link"]),
            # CloseGripperSpec("grasp"),
            LiftSpec("lift", up=0.1, grasp_type="bodex", hand_uid="inspire_right"),
        ]
