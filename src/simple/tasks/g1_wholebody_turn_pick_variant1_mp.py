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
_PLACE_HEIGHT = 0.45
_LOWER_HEIGHT = 0.05
@TaskRegistry.register("g1_wholebody_turn_pick_variant1_mp")
class G1WholebodyTurnPickVariant1MP(Task):
    uid:str = "g1_wholebody_turn_pick_variant1_mp"
    label: str = "G1 Wholebody Turn Pick Variant1 MP"
    description: str = "A task where the G1 robot must turn and pick an object."

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "dr_level": 0,
        "version": 1.0,
        "need_gravity": True,
    }

    robot_cfg: dict[str, Any] = dict(
        uid="g1_wholebody",
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
                position=[0.00, 0.0, 0.05],
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
                "from the start position, turn to the target position and pick up the object",
            ]
        ),

        target = TargetDRCfg(
            asset_id="graspnet1b:0" # e.g., "primitive:cube"
        ),
        container = TargetDRCfg(
            asset_id="graspnet1b:46"  # or used "graspnet1b:6"
        ),
        distractors = DistractorDRCfg(
            res_id="graspnet1b", 
            number_of_distractors=2,
            allow_duplicates=False,
            exclude=["0"]  # Exclude the target object 
        ),

        spatial = SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-1.35, 0.21], high=[-1.32, 0.21]),
            robot_orientation_region = Box(
                low = np.array([0.96592583, 0.0, 0.0, -0.4], dtype=np.float32),
                high= np.array([1.00000000, 0.0, 0.0,  0.4], dtype=np.float32)
            ),

            target_region=Box(low=[-0.83,0.21], high=[-0.87, 0.21]),
            container_region=Box(low=[-0.5, 0.2], high=[-0.55, 0.28]),
            distractors_region=Box(low=[-0.2, -0.3], high=[0.4, 0.3]),
            target_stable_indices = [0],
            target_rotate_z = Box(low=-0.02, high=0.02),
            container_rotate_z = Box(low=1.57, high=1.59)
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
            table_position=Box(low=[-0.0, 0.25], high=[-0.0, 0.25]),
            # table_height=Box(low=0.0, high=0.0),
            # rotation_z=Box(low=0, high=3.14),  # Rotation around the Z-axis
            table_height=Box(low=0.0, high=0.0),
            room_choices=["hssd:scene25"],
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
        robot_uid: str = "g1_wholebody",
        scene_uid: str | Scene = "hssd:scene25",
        target_object: str | Object = "graspnet1b:0",
        container_object: str | Object = "objects:0",
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
        self._container = None
        self._layout = None
        self._init_target_height = None
        self._contact_started = False
        
        self.robot_cfg.update(dict(
            uid=robot_uid,
            # controller_uid=controller_uid,
        ))

        self.reward = 0

        self._robot=RobotRegistry.make(**self.robot_cfg)

         # domain randomization confs
        #HACK THIS task is only in scene25
        # if scene_uid is not None:
        #     assert isinstance(self.dr_cfgs["scene"], TabletopSceneDRCfg)
        #     self.dr_cfgs["scene"].room_choices = [scene_uid] # type:ignore

        if target_object is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target_object # type:ignore
            
            # Exclude target object from distractors to avoid duplicates
            target_id = self.dr_cfgs["target"].asset_id.split(":")[-1]
            distractor_cfg = self.dr_cfgs.get("distractors")
            if distractor_cfg is not None and isinstance(distractor_cfg, DistractorDRCfg):
                if distractor_cfg.exclude is None:
                    distractor_cfg.exclude = []
                if target_id not in distractor_cfg.exclude:
                    distractor_cfg.exclude.append(target_id)
            
        if container_object is not None:
            assert isinstance(self.dr_cfgs["container"], TargetDRCfg)
            self.dr_cfgs["container"].asset_id = container_object # type:ignore
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
    def container(self) -> Actor:
        assert self._container is not None, "call reset() first"
        return self._container

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space
    @property
    def observation_space(self) -> spaces.Space:
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            # "agent": spaces.Box(-np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32), #type:ignore
            "joint_qpos": spaces.Box(-np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32),#type:ignore
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
        self._container = self.layout.actors.get("container")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        language_template = lang_dr(split)
        self._instruction = language_template.format(self._target.asset.name) # type: ignore
        self._init_target_height = None
        self.reward = 0
        self.robot.reset()

    # def clone_layout(self, options: dict[str, Any]) -> None:
    #     """Replicates an environment layout from given options."""
    #     self._layout = self.dr.replicate_env(self.robot, self.sensor_cfgs, deepcopy(options), self.split)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({
            "container_uid": self.container.uid if self._container else None,
        }) # TODO
        return state_dict
    
    # def load_state(self, options: Dict[str, Any]) -> None:
    #     return super().load_state(options)

    def check_object_in_container(self, *args, **kwargs) -> bool:
        """
        Check if the target object is successfully placed in the container.
        """
        target_name = str(self.target.asset.label)
        container_name = str(self.container.asset.label)
        mujoco_env = kwargs.get("mujoco_env", None)

        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name

            # Check if target and container are in contact
            if (target_name in body1 and container_name in body2) or \
               (target_name in body2 and container_name in body1):
                return True
        return False
    
    def check_wether_handover_success(self, *args, **kwargs) -> bool:
        target_name = str(self.target.asset.label)
        mujoco_env = kwargs.get("mujoco_env", None)

        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name

            # Check if target and container are in contact
            if (target_name in body1 and "left_hand" in body2) or \
               (target_name in body2 and "left_hand" in body1):
                return True
        return False

    def check_success(self,  info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)
        return reward >= 0.9
    
    def compute_reward(self, info: dict[str, Any] ,*args, **kwargs) -> float:
        target_obj_height = info["target"][2]
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
            StandSpec,
            WalkSpec,
            TurnSpec,
            HeightAdjustSpec,
            OpenGripperSpec,
            MoveEEFToPoseSpec,
            GraspObjectSpec,
            LiftSpec,
            PhaseBreakSpec,
            LowerSpec,
            RetreatSpec
        )

        handover_region=Box(low=[-0.3 ,-0.04], high=[-0.34, 0.04])
        handover_position = np.array(handover_region.sample())

        transition_region = Box(low=[-0.92, 0.3], high=[-1, 0.4])
        transition_position = np.array(transition_region.sample())

        return [
            StandSpec("initialize",),
            
            TurnSpec("turn", vx=0.1, target_yaw=0),
            StandSpec("initialize",),


            PhaseBreakSpec("phase_break_before_pick",grasp_type="bodex"),

            GraspObjectSpec("approach", target_uid=self.target.uid, pregrasp=False,grasp_type="bodex",hand_uid="dex3_right",lock_links=[ "left_hand_palm_link"]),
            LiftSpec("lift", height =0.06, grasp_type="bodex",hand_uid="dex3_right"),
            # WalkSpec("walk_to_container", height =0, vx = -0.1, vy= 0.35,target_yaw=0, distance=walk_distace),
            # HeightAdjustSpec("adjust_height",height=-0.2),
            PhaseBreakSpec("phase_break_before_place",grasp_type="bodex"),
            
           
            # GraspObjectSpec("approach", target_uid=self.target.uid, pregrasp=False,grasp_type="bodex",hand_uid="dex3_left",
            #                 lock_links=["right_hand_palm_link","right_hand_index_1_link","right_hand_middle_1_link","right_hand_thumb_2_link"],keep_force=True),
            # OpenGripperSpec("init",grasp_type="bodex",hand_uid="dex3_right"),
            # LowerSpec("lower", down=_LOWER_HEIGHT ,grasp_type="bodex",hand_uid="dex3_left"),

            # RetreatSpec("retreat_right_arm",                     
            #             grasp_type="bodex",hand_uid="dex3_right",
            #             lock_links=["waist_yaw_link", "waist_roll_link", "torso_link","left_hand_palm_link", "left_hand_index_1_link","left_hand_middle_1_link","left_hand_thumb_2_link"]),

            # MoveEEFToPoseSpec("move_to_transition",                     
            #                 position=transition_position,
            #                 grasp_type="bodex",hand_uid="dex3_left",lock_links=[ "right_hand_palm_link",]
            #                 ),
            StandSpec("initialize",)            # OpenGripperSpec("place" ,grasp_type="bodex",hand_uid="dex3_right"),

            


            
        ]

