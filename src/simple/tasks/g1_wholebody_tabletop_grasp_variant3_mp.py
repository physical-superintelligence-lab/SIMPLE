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


_LIFT_HEIGHT = 0.02
@TaskRegistry.register("g1_wholebody_tabletop_grasp_variant3_mp")
class G1WholebodyTabletopGraspVariant3MP(Task):
    uid:str = "g1_wholebody_tabletop_grasp_variant3_mp"
    label: str = "G1 Wholebody Tabletop Grasp Variant 3 MP"
    description: str = "A task where the G1 robot must grasp a obj from the tabletop."

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
            fov=np.deg2rad(90), 
            near=0.2,
            far=5,
            baseline=0.055,
            pose=dict( 
                # https://en.wikipedia.org/wiki/Spherical_coordinate_system
                # robot is at the origin, camera is orbiting around the robot
                distance=1.5, # radial distance r
                polar=np.deg2rad(40),  # polar angle θ
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
                "grasp the object on the tabletop.",
            ]
        ),

        target = TargetDRCfg(
            asset_id="graspnet1b:14" # e.g., "primitive:cube"
        ),

        distractors = DistractorDRCfg(
            res_id="graspnet1b", 
            number_of_distractors=3,
            allow_duplicates=False,
            exclude=["14"]  # Exclude the target object 
        ),

        spatial = SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[0.58,0,0.0], high=[0.63,0,0.0]),
            robot_orientation_region=Box(low=[0.0, 0, 0, 1.0], high=[0.0, 0, 0, 1.0]),

            target_region=Box(low=[0.25, -0.02], high=[0.28, 0.05]),

            distractors_region=Box(low=[-0.3, -0.3], high=[-0.5, 0.3]),
            target_stable_indices = [0],
            target_rotate_z = Box(low=-3.14, high=-3.14),
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
            # table_position=Box(low=[0, 0.25], high=[0.0, 0.25]),
            # table_height=Box(low=0.0, high=0.0),
            # rotation_z=Box(low=0, high=3.14),  # Rotation around the Z-axis
            table_height=Box(low=0.0, high=0.0),
            room_choices=["hssd:scene49"],
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
        scene_uid: str | Scene = "hssd:scene2",
        target_object: str | Object = "graspnet1b:12",
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
        self.reward = 0
        self.robot_cfg.update(dict(
            uid=robot_uid,
            # controller_uid=controller_uid,
        ))

        self._robot=RobotRegistry.make(**self.robot_cfg)

         # domain randomization confs
        #HACK This task only in scene3
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
        state_dict.update({}) # TODO
        return state_dict
    
    def check_object_in_destination(self, *args, **kwargs) -> bool:
        """
        Check if the target object is successfully placed in the container.
        """
        # check if the target object is in the container
        
        target_pos = np.array(self.layout.actors["target"].pose.position)

        err_xy = np.abs(target_pos[:2]-[0.8,0])
        success = np.all(err_xy <= 1)



        target_name = str(self.target.asset.label)# type: ignore
        container_name = str("table")
        mujoco_env = kwargs.get("mujoco_env", None)

        mj_physics_data = mujoco_env.mjData # type: ignore
        mj_physics_model = mujoco_env.mjModel # type: ignore

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name

            # Check if target and container are in contact
            if (target_name in body1 and container_name in body2) or \
               (target_name in body2 and container_name in body1) and success:
                return True
        return False

    def check_success(self,  info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)

        return reward >= 0.9
    
    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
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
            GraspObjectSpec,
            LiftSpec,
            PhaseBreakSpec,
        )



        return [
            StandSpec("initialize",),
            PhaseBreakSpec("phase_break_before_handover",grasp_type="bodex"),
            # GraspObjectSpec("approach", target_uid=self.target.uid, pregrasp=False,grasp_type="bodex",hand_uid="dex3_right",lock_links=[ "left_hand_palm_link"]),
            GraspObjectSpec("approach", target_uid=self.target.uid, pregrasp=False,grasp_type="bodex",hand_uid="dex3_right",lock_links=[ "left_hand_palm_link"]),
            LiftSpec("lift", up=0.1, grasp_type="bodex",hand_uid="dex3_right"),

            # TurnSpec("turn",vx=0.1,target_yaw=1.57),
            # WalkSpec("walk",vx = 0.35,vy=0.15,target_yaw=1.57,target_distance=target_distance_0),
            # TurnSpec("turn",vx=0.1,target_yaw=0),
            # WalkSpec("walk",vx = 0.4,vy=0.15,target_yaw=0,target_distance=target_distance_1),
            # TurnSpec("turn",vx=0.1,target_yaw=-1.57),
            # WalkSpec("walk",vx = 0.35,vy=0.15,target_yaw=-1.57,target_distance=target_distance_2),
            # TurnSpec("turn",vx=0.1,target_yaw=-3.14),
            # OpenGripperSpec("open_gripper",hand_uid="dex3_right"),
            # StandSpec("stop"),


            
        ]

