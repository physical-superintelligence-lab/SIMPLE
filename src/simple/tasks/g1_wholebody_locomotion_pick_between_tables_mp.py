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

@TaskRegistry.register("g1_wholebody_locomotion_pick_between_tables_mp")
class G1WholebodyLocomotionPickBetweenTablesTaskMP(Task):
    uid:str = "g1_wholebody_locomotion_pick_between_tables_mp"
    label: str = "G1 Wholebody Locomotion Pick and Place MP"
    description: str = "A task where the G1 robot must pick an object from table1, locomote to table2, and place it in a container."

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
                position=[1.0, 0, 1.475],
                eulers=np.deg2rad([0, 38, 180]),
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
            fov=np.deg2rad(90),
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
                position=[-1.5, 3.0, 1.875],
                eulers=np.deg2rad([0, 40, -90]),
            )
        ),
    )


    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language = LanguageDRCfg(
            instructions = [
                "pick the object from table1, locomote to table2, and place it on table2.",
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
            number_of_distractors=4,
            allow_duplicates=False,
            exclude=["0","12","46","6"]  # Exclude the target object 
        ),

        spatial = SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-0.69, 0.0,], high=[-0.78, 0.0]),
            # robot_region=Box(low=[-1.15, 0.0,], high=[-1.17, 0.0]),
            target_region=Box(low=[-0.39, -0.07], high=[-0.35, 0.09]),
            container_region=Box(low=[-2.7, 0.35], high=[-2.6, 0.45]),
            
            distractors_region=[
                Box(low=[-0.2, -0.3], high=[-0.0, 0.3]),  # distractor_0 on table1
                Box(low=[-0.2, -0.3], high=[-0.0, 0.3]),  # distractor_1 on table1
                Box(low=[-0.2, -0.3], high=[-0.0, 0.3]),  # distractor_2 on table1
                Box(low=[-2.8, -0.3], high=[-2.6, 0.3])   # distractor_3 on table2
            ],
            target_stable_indices = [0],
            target_rotate_z = Box(low=-0.15, high=0.15),
            obj_surface_map={
                "target": "table",
                "container": "table2",
                "distractor": ["table", "table", "table", "table2"] 
            }
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
            table2_height=Box(low=-0.14, high=-0.14),
            room_choices=["hssd:scene6"],
            scene_manager="hssd",
            enable_table2=True,
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
        scene_uid: str | Scene = "hssd:scene6",
        target_object: str | Object = "graspnet1b:0",
        # container_object: str | Object = "graspnet1b:46",
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

        self._robot=RobotRegistry.make(**self.robot_cfg)

         # domain randomization confs
        #HACK This task only in scene6
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
            
        # if container_object is not None:
        #     assert isinstance(self.dr_cfgs["container"], TargetDRCfg)
        #     self.dr_cfgs["container"].asset_id = container_object # type:ignore

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
            # "agent": spaces.Box(-np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32), 
            "joint_qpos": spaces.Box(-np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32),
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
        self._contact_started = False
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

    def check_object_on_table2(self, *args, **kwargs) -> bool:
        """
        Check if the target object is on table2 by detecting contact with table2.
        """
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False
            
        target_name = str(self.target.asset.label)
        
        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name

            # Check if target is in contact with table2
            # table2_geom is the name used in scene.py for table2
            if (target_name in body1 and "table2" in body2) or \
               (target_name in body2 and "table2" in body1):
                return True
        return False
    
    def check_object_in_container(self, *args, **kwargs) -> bool:
        """
        Check if the target object is successfully placed in the container.
        """
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False
            
        target_name = str(self.target.asset.label)
        container_name = str(self.container.asset.label)

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
    
    def check_hand_object_contact(self, *args, **kwargs) -> bool:
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False
            
        target_name = str(self.target.asset.label)

        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name

            # Check if target and container are in contact
            if (target_name in body1 and "hand" in body2) or \
               (target_name in body2 and "hand" in body1):
                return True
        return False

    def check_success(self,  info: dict[str, Any], *args, **kwargs) -> bool:
        return self.reward >= 0.9
    
    def compute_reward(self, info: dict[str, Any] ,*args, **kwargs) -> float:
        is_object_in_container = self.check_object_in_container(info=info, *args, **kwargs)
        is_object_contacted_by_hand = self.check_hand_object_contact(*args, **kwargs)

        if is_object_in_container and not is_object_contacted_by_hand:
            self.reward = 1.0
        else:
            self.reward = 0.0

        return self.reward

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
            LowerSpec,
            RetreatSpec,
            PhaseBreakSpec,
            MoveEEFToPoseSpec,
        )
        
        place_position = np.array([*self.container.pose.position[:2], 0.25])
    
        return [
            StandSpec("initialize"),
            # WalkSpec("walk_to_table", vx=0.35, vy=0.0, target_yaw=0, target_distance=0.55),
            # StandSpec("initialize2"),
            PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"),
            GraspObjectSpec(
                "grasp_object", 
                target_uid=self.target.uid, 
                pregrasp=False,
                grasp_type="bodex",
                hand_uid="dex3_right",
                lock_links=["left_hand_palm_link"]
            ),
            LiftSpec("lift", up=0.08, grasp_type="bodex", hand_uid="dex3_right"),
            TurnSpec("turn_90", vx=0.1, target_yaw=np.pi/2),
            TurnSpec("turn_180", vx=0.1, target_yaw=np.pi),
            WalkSpec("walk_to_table2", vx=0.35, vy=0.15, target_yaw=np.pi, target_distance=1.39),
            # StandSpec("stand_at_table2"),
            PhaseBreakSpec("phase_break_before_place", grasp_type="bodex"),
            MoveEEFToPoseSpec(
                "move_to_container",                     
                position=place_position,
                grasp_type="bodex",
                hand_uid="dex3_right",
                lock_links=["waist_roll_link", "left_hand_palm_link"],
            ),
            # LowerSpec("lower", down=0.02, grasp_type="bodex"),
            OpenGripperSpec("release", hand_uid="dex3_right"),
            StandSpec("stop"),
        ]

