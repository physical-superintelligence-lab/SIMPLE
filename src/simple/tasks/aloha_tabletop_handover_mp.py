"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional
if TYPE_CHECKING:
    from simple.core.randomizer import RandomizerCfg
from simple.core.task import Task
from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.object import Object
from simple.core.actor import Actor, ObjectActor
from simple.core.layout import Layout
from simple.core.robot import Robot
from simple.dr import *
from simple.dr.manager import DRManager, TabletopGraspDRManager

from simple.dr.types import Box
from simple.robots.registry import RobotRegistry
from simple.tasks.registry import TaskRegistry
from simple.core.actor import ActorReigstry
from simple.sensors import StereoCameraCfg, SensorCfg, CameraCfg
from simple.robots.protocols import Controllable
from copy import deepcopy
from gymnasium import spaces
import numpy as np
from typing import Any
import math
import transforms3d as t3d
from simple.assets import AssetManager

_LIFT_HEIGHT = 0.18
_LOWER_HEIGHT = 0.08

@TaskRegistry.register("aloha_tabletop_handover_mp")
class AlohaTabletopHandoverTaskMP(Task):
    
    uid: str = "aloha_tabletop_handover_mp"
    label: str = "Aloha Tabletop Handover Task"
    description: str = "A task where the Aloha robot's left arm grasps an object and hands it over to the right arm."

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "dr_level": 0,
        "version": 1.0,
    }
    
    robot_cfg: dict[str, Any] = dict(
        uid="aloha",
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
                distance=1.35, # radial distance r
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
                position=[0.0, 0.0, 0.0],
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
                "Hand over {} from left hand to right hand.",
                "Pass {} from left arm to right arm.",
                "Transfer {} from left gripper to right gripper.",
            ]
        ),

        target = TargetDRCfg(
            asset_id="graspnet1b:1" # e.g., "primitive:cube"
        ),

        distractors = DistractorDRCfg(
            res_id="graspnet1b", 
            number_of_distractors=0,
            allow_duplicates=False,
            exclude=["1"]  # Exclude the target object 
        ),

        spatial = SpatialDRCfg(
            spatial_mode="random",
            # fixed_stable_pose_idx=5,
            robot_region=Box(low=[0.1, -0.0], high=[0.1, -0.0]),
            target_region=Box(low=[0.1, -0.1], high=[0.1, 0.1]),
            distractors_region=Box(low=[-0.3, -0.3], high=[0.3, 0.3]),
        ),

        camera = CameraDRCfg(
            cam_id="aloha_camera",
            # position=[0.5, 0, 0.5], # TODO
            # orientation=[0, 0, 0], # TODO define a range
        ),

        scene = TabletopSceneDRCfg(
            # asset_id="primitive:table",
            # table_size=Box(low=[1.4, 1.4, 0.1], high=[1.8, 1.8, 0.2]),  # Table dimensions
            # table_position=Box(low=[0.2, -0.25], high=[0.6, 0.25]),
            # table_height=Box(low=0., high=0.), #Box(low=0.4, high=0.6),
            # rotation_z=Box(low=0, high=3.14),  # Rotation around the Z-axis
            # room_choices="[]"
            table_height=Box(low=0.0, high=0.0),
            scene_manager="hssd"
        ),

        lighting = LightingDRCfg(
            light_mode="random", # fixed, random
            light_num=(2,3),
            light_color_temperature=Box(low=4001, high=6001),
            light_intensity=Box(low=8e4, high=1e5),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 2.1),
            light_spacing=Box((1., 1.), (2.5, 2.5)),
            light_position=Box((-1.1, -1.1, 2.1), (1.1, 1.1, 4.1)),
            light_eulers=Box((0,0,-0.5*math.pi), (0,0,0.5*math.pi))
        ),

        material = MaterialDRCfg(
            material_mode="rand_all", # fixed, rand_all, rand_tableground, rand_objects
        )
    )

    def __init__(
        self, 
        robot_uid: str = "aloha",
        target_object: str | None = None, # "graspnet1b:63"
        scene_uid: str | None = None,
        split: str = "train",
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
        self._handover_stage = 0  # 0: grasp, 1: lift, 2: handover, 3: complete
        
        self._lift_reward = 0.0
        self._handover_reward = 0.0
        self._lower_reward = 0.0
        
        self.robot_cfg.update(dict(uid=robot_uid))
        self._robot = RobotRegistry.make(**self.robot_cfg)

        # domain randomization confs
        if scene_uid is not None:
            assert isinstance(self.dr_cfgs["scene"], TabletopSceneDRCfg)
            self.dr_cfgs["scene"].room_choices = [scene_uid] # type:ignore

        if target_object is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target_object
            
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
        # TODO derive from robot and sensors
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            "agent": spaces.Box(-np.pi, np.pi, shape=(self.robot.dof,), dtype=np.float32), # FIXME handle gripper
            "joint_qpos": spaces.Box(-np.pi, np.pi, shape=(self.robot.dof,), dtype=np.float32),
            "eef_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
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
        self._handover_stage = 0


    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({})
        return state_dict

    def check_grasp_contact(self, arm: str, *args, **kwargs) -> bool:
        """
        Check if the specified arm is in contact with the target object.
        """
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False
        
        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel
        
        target_name = str(self.target.asset.label)
        gripper_name = f"{arm}_right_finger"
        
        # Iterate through all contacts
        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name

            # Check if target and gripper are in contact
            if (target_name in body1 and gripper_name in body2) or \
               (target_name in body2 and gripper_name in body1):
                return True
        return False

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info)
        return reward >= 0.9 and self._handover_stage >= 2
    
    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:        
        # target_pos = np.array(info[str(self.target.uid)])
        target_pos = info["target"]
        self.target.pose.position = target_pos[:3].tolist()
        self.target.pose.quaternion = target_pos[3:7].tolist()
        
        target_actor = self.layout.actors["target"]
        object_pose = target_actor.pose
        # print(f"target_pos: {target_pos}\n, object_pose: {object_pose.position, object_pose.quaternion}")
        
        if self._init_target_height is None:
            self._init_target_height = target_pos[2]
        
        left_grasped = self.check_grasp_contact("left", *args, **kwargs)
        right_grasped = self.check_grasp_contact("right", *args, **kwargs)
        
        if self._handover_stage == 0:
            self._lift_reward = np.clip(
                (target_pos[2] - self._init_target_height) / _LIFT_HEIGHT, 
                0, 1
            )
            
            if left_grasped:
                self._handover_stage = 1
                self._lift_reward = 1.0
                print(f"Stage 0 finished: Left arm has lifted the object, reward: {self._lift_reward}")
        
        elif self._handover_stage == 1:
            if right_grasped:
                self._handover_reward = 0.5
                self._handover_stage = 2
                print("Stage 1 finished: Right arm has contacted the object.")
                
        elif self._handover_stage == 2:
            if right_grasped and not left_grasped:
                self._handover_reward += 0.5
                self._handover_stage = 3
                print("Stage 2 finished: Right arm has handed over the object.")
                self._init_target_height = target_pos[2] # reset init_target_height
        
        elif self._handover_stage == 3:
            self._lower_reward = np.clip(
                (self._init_target_height - target_pos[2]) / _LOWER_HEIGHT, 
                0, 1
            )
            
        reward = self._lift_reward*0.3 + self._handover_reward*0.5 + self._lower_reward*0.2
        return float(np.clip(reward, 0, 1))

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]
    
    def decompose(self):
        from simple.datagen.subtask_spec import (
            OpenGripperSpec,
            CloseGripperSpec,
            GraspObjectSpec,
            LiftSpec,
            # MoveEEFToPoseSpec,
            PhaseBreakSpec,
        )

        return [
            OpenGripperSpec("init_left", hand_uid="left"),
            GraspObjectSpec(
                "approach_left", 
                target_uid=self.target.uid, 
                pregrasp=False, 
                hand_uid="left",
                # grasp_bias="outward",
                # store_grasp_key="left_arm_grasps"
            ),
            CloseGripperSpec("grasp_left", hand_uid="left"),
            LiftSpec("lift_left", hand_uid="left", up=0.18),
            # CloseGripperSpec("grasp_left", hand_uid="left"),
            OpenGripperSpec("init_right", hand_uid="right"),
            PhaseBreakSpec("phase_break_after_left"),
            GraspObjectSpec(
                "approach_right", 
                target_uid=self.target.uid, 
                pregrasp=False, 
                hand_uid="right", 
                use_negative_x_filter=True,
                approach_axis='x'
            ),
            CloseGripperSpec("grasp_right", hand_uid="right"),
            OpenGripperSpec("release_left", hand_uid="left"),
            # RetreatSpec("retreat", hand_uid="left", up=_LIFT_HEIGHT),
            LiftSpec("lower_right", hand_uid="right", up=0.1, step_distance=-0.01),
        ]