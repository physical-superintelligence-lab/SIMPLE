"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from simple.core.randomizer import RandomizerCfg
    from simple.datagen.subtask_spec import SubtaskSpec

from typing import Any

import numpy as np
from gymnasium import spaces
import random

from simple.assets import AssetManager
from simple.core.actor import Actor, ActorReigstry, ObjectActor
from simple.core.layout import Layout
from simple.core.object import Object
from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.robot import Robot
from simple.core.scene import Scene
from simple.core.task import Task
from simple.dr import *
from simple.dr.manager import DRManager, TabletopGraspDRManager  # , LayoutManager
from simple.dr.types import Box
from simple.robots.protocols import Controllable
from simple.robots.registry import RobotRegistry
from simple.sensors import CameraCfg, SensorCfg, StereoCameraCfg
from simple.tasks.registry import TaskRegistry

_LIFT_HEIGHT = 0.1
_PLACE_HEIGHT = 0.5
_LOWER_HEIGHT = 0.1


@TaskRegistry.register("g1_wholebody_open_trash_can_teleop")
class G1WholebodyOpenTrashCanTaskTeleop(Task):
    uid: str = "g1_wholebody_open_trash_can_teleop"
    label: str = "G1 TELEOP Open Trash Can"
    description: str = (
        "A task where the G1 robot must open the trash can."

    )

    metadata: dict[str, Any] = {
        "physics_dt": 0.005,
        "control_hz": 200,
        "render_hz": 50,
        "dr_level": 0,
        "version": 1.0,
        "reward_dt": 0.02,
        "image_dt": 0.033333,
        "need_gravity": True,
        "max_episode_steps": 800,
        # "debug": True
    }

    robot_cfg: dict[str, Any] = dict(
        uid="g1_sonic",
    )

    sensor_cfgs: dict[str, SensorCfg] = dict(
        head_stereo=StereoCameraCfg(
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
            ),
        ),
    )

    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language=LanguageDRCfg(
            instructions=[
                "move forward to the trash can and open it",
            ]
        ),
        target=TargetDRCfg(asset_id="graspnet1b:12"),  # e.g., "primitive:cube"
        distractors=DistractorDRCfg(
            res_id="graspnet1b",
            number_of_distractors=3,
            allow_duplicates=False,
            exclude=["0"],  # Exclude the target object
        ),
        articulated = ArticulatedObjectDrCfg(
            asset_id="articulated:5", # e.g., "
        ),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-1.5, 0, 0.0], high=[-1.6, 0.0, 0.0]),
            target_region=Box(low=[0.3, -0.3], high=[0.5, -0.4]),
            distractors_region=Box(low=[0.2, -0.3], high=[0.4, 0.3]),
            target_stable_indices=[0],
            target_rotate_z=Box(low=-0.15, high=0.15),
            articulated_region=Box(low=[-0.85, -0.1,0.3], high=[-1, 0.1,0.3]),
            articulated_rotate_z= Box(low=-0.12, high=0.12),
        ),
        camera=CameraDRCfg(
            cam_id="franka_camera",
            # position=[0.5, 0, 0.5], # TODO
            # orientation=[0, 0, 0], # TODO define a range
        ),
        scene=TabletopSceneDRCfg(
            # asset_id="primitive:table",
            # scene_mode="random", # fixed, random
            # table_size=Box(low=[1.4, 1.4, 0.1], high=[1.8, 1.8, 0.2]),  # Table dimensions
            table_position=Box(low=[0.3, 0], high=[0.3, 0]),
            # table_height=Box(low=0.0, high=0.0),
            # rotation_z=Box(low=0, high=3.14),  # Rotation around the Z-axis
            table_height=Box(low=0.67, high=0.68),
            room_choices=["hssd:scene4"],
            scene_manager="hssd",
        ),
        lighting=LightingDRCfg(
            light_mode="random",  # fixed, random
            light_num=(2, 3),
            light_color_temperature=Box(low=2001, high=8001),  # I was not joking :)
            light_intensity=Box(low=1e4, high=1e4),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 1.1),
            light_spacing=Box((1.0, 1.0), (2.0, 2.0)),
            light_position=Box((-0.5, -0.5, 1.3), (0.5, 0.5, 1.5)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        material=MaterialDRCfg(
            material_mode="rand_all",  # fixed, rand_all, rand_tableground, rand_objects
        ),
    )

    def __init__(
        self,
        robot_uid: str = "g1_sonic",
        scene_uid: str | Scene = "hssd:scene1",
        target: str | None = None,
        controller_uid: str = "pd_joint_pos",  # pd_joint_vel, pd_ee_pose, pd_delta_ee_pose
        split: str = "train",  # train, val, test
        render_hz: int | None = None,
        dr_level: int = 0,
        # physics_dt: float = 0.002,
        success_criteria: float = 0.9,
        *args,
        **kwargs,
    ):
        # lazy init instance variables
        self._instruction = None
        self._target = None
        self._layout = None
        self._init_target_height = None
        self._contact_started = False

        self.robot_cfg.update(
            dict(
                uid=robot_uid,
                # controller_uid=controller_uid,
            )
        )

        self.reward = 0
        self.success_criteria = success_criteria

        self._robot = RobotRegistry.make(**self.robot_cfg, **kwargs)

        # domain randomization confs
        # HACK THIS task is only in scene0
        # if scene_uid is not None:
        #     assert isinstance(self.dr_cfgs["scene"], TabletopSceneDRCfg)
        #     self.dr_cfgs["scene"].room_choices = [scene_uid] # type:ignore

        if target is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target  # type:ignore

            # Exclude target object from distractors to avoid duplicates
            target_id = self.dr_cfgs["target"].asset_id.split(":")[-1]
            distractor_cfg = self.dr_cfgs.get("distractors")
            if distractor_cfg is not None and isinstance(
                distractor_cfg, DistractorDRCfg
            ):
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
            # physics_dt=physics_dt,
            *args,
            **kwargs,
        )

        
        # assert render_hz == (0.1 / self.metadata["physics_dt"]), f"only supports render/physics step parity for g1 wholebody tasks (also follow AMO)"

    @property
    def layout(self) -> Layout:
        """Returns the layout of the task."""
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def instruction(self) -> str:
        assert self._instruction is not None, "call reset() first"
        return self._instruction  # type: ignore

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
            "joint_qpos": spaces.Box(
                -np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32
            ),  # type:ignore
        }
        if isinstance(default_obs, spaces.Dict):
            obs.update(dict(default_obs))
        return spaces.Dict(obs)

    def reset(
        self, seed: int | None = None, options: Optional[dict[str, Any]] = None
    ) -> None:
        super().reset(seed, options)
        split = self.metadata.get("split", "train")
        self._target = self.layout.actors.get("target")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        language_template = lang_dr(split)
        self._instruction = language_template.format(self._target.asset.name)  # type: ignore
        self._init_target_height = None
        self.reward = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({})  # TODO
        return state_dict

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)
        return reward >= self.success_criteria
    
    def check_wether_trash_can_is_opened(self, info: dict[str, Any], *args, **kwargs) -> bool:
        mujoco_env = kwargs.get("mujoco_env",None)
        if mujoco_env is  None:
            return False
        
        trash_can_joint0_qpos = mujoco_env.mjData.joint("articulate_joint_0").qpos[0]
        # faucet_joint1_qpos = mujoco_env.mjData.joint("articulate_joint_1").qpos[0]
        return trash_can_joint0_qpos > 0.8  
        

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        trash_can_is_opened = self.check_wether_trash_can_is_opened(info, *args, **kwargs)
        if trash_can_is_opened:
            self.reward +=0.03
        else:
            self.reward = 0.0
        return self.reward

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]

    def decompose(self) -> list[SubtaskSpec]:
        from simple.datagen.subtask_spec import (  # WalkSpec,; TurnSpec,; OpenGripperSpec,; MoveEEFToPoseSpec,; LiftSpec,; LowerSpec,; RetreatSpec
            GraspObjectSpec,
            HeightAdjustSpec,
            PhaseBreakSpec,
            StandSpec,
        )

        return [
            StandSpec(
                "initialize",
            ),
            HeightAdjustSpec("adjust_height", height=-0.3, keep_waist_pose=True),
            PhaseBreakSpec("phase_break_before_pick", grasp_type="bodex"),
            GraspObjectSpec(
                "approach",
                target_uid=self.target.uid,
                pregrasp=False,
                grasp_type="bodex",
                hand_uid="dex3_right",
                lock_links=["left_hand_palm_link"],
            ),
            HeightAdjustSpec("adjust_height", height=0, keep_waist_pose=True),
            # TurnSpec("turn", target_yaw=0),
            StandSpec(
                "end",
            ),
        ]
