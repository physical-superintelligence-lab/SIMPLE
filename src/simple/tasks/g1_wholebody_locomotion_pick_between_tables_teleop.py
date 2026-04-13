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

_PLACE_HEIGHT = 0.4
_PLACE_HEIGHT = 0.5
_LOWER_HEIGHT = 0.1


@TaskRegistry.register("g1_wholebody_locomotion_pick_between_tables_teleop")
class G1WholebodyLocomotionPickBetweenTablesTaskTeleop(Task):
    uid: str = "g1_wholebody_locomotion_pick_between_tables_teleop"
    label: str = "G1 TELEOP Pick Between Tables"
    description: str = (
        "A task where the G1 robot must pick up an object between two tables."
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
        "max_episode_steps": 1200,
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
                " pick up the {} from table1,locomotion to table2,and place  on table2.",
            ]
        ),
        target=TargetDRCfg(asset_id="graspnet1b:0"),  # e.g., "primitive:cube"
        container=TargetDRCfg(asset_id="objects:0"),  # or used "graspnet1b:6"
        distractors=DistractorDRCfg(
            res_id="graspnet1b",
            number_of_distractors=4,
            allow_duplicates=False,
            exclude=["0", "12", "46", "6"],  # Exclude the target object
        ),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-0.63, 0, 0], high=[-0.65, 0, 0]),
            # robot_orientation_region=Box(low=[0.717, 0, 0, -0.717], high=[0.717, 0, 0, -0.717]),
            # robot_region=Box(low=[-1.15, 0.0,], high=[-1.17, 0.0]),
            target_region=Box(low=[-0.33, -0.04], high=[-0.31, 0.04]),
            container_region=Box(low=[-2.45, 0.15], high=[-2.5, 0.25]),
            container_rotate_z=Box(low=1.57, high=1.57),
            distractors_region=[
                Box(low=[-0.2, -0.3], high=[-0.0, 0.3]),  # distractor_0 on table1
                Box(low=[-0.2, -0.3], high=[-0.0, 0.3]),  # distractor_1 on table1
                Box(low=[-0.2, -0.3], high=[-0.0, 0.3]),  # distractor_2 on table1
                Box(low=[-2.8, -0.3], high=[-2.6, 0.3]),  # distractor_3 on table2
            ],
            target_stable_indices=[0],
            target_rotate_z=Box(low=-0.15, high=0.15),
            obj_surface_map={
                "target": "table",
                "container": "table2",
                "distractor": ["table", "table", "table", "table2"],
            },
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
            table_position=Box(low=[0, 0.25], high=[0, 0.25]),
            # table_height=Box(low=0.0, high=0.0),
            # rotation_z=Box(low=0, high=3.14),  # Rotation around the Z-axis
            table_height=Box(low=0.65, high=0.65),
            table2_height=Box(low=0.65, high=0.65),
            room_choices=["hssd:scene6"],
            scene_manager="hssd",
            enable_table2=True,
        ),
        lighting=LightingDRCfg(
            light_mode="random",  # fixed, random
            light_num=(2, 3),
            light_color_temperature=Box(low=6001, high=8001),  # I was not joking :)
            light_intensity=Box(low=5e4, high=5e4),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 2.1),
            light_spacing=Box((1.0, 1.0), (2.5, 2.5)),
            light_position=Box((-1.1, -1.1, 1.3), (1.1, 1.1, 1.5)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        material=MaterialDRCfg(
            material_mode="rand_all",  # fixed, rand_all, rand_tableground, rand_objects
        ),
    )

    def __init__(
        self,
        robot_uid: str = "g1_sonic",
        scene_uid: str | Scene = "hssd:scene6",
        target_object: str | Object = "graspnet1b:0",
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
        self._container = None
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

        if target_object is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target_object  # type:ignore

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
        self._container = self.layout.actors.get("container")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        language_template = lang_dr(split)
        self._instruction = language_template.format(self._target.asset.name)  # type: ignore
        self._init_target_height = None
        self._contact_started = False
        self.reward = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update(
            {
                "container_uid": self.container.uid if self._container else None,
            }
        )  # TODO
        return state_dict

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
            if (target_name in body1 and "table2" in body2) or (
                target_name in body2 and "table2" in body1
            ):
                return True
        return False

    def check_object_in_container(self, *args, **kwargs) -> bool:
        """
        Check if the target object is successfully placed in the container.
        """
        # check if the target object is in the container
        container_pos = np.array(self.layout.actors["container"].pose.position)
        target_pos = np.array(self.layout.actors["target"].pose.position)

        err_xy = np.abs(container_pos[:2] - target_pos[:2])
        success = np.all(err_xy <= 0.12)



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
               (target_name in body2 and container_name in body1) and success:
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
            if (target_name in body1 and "hand" in body2) or (
                target_name in body2 and "hand" in body1
            ):
                return True
        return False

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)
        return reward >= self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        is_object_in_container = self.check_object_in_container(
            info=info, *args, **kwargs
        )
        is_object_contacted_by_hand = self.check_hand_object_contact(*args, **kwargs)

        if is_object_in_container and not is_object_contacted_by_hand:
            self.reward += 0.02
        else:
            self.reward = 0.0
        return self.reward

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]

    def decompose(self):
        from simple.datagen.subtask_spec import (
            GraspObjectSpec,
            HeightAdjustSpec,
            LiftSpec,
            LowerSpec,
            MoveEEFToPoseSpec,
            OpenGripperSpec,
            PhaseBreakSpec,
            RetreatSpec,
            StandSpec,
            TurnSpec,
            WalkSpec,
        )

        container_pos = self.container.pose.position
        place_origon = Box(low=[-0.15, -0.12], high=[-0.1, -0.02])
        place_position = container_pos + np.array(
            place_origon.sample() + [_PLACE_HEIGHT]
        )

        import transforms3d as t3d

        place_orientation = t3d.euler.euler2quat(0, 0, 0)

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
                lock_links=["left_hand_palm_link"],
            ),
            LiftSpec("lift", up=0.08, grasp_type="bodex", hand_uid="dex3_right"),
            TurnSpec("turn_90", vx=0.1, target_yaw=np.pi / 2),
            TurnSpec("turn_180", vx=0.1, target_yaw=np.pi),
            WalkSpec(
                "walk_to_table2",
                vx=0.35,
                vy=0.15,
                target_yaw=np.pi,
                target_distance=1.39,
            ),
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

