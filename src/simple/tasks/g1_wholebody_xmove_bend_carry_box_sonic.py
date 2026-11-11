# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from simple.core.randomizer import RandomizerCfg
    from simple.datagen.subtask_spec import SubtaskSpec

import os
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

_LIFT_HEIGHT = 0.1
_PLACE_HEIGHT = 0.5
_LOWER_HEIGHT = 0.1


@TaskRegistry.register("g1_wholebody_xmove_bend_carry_box_sonic")
class G1WholebodyXMoveBendCarryBoxTaskSonic(Task):
    uid: str = "g1_wholebody_xmove_bend_carry_box_sonic"
    label: str = "G1 SONIC XMove Bend Carry Box"
    description: str = (
        "A task where the G1 robot must xmove to bend and pick up an object ."
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
            # fov is HORIZONTAL here (fx = W / 2tan(fov/2), fy = fx). Set to
            # 110° to match the ego view of the training set rather than the
            # D435i depth spec (87° x 58°, i.e. 87° x 56.2° at 16:9 under the
            # square-pixel fx=fy approximation).
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
                "xmove to the table and bend to pick up the {}",
            ]
        ),
        # Big square box (movable primitive), no mesh/grasp files needed.
        # Box: 0.23 (depth x) x 0.38 (width y) x 0.434 (height z) m.
        target=TargetDRCfg(asset_id="primitive:cube", size=[0.23, 0.38, 0.434]),
        # No distractors for this variant.
        spatial=SpatialDRCfg(
            # Deterministic (non-randomized) robot + box spawn: fixed mode uses
            # robot_region.middle(), and a single-point target_region + zero yaw
            # pins the box (its xy/yaw are otherwise sampled even in fixed mode).
            # spatial_mode="fixed",
            # Robot back far enough to clear the tall box at spawn, but short of
            # the HSSD room wall/furniture (~-1.5) that shoves it forward.
            robot_region=Box(low=[-1.2, -0.0, 0.0], high=[-1.2, 0.0, 0.0]),
            # Box on the floor just outside the table near edge (x=-0.325): with
            # half-depth 0.115, center -0.55 -> near face -0.435 (clear), back -0.665.
            target_region=Box(low=[-0.5, -0.05], high=[-0.6, +0.05]),
            distractors_region=Box(low=[-0.2, -0.3], high=[0.4, 0.3]),
            target_stable_indices=[0],
            target_rotate_z=Box(low=0.0, high=0.0),
            # Spawn the box on the floor (z=0), not the table, so it is picked
            # up from the ground and placed onto the table.
            obj_surface_map={"target": "ground"},
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
            table_height=Box(low=0.45, high=0.45),
            room_choices=["hssd:scene0"],
            scene_manager="hssd",
            # Disable room orientation randomization (the ~+-15 deg yaw from the
            # scene config); keep the room/table pose fixed every episode.
            randomize_scene_pose=False,
        ),
        lighting=LightingDRCfg(
            light_mode="random",  # fixed, random
            light_num=(2, 3),
            light_color_temperature=Box(low=6001, high=8001),  # I was not joking :)
            light_intensity=Box(low=1e4*0.8, high=1e4*1.2),
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
        scene_uid: str | Scene = "hssd:scene0",
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

    def _target_contacts(self, mujoco_env, name_substr: str) -> bool:
        """Whether the box is in contact with any body whose name contains
        ``name_substr`` (e.g. "hand")."""
        target_name = str(self.target.asset.label)
        mj_data = mujoco_env.mjData
        mj_model = mujoco_env.mjModel
        for i_contact in range(mj_data.ncon):
            contact = mj_data.contact[i_contact]
            body1 = mj_model.body(mj_model.geom(contact.geom1).bodyid.item()).name
            body2 = mj_model.body(mj_model.geom(contact.geom2).bodyid.item()).name
            if (target_name in body1 and name_substr in body2) or (
                target_name in body2 and name_substr in body1
            ):
                return True
        return False

    def _target_on_table_top(self, mujoco_env, table_top: float, tol: float = 0.05) -> bool:
        """Whether the box contacts the table's UPPER surface.

        A box<->table contact whose contact point sits at the table top height,
        so a side/edge contact lower down does not count.
        """
        target_name = str(self.target.asset.label)
        mj_data = mujoco_env.mjData
        mj_model = mujoco_env.mjModel
        for i_contact in range(mj_data.ncon):
            contact = mj_data.contact[i_contact]
            body1 = mj_model.body(mj_model.geom(contact.geom1).bodyid.item()).name
            body2 = mj_model.body(mj_model.geom(contact.geom2).bodyid.item()).name
            is_box_table = (target_name in body1 and "table" in body2) or (
                target_name in body2 and "table" in body1
            )
            if is_box_table and abs(float(contact.pos[2]) - table_top) <= tol:
                return True
        return False

    def _box_placed_on_table(self, info: dict[str, Any], *args, **kwargs) -> bool:
        """Success when all three hold:
          1. the robot's hands are off the box (no hand contact),
          2. the box is in contact with the table's upper surface, and
          3. the box center of mass is above the table top surface.
        """
        mujoco_env = kwargs.get("mujoco_env", None)
        table = self.layout.actors.get("table")
        if mujoco_env is None or table is None or "target" not in info:
            return False

        table_top = table.pose.position[2]

        hands_off = not self._target_contacts(mujoco_env, "hand")
        on_table_top = self._target_on_table_top(mujoco_env, table_top)
        com_above_table = float(info["target"][2]) >= table_top
        # print(f"[Task] hands_off={hands_off}, on_table_top={on_table_top}, com_above_table={com_above_table}")
        return hands_off and on_table_top and com_above_table

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self.reward > self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        self.reward += 0.02 if self._box_placed_on_table(info, *args, **kwargs) else 0.0
        return self.reward

    def _log_state(self, info: dict[str, Any], mujoco_env) -> None:
        """Print live item state + the 3 success conditions (~2 Hz).

        Enabled by setting the env var XBENDPICK_DEBUG=1 when launching teleop.
        """
        self._dbg_ctr = getattr(self, "_dbg_ctr", 0) + 1
        if self._dbg_ctr % 25 != 0:  # compute_reward runs ~50 Hz -> print ~2 Hz
            return
        table = self.layout.actors.get("table")
        if mujoco_env is None or table is None or "target" not in info:
            print(f"[xbendpick] waiting: mujoco_env={mujoco_env is not None} "
                  f"table={table is not None} target_in_info={'target' in info}")
            return
        table_top = table.pose.position[2] + 0.5 * table.size[2]
        box_xyz = np.round(np.asarray(info["target"][:3], dtype=float), 3).tolist()
        hands_off = not self._target_contacts(mujoco_env, "hand")
        on_table_top = self._target_on_table_top(mujoco_env, table_top)
        com_above = float(info["target"][2]) > table_top
        print(
            f"[xbendpick] box_xyz={box_xyz} table_top={table_top:.3f} "
            f"| hands_off={hands_off} on_table_top={on_table_top} com_above={com_above} "
            f"success={hands_off and on_table_top and com_above}"
        )

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
