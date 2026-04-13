"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import random
from simple.core.actor import RobotActor , ObjectActor, ArticulatedObjectActor
from simple.core.randomizer import Randomizer, RandomizerCfg
from typing import TYPE_CHECKING, List
# if TYPE_CHECKING:
from simple.core.layout import Layout
from simple.core.object import Object, SpatialAnnotated

from .types import Box
import numpy as np
import trimesh
import transforms3d as t3d
from typing import Type, Any

from dataclasses import dataclass


     

    # def __init__(
    #     self, 
    #     robot_region: Box | None = None, 
    #     target_region: Box | None = None, 
    #     distractors_region: Box | None = None,
    #     rand_stable_pose: bool = True,
    #     # rand_grasp_psoe: bool = False
    #     collide_threshold: float = 0.005
    # ) -> None:
        
    #     self.robot_region = robot_region
    #     self.target_region = target_region
    #     self.distractors_region = distractors_region

    #     self.rand_stable_pose = rand_stable_pose
    #     # self.rand_grasp_pose = rand_grasp_psoe
    #     self.collide_threshold = collide_threshold


class SpatialDR(Randomizer):

    cfg: SpatialDRCfg # type: ignore

    def __init__(
        self, 
        # robot_region: Box | None = None, 
        # target_region: Box | None = None, 
        # distractors_region: Box | None = None,
        # rand_stable_pose: bool = True,
        # # rand_grasp_psoe: bool = False
        # collide_threshold: float = 0.005
        cfg: SpatialDRCfg
    ) -> None:
        super().__init__(cfg)
        # self.robot_region = robot_region
        # self.target_region = target_region
        # self.distractors_region = distractors_region

        # self.rand_stable_pose = rand_stable_pose
        # # self.rand_grasp_pose = rand_grasp_psoe
        # self.collide_threshold = collide_threshold
        # self.cfg = cfg
        self.spatial_mode = cfg.spatial_mode
        self.fixed_stable_pose_idx = cfg.fixed_stable_pose_idx
        # self._inner_state = {} # FIXME

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict()
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return super().load_state_dict(state_dict)

    def __call__(
        self, 
        split: str, 
        layout: Layout,
        table_height: float = 0.0
    ) -> None:
        """  Randomize the spatial arrangement of robot and objects in the layout.
        
        Args:
            split (str): The data split, e.g., "train", "val", "test".
            layout (Layout): The layout containing the robot and objects to be arranged.
            table_height (float): The height of the table surface. 
            
            HACK: we always has ground plane at z=0, and objects are on the GROUND plane.
            So table_height is only used to adjust the robot base height.
        
        """
        self.collision_manager= trimesh.collision.CollisionManager()

        # self._inner_state = {} # keep track of placed objects
        if self._inner_state is None:
            self._inner_state = {}

        # Determine surface heights for different regions
        surface_heights = {"default": table_height}
        if hasattr(layout.scene, "table") and layout.scene.table is not None:
            surface_heights["table"] = layout.scene.table.pose.position[2] + 0.5 * layout.scene.table.size[2]
        
        if hasattr(layout.scene, "table2") and layout.scene.table2 is not None:
            surface_heights["table2"] = layout.scene.table2.pose.position[2] + 0.5 * layout.scene.table2.size[2]

        distractor_count = 0
        for objtype, obj in layout.actors.items():
            if objtype == "articulated":
                assert isinstance(obj, ArticulatedObjectActor)
                if self._inner_state is not None and f"articulated_{str(obj.uid)}" in self._inner_state:
                    state = self._inner_state[f"articulated_{str(obj.uid)}"]
                    obj.pose.position = state["position"]
                    obj.pose.quaternion = state["quaternion"]
                else:
                    if self.cfg.articulated_region is not None:
                        if self.spatial_mode == "fixed":
                            position = self.cfg.articulated_region.middle()

                            rand_z = self.cfg.articulated_rotate_z.sample()
                            rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, rand_z)
                            obj.pose.position = position
                            obj.pose.quaternion = t3d.quaternions.mat2quat(rand_ori_mat).tolist()

                            self._inner_state[f"articulated_{str(obj.uid)}"] = {"position": position, "quaternion": obj.pose.quaternion}

                        else:
                            position = self.cfg.articulated_region.sample()
                            rand_z = self.cfg.articulated_rotate_z.sample()
                            rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, rand_z)
                            surface_height = surface_heights.get(surface_name, surface_heights["default"])
                            if len(position) == 2:
                                position.append(0.0)
                                position[2] = surface_height
                            
                            obj.pose.position = position
                            obj.pose.quaternion = t3d.quaternions.mat2quat(rand_ori_mat).tolist()

                            self._inner_state[f"articulated_{str(obj.uid)}"] = {"position": position, "quaternion": obj.pose.quaternion}
                       
                            

            if objtype == "robot":
                assert isinstance(obj, RobotActor)
                if self._inner_state is not None and str(obj.robot.uid) in self._inner_state:
                    state = self._inner_state[str(obj.robot.uid)]
                    robot_position = state["position"]
                
                else:
                    if self.cfg.robot_region is not None:
                        if self.spatial_mode == "fixed":
                            robot_position = self.cfg.robot_region.middle()
                        else:
                            robot_position = self.cfg.robot_region.sample()
                        if len(robot_position) == 2:
                            assert isinstance(robot_position, list)
                            # obj.pose.position = [robot_position[0], robot_position[1], 0.0]
                            # robot_position = obj.pose.position
                            robot_position.append(0.0)
                            robot_position[2] = - table_height
                        else:
                            robot_position = robot_position
                    else:
                        robot_position = [0.0, 0.0, 0.0]
                    # print(f"Placed robot at {robot_position}")
                obj.pose.position = robot_position

                if self.cfg.robot_orientation_region is not None:
                    if self.spatial_mode == "fixed":
                        robot_orientation = self.cfg.robot_orientation_region.middle()
                    else:
                        robot_orientation = self.cfg.robot_orientation_region.sample()
                    obj.pose.quaternion = robot_orientation
                self._inner_state[obj.robot.uid] = {
                    "position": robot_position,
                    "quaternion": obj.pose.quaternion
                }
            elif objtype == "container":
                if self.cfg.rand_stable_pose and isinstance(obj.asset, SpatialAnnotated):
                    if self._inner_state is not None and f"container_{obj.uid}" in self._inner_state:
                        state = self._inner_state[f"container_{obj.uid}" ]
                        obj.pose.position = state["position"]
                        obj.pose.quaternion = state["quaternion"]
                    else:
                        surface_name = self.cfg.obj_surface_map.get("container", "table") if self.cfg.obj_surface_map else "table"
                        surface_height = surface_heights.get(surface_name, surface_heights["default"])
                        self._random_place_one_object(obj, self.cfg.container_region, objtype, surface_height=surface_height)
            elif objtype == "target": # isinstance(obj, Object):
                # if self.cfg.target_region is not None:
                #     #TODO Randomize target position within the specified region
                #     pass
                # # obj.pose.position = [0.3, 0.0, 0.0]
                
                if self.cfg.rand_stable_pose and isinstance(obj.asset, SpatialAnnotated):
                    if self._inner_state is not None and str(obj.uid) in self._inner_state:
                        state = self._inner_state[str(obj.uid)]
                        obj.pose.position = state["position"]
                        obj.pose.quaternion = state["quaternion"]
                    else:
                        surface_name = self.cfg.obj_surface_map.get("target", "table") if self.cfg.obj_surface_map else "table"
                        surface_height = surface_heights.get(surface_name, surface_heights["default"])
                        self._random_place_one_object(obj, self.cfg.target_region , objtype, surface_height=surface_height)
                        # self._inner_state[obj.uid] = {
                        #     "position": [-0.4, 0.0, 0.04],
                        #     "quaternion": [1,0,0,0]
                        # }


            elif  "distractor" in objtype: 
                if self.cfg.distractors_region is not None:
                    pass

                if self.cfg.rand_stable_pose and isinstance(obj.asset, SpatialAnnotated):
                    if self._inner_state is not None and str(obj.uid) in self._inner_state:
                        state = self._inner_state[str(obj.uid)]
                        obj.pose.position = state["position"]
                        obj.pose.quaternion = state["quaternion"]
                    else:
                        surface_entry = self.cfg.obj_surface_map.get(objtype) or self.cfg.obj_surface_map.get("distractor", "table") if self.cfg.obj_surface_map else "table"
                        if isinstance(surface_entry, list):
                            surface_name = surface_entry[distractor_count % len(surface_entry)]
                        else:
                            surface_name = surface_entry
                        
                        surface_height = surface_heights.get(surface_name, surface_heights["default"])

                        region = self.cfg.distractors_region
                        if isinstance(region, list):
                            region = region[distractor_count % len(region)]

                        self._random_place_one_object(obj, region , objtype, surface_height=surface_height)
                        distractor_count += 1


    def _random_place_one_object(self, obj: Object, region: Box, objtype: str, surface_height: float = 0.0):
        
        object_msh=trimesh.load_mesh(obj.asset.collision_mesh_curobo)
        for _ in range(100):
            if objtype == "container":
                stable_pose = obj.asset.stable_poses[0] # only use the first stable pose for container
            elif self.spatial_mode == "fixed":
                stable_pose = obj.asset.stable_poses[self.fixed_stable_pose_idx]
            else:
                # stable_pose = random.choice(obj.asset.stable_poses)
                if self.cfg.target_stable_indices is not None:
                    stable_pose = obj.asset.stable_poses[random.choice(self.cfg.target_stable_indices)]
                else:
                    stable_pose = random.choice(obj.asset.stable_poses)
            p = np.zeros((3,), dtype=np.float32)
            p[:2] += np.asarray(region.sample(), dtype=np.float32) # random xy
            p[2] = stable_pose[2] + surface_height
            obj.pose.position = p.tolist()
            # print(f"Trying to place object {obj.uid} at position {p.tolist()}")
            
            stable_ori_mat = t3d.quaternions.quat2mat(stable_pose[3:]) # 3x3
            if objtype == "target" :
            
                rand_z = self.cfg.target_rotate_z.sample()
               
                rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, rand_z)

            elif objtype == "container" and self.cfg.container_rotate_z is not None:
                rand_z = self.cfg.container_rotate_z.sample()
                rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, rand_z)
            else:
                rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, random.uniform(0.0, 2.0 * np.pi))
            rand_ori_mat = rand_ori_mat @ stable_ori_mat

            transformation = np.eye(4)
            transformation[:3, :3] = rand_ori_mat
            transformation[:3, 3] = p
            distance=(self.collision_manager.min_distance_single(object_msh, transformation))
            if distance >self.cfg.collide_threshold: # type: ignore
                self.collision_manager.add_object(obj.uid, object_msh, transformation)
                # obj.pose.quaternion = [0.6013860816541081, 9.584801678524222e-05, -0.0020859967168959104, -0.7989558312094444 ] # FIXME #t3d.quaternions.mat2quat(rand_ori_mat).tolist()
                obj.pose.quaternion = t3d.quaternions.mat2quat(rand_ori_mat).tolist()
                if objtype == "container":
                    self._inner_state[f"container_{obj.uid}"] = {
                        "position": obj.pose.position,
                        "quaternion": obj.pose.quaternion
                    }
                else:
                    self._inner_state[obj.uid] = {
                        "position": obj.pose.position,
                        "quaternion": obj.pose.quaternion
                    }
                return True
        print(f"Warning: Failed to place object {obj.uid} without collision after 100 attempts.")
        return None
            

@dataclass
class SpatialDRCfg(RandomizerCfg):
    robot_region: Box | None = None
    robot_orientation_region: Box | None = None
        
    target_region: Box | None = None
    container_region: Box | None = None
    distractors_region: Box | list[Box] | None = None
    rand_stable_pose: bool = True
    collide_threshold: float = 0.005
    spatial_mode: str = "random" # fixed, random
    fixed_stable_pose_idx: int = 0
    target_stable_indices: list[int] | None = None
    target_rotate_z: Box = Box(low=0.0, high=2.0 * np.pi)

    articulated_region: Box | None = None
    articulated_rotate_z: Box | None = None

    container_rotate_z: Box | None = None
    # container_rotate_z: Box = Box(low=0.0, high=0.0)
    obj_surface_map: dict[str, str | list[str]] | None = None
    randmizer_class: Type[Randomizer] = SpatialDR
