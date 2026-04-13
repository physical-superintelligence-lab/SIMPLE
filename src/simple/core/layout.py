"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simple.core.robot import Robot
    from simple.core.actor import Actor, Light , ArticulatedObjectActor #, Entity, RobotActor
    from simple.core.asset import Asset
    # from simple.core.types import Entity
    from simple.sensors.config import CameraCfg, SensorCfg
    from simple.core.scene import Scene
    # from simple.core.robot import RobotActor

from abc import ABC, abstractmethod
# from simple.core.actor import Actor, Light
# from simple.core.asset import Asset
# from simple.core.types import Pose

from simple.core.actor import *
# from simple.robots.registry import RobotRegistry

from typing import Any
import transforms3d as t3d
import numpy as np

class Layout:

    actors: dict[str, Actor] 

    lights: list[Light] 

    cameras: dict[str, CameraEntity] 

    scene: Scene

    scene_info: dict[str, Any] 

    # visuals: dict[str, VisualEntity] = {}

    def __init__(self):
        self.actors: dict[str, Actor] = {}
        self.lights: list[Light] = []
        self.cameras: dict[str, CameraEntity] = {}
        # self.scene: Scene = None
        self.scene_info: dict[str, Any] = {}
        self.visuals: dict[str, VisualEntity] = {}
        
    # TODO merge following two methods
    def add_object(self, name: str, obj: Asset) -> None:
        if name in self.actors:
            raise ValueError(f"Actor with name '{name}' already exists in the layout.")
        
        self.actors[name] = ObjectActor(asset=obj) #ActorReigstry.make("object", obj) #Actor.from_asset(obj)

    def add_articulated_object(self, name: str, obj: ArticulatedAsset) -> None:
        if name in self.actors:
            raise ValueError(f"Actor with name '{name}' already exists in the layout.")
        
        self.actors[name] = ArticulatedObjectActor(asset=obj) #ActorReigstry.make("articulated_object", obj) #Actor.from_asset(obj)
        
    def add_robot(self, robot: Robot) -> None:
        if "robot" in self.actors:
            raise ValueError("Robot already exists in the layout.")
        
        self.actors["robot"] = RobotActor(robot) #ActorReigstry.make("robot", robot) # Actor.from_robot(robot)

    def add_primitive(self, name: str, actor: Actor) -> None:
        if name in self.actors:
            raise ValueError(f"Actor with name '{name}' already exists in the layout.")
        
        self.actors[name] = actor

    def add_light(self, light: Light) -> None:
        self.lights.append(light)

    def add_camera(self, cam_id: str, cam_cfg: CameraCfg) -> None:
        from simple.sensors.config import SensorCfg, StereoCameraCfg
        if cam_id in self.cameras:
            raise ValueError(f"Camera with ID '{cam_id}' already exists in the layout.")
        
        if isinstance(cam_cfg, StereoCameraCfg):
            stereo_left = CameraEntity(cam_id, cam_cfg) #CameraEntity(cam_id, cam_cfg)
            stereo_right = CameraEntity(cam_id, cam_cfg) #CameraEntity(cam_id, cam_cfg)

            Rwc = t3d.quaternions.quat2mat(stereo_left.pose.quaternion)
            p = np.array(stereo_left.pose.position, dtype=np.float32) + -Rwc[:3, 1] * cam_cfg.baseline # -y axis (right)
            stereo_right.pose.position = p.tolist()

            self.cameras[f"{cam_id}_left"] = stereo_left
            self.cameras[f"{cam_id}_right"] = stereo_right

        elif isinstance(cam_cfg, SensorCfg):
            self.cameras[cam_id] = CameraEntity(cam_id, cam_cfg)
        else:
            raise TypeError(f"Unsupported camera config type: {type(cam_cfg)}")
    
    def add_visual_frame(self, name:str, pose: list[float]):
        p = Pose.from_vec(pose)
        self.visuals[f"debug_{name}"] = VisualFrame(name, p)

    def add_visual_grasp(self, name:str, grasp:dict):
        self.visuals[f"debug_{name}"] = VisualGrasp(name, grasp)

    def remove_visual(self, name: str) -> None:
        self.visuals.pop(f"debug_{name}", None)

    def clear_visuals(self, prefix: str | None = None) -> None:
        if prefix is None:
            self.visuals.clear()
        else:
            debug_prefix = f"debug_{prefix}"
            keys = [k for k in list(self.visuals.keys()) if k.startswith(debug_prefix)]
            for k in keys:
                self.visuals.pop(k, None)
        
    @property
    def robot(self) -> RobotActor:
        """Returns the robot in the layout, if any."""
        assert isinstance(self.actors["robot"], RobotActor)
        return self.actors["robot"]

    def add_scene_info(self, scene_info: dict[str, Any]) -> None:

        self.scene_info = scene_info
    
    # @property
    # def all_entities(self) -> dict[str, Entity]:
    #     """Returns all actors in the layout."""
    #     return self.actors

    # @property
    # @abstractmethod
    # def actors(self) -> dict[str, 'Actor']:

    def to_dict(self) -> dict[str, Any]:
        """Convert the layout to a dictionary representation."""
        test = self.actors["target"].to_dict()  # CHECK
        layout_dict = {
            "actors": {name: actor.to_dict() for name, actor in self.actors.items() if name != "robot"}, # CHECK 
            "lights": [light.to_dict() for light in self.lights],
            "cameras": {cam_id: cam.to_dict() for cam_id, cam in self.cameras.items()},
            "scene": self.scene.to_dict(),
            "scene_info": self.scene_info,
        }
        return layout_dict