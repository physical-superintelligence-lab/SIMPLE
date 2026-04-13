"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
import builtins

from .types import Pose
import numpy as np
import transforms3d as t3d

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # from .object import Object
    from .asset import Asset
    from .asset import ArticulatedAsset
    # from .actor import Actor
    from .robot import Robot
    from simple.sensors.config import CameraCfg

class Entity:
    uid: str
    pose: Pose

    def to_dict(self):
        def _convert(obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            elif hasattr(obj, "__dict__"):
                return {k: _convert(v) for k, v in vars(obj).items()}
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(_convert(v) for v in obj)
            else:
                return obj

        # return _convert(self)

        result = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):   # skip private attrs
                result[k] = v
            if getattr(type(v), "__module__", None) != "builtins":
                result[k] = _convert(v)
                # if hasattr(v, "to_dict"):
                #     result[k] = v.to_dict()
                # elif hasattr(v, "state_dict"):
                #     result[k] = v.state_dict()

        return result

class Actor(Entity):
    
    # pose: Pose
    ...

    # @classmethod
    # def from_asset(cls, asset: Asset) -> Actor:
    #     from .object import Object # prevent circular import
        
    #     """Create an Actor instance from an Asset."""
    #     actor = Object(asset)
    #     actor.pose = Pose()
    #     return actor
    

    # @classmethod
    # def from_robot(cls, robot: Robot) -> Actor:
    #     """Create an Actor instance from a Robot."""
    #     from .robot import Robot
    #     actor = RobotActor(robot)
    #     actor.pose = Pose()
    #     return actor

from typing import ClassVar, Type, Generic
from simple.core.registry import RegistryMixin

from typing import Type, TypeVar
T = TypeVar("T", bound=Actor)

class ActorReigstry(Generic[T]): # RegistryMixin[Actor]

    _registry: dict[str, Type[T]] = {}


    @classmethod
    def register(cls, name: str):
        def wrapper(actor_cls: Type[T]) -> Type[T]:
            cls._registry[name] = actor_cls
            return actor_cls
        return wrapper
    
    # @classmethod
    # def _base_type(cls) -> Type:
    #     return Actor

    @classmethod
    def make(cls, name: str, *args, **kwargs) -> T:
        if name not in cls._registry:
            raise ValueError(f"No class registered under name '{name}'")

        # if name not in cls._instances:
        #     cls._instances[name] = cls._registry[name](*args, **kwargs)

        # return cls._instances[name]

        return cls._registry[name](*args, **kwargs)

# @ActorReigstry.register("object")
class ObjectActor(Actor):
    """Actor representing an object in the simulation."""
    asset: Asset
    material: dict 

    def __init__(self, asset: Asset, uid:str|None=None) -> None:
        self.uid = asset.uid if uid is None else uid
        self.asset = asset
        self.pose = Pose()

    def set_material(self, material: dict) -> None:
        self.material = material

class ArticulatedObjectActor(Actor):
    """Actor representing an object in the simulation."""
    asset: ArticulatedAsset
    material: dict
    def __init__(self, asset: ArticulatedAsset, uid:str|None=None) -> None:
        self.uid = asset.uid if uid is None else uid
        self.asset = asset
        self.pose = Pose()
    
    def set_material(self, material: dict) -> None:
        self.material = material
    
# @ActorReigstry.register("robot")
class RobotActor(Actor):
    """Actor representing a robot in the simulation."""
    robot: Robot

    shaders: dict[str, float]

    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.pose = Pose()
    
    def set_shaders(self, shaders):
        self.shaders = shaders

class VisualEntity(Entity):
    """ For visualization purpose only. """
    ...

class VisualFrame(VisualEntity):
    def __init__(self, uid: str, pose: Pose) -> None:
        self.uid = uid
        self.pose = pose

class VisualGrasp(VisualEntity):

    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    def __init__(self, uid: str, grasp: dict) -> None:
        self.uid = uid
        self.pose = Pose()
        self.pose.position = grasp["position"]
        self.pose.quaternion = grasp["orientation"]
        self.grasp = grasp

    def plot_lines(self):
        origin = np.array(self.pose.position, dtype=np.float32)
        rotation_mat = t3d.quaternions.quat2mat(self.pose.quaternion)
        x_axis = rotation_mat[:3, 0]
        y_axis = rotation_mat[:3, 1]
        # z_axis = rotation_mat[:3, 2]

        tail_start = origin - x_axis * (self.tail_length + self.depth_base)
        tail_end = origin - x_axis * self.depth_base

        grasp_width = 0.08 # self.grasp.get("width", 0.1)
        bottom_start = origin - y_axis * (grasp_width / 2) - x_axis * self.depth_base
        bottom_end = bottom_start + y_axis * grasp_width

        gripper_depth = self.grasp.get("depth", 0.1)
        gripper_left_start = bottom_start
        gripper_left_end = bottom_start + x_axis * (self.depth_base + gripper_depth)

        gripper_right_start = bottom_end
        gripper_right_end = bottom_end + x_axis * (self.depth_base + gripper_depth)
        return [
            # start point, end point, color (r,g,b,a), line width
            (tail_start, tail_end, (1, 0, 0, 1), 4.0),
            (bottom_start, bottom_end, (0, 1, 0, 1), 4.0),
            (gripper_left_start, gripper_left_end, (0, 0, 1, 1), 4.0),
            (gripper_right_start, gripper_right_end, (0, 0, 1, 1), 4.0),
        ]

class Light(Entity):
    
    type: str

    light_radius: float
    light_length: float
    light_intensity: float
    light_color_temperature: float

    center_light_postion: list[float] # TODO remove this thing! light center position 
    center_light_orientation: list[float]

    def __init__(self, uid: str, type: str) -> None:
        self.uid = uid
        self.type = type




class CameraEntity(Entity):

    cam_id: str
    mount: str
    position: list[float]
    orientation: list[float]
    resolution: tuple[int, int]
    fx: float
    fy: float
    cx: float
    cy: float
    focal_length: float
    baseline: float | None = None  # Only for stereo cameras
    
    cam_cfg: CameraCfg

    def __init__(self, cam_id: str, cam_cfg: CameraCfg) -> None:
        self.cam_id = cam_id

        self.cam_cfg = cam_cfg

        # TODO better way to handle pose
        if "distance" in cam_cfg.pose: # spherical pose specification
            r = cam_cfg.pose["distance"]
            theta = cam_cfg.pose["polar"]
            phi = cam_cfg.pose["azimuth"]

            camera_look_at = np.array([0.0, 0.0, 0.0]) # TODO configure this

            # https://en.wikipedia.org/wiki/Spherical_coordinate_system
            camera_pos = camera_look_at + np.array([
                r * np.cos(phi) * np.sin(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(theta),
            ])
            directiron_vec = camera_look_at - camera_pos
            roll = 0.0
            pitch = -np.arctan2(directiron_vec[2], np.linalg.norm(directiron_vec[:2]))
            yaw = np.arctan2(directiron_vec[1], directiron_vec[0])
            quat = t3d.euler.euler2quat(roll, pitch, yaw)
            # W, H = (640, 360)
            self.resolution = cam_cfg.resolution
            self.focal_length = cam_cfg.focal_length
            position = camera_pos.tolist()
            quaternion = quat.tolist()

            
        elif "position" in cam_cfg.pose and "quaternion" in cam_cfg.pose:
            position = cam_cfg.pose["position"]
            quaternion = cam_cfg.pose["quaternion"]
        elif "position" in cam_cfg.pose and "eulers" in cam_cfg.pose:
            position = cam_cfg.pose["position"]
            quaternion = t3d.euler.euler2quat(*cam_cfg.pose["eulers"]).tolist()
        else:
            raise ValueError("invalid camera pose specification")

        self.pose = Pose(position=position, quaternion=quaternion)

        h = 2 * cam_cfg.focal_length * np.tan(cam_cfg.fov/2)
        self.fy = self.fx = cam_cfg.width * cam_cfg.focal_length  / h
        self.cx = 0.5 * cam_cfg.width
        self.cy = 0.5 * cam_cfg.height
        self.mount = cam_cfg.mount
        self.resolution = cam_cfg.resolution
        self.focal_length = cam_cfg.focal_length

        # self.height = cam_cfg.height
        # self.width = cam_cfg.width
  