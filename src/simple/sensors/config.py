"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from simple.core.types import Pose
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass

@dataclass
class SensorCfg:
    uid: str
    mount: str
    # parent_link: str

    @property
    def observation_space(self) -> spaces.Space:
        ...

@dataclass
class CameraCfg(SensorCfg):

    # uid: str 
    
    width: int
    height: int

    fov: float
    near: float
    far: float

    focal_length: float

    pose: dict
    

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)

    @property
    def resolution(self) -> tuple[int, int]: # [W, H]
        return (self.width, self.height)
    
    @property
    def fy(self) -> float:
        fx = self.width / (2 * np.tan(self.fov / 2))
        fy = fx
        return fy
    
    @property
    def position(self) -> list[float]:
        if 'position' in self.pose:
            return self.pose['position']

    @property
    def quaternion(self) -> list[float]:
        if 'quaternion' in self.pose:
            return self.pose['quaternion']

@dataclass
class StereoCameraCfg(CameraCfg):

    baseline: float

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict({
            f"left": spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8),
            f"right": spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)
        })