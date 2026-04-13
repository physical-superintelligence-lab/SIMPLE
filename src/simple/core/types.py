"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import List, Dict, Union
from dataclasses import dataclass, field
import numpy as np
import transforms3d as t3d
class Vec3:
    data: List[float] = [0, 0, 0]  # x, y, z coordinates

class Vec7:
    data: List[float] = [0, 0, 0, 1, 0, 0, 0]  # x, y, z, qw, qx, qy, qz


# @dataclass
class Pose:
    # position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # x,y,z 
    # quaternion: List[float]  = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0]) # qw,qx,qy,qz

    # # def __init__(self, position: List[float]|None = None, quaternion: List[float]|None = None) -> None:
    # #     if position is not None:
    # #         self.position = position
    # #     if quaternion is not None:
    # #         self.quaternion = quaternion

    position: List[float] = [0.0, 0.0, 0.0] # x,y,z 
    quaternion: List[float]  = [1.0, 0.0, 0.0, 0.0] # qw,qx,qy,qz

    def __init__(self, position: List[float]|None = None, quaternion: List[float]|None = None) -> None:
        if position is not None:
            self.position = position
        if quaternion is not None:
            self.quaternion = quaternion

    def as_matrix(self) -> np.ndarray:
        """  
        Returns 
        a 4x4 transformation matrix
        """
        mat = np.eye(4, dtype=np.float32)
        mat[:3, 3] = np.array(self.position, dtype=np.float32)
        mat[:3, :3] = t3d.quaternions.quat2mat(np.array(self.quaternion, dtype=np.float32))
        return mat
    
    @staticmethod
    def from_vec(vec: List[float]) -> "Pose":
        """ 
        Args:
            vec: a 1D vector representation of the pose. 
                 [x, y, z, qw, qx, qy, qz]
        Returns 
            Pose instance
        """
        position = vec[0:3]
        quaternion = vec[3:7]
        return Pose(position=position, quaternion=quaternion)

    def as_vec(self) -> np.ndarray:
        """ 
        Returns 
            a 1D vector representation of the pose. 
            [x, y, z, qw, qx, qy, qz]
        """
        if type(self.position) == np.ndarray:
            self.position = self.position.tolist()
        return np.array(self.position + self.quaternion, dtype=np.float32)
    
    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "position": self.position,
            "quaternion": self.quaternion
        }
    
    def __mul__(self, other: "Pose") -> "Pose":
        """ 
        Pose multiplication (composition)
        Args:
            other: another Pose instance
        Returns:
            composed Pose instance
        """
        mat1 = self.as_matrix()
        mat2 = other.as_matrix()
        mat_res = mat1 @ mat2
        position_res = mat_res[:3, 3].tolist()
        quaternion_res = t3d.quaternions.mat2quat(mat_res[:3, :3]).tolist()
        return Pose(position=position_res, quaternion=quaternion_res)

class GraspPose:
    position: List[float] = [0.0, 0.0, 0.0] # x,y,z 
    quaternion: List[float]
    width: float
    depth: float