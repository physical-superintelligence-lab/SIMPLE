"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from .asset_manager import AssetManager
from simple.core.asset import Asset
from simple.core.actor import Actor
# import os
from simple.core.types import Pose

class Primitive(Asset, Actor):
    material: dict


class Box(Primitive):
    def __init__(self, size, position, quaternion) -> None:
        self.uid = "box"

        self.size = size
        # self.position = position
        # self.quaternion = quaternion
        self.pose = Pose(position, quaternion)

    def set_material(self, material: dict) -> None:
        self.material = material

    def __repr__(self) -> str:
        return f"Box(size={self.size}, position={self.pose.position}, quaternion={self.pose.quaternion})"

@AssetManager.register("primitive")
class PrimitiveAsseManager(AssetManager):
    
    def __init__(self) -> None:
        ...

    def load(self, asset_id: str, *args, **kwargs) -> Asset: 
        if asset_id == "box":
            return Box(*args, **kwargs)
        else:
            raise ValueError(f"Unknown primitive asset_id: {asset_id}")

    def sample(self, exclude: list[str] | None = None) -> Asset:
        ...

    # def box(self, size, position, height, quaternion) -> Asset:
    #     return Box(size, position, height, quaternion)
    