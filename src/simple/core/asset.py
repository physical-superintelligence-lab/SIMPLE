"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from dataclasses import dataclass
from typing import Any

class Asset:

    uid: str
    # name: str
    usd_path: str # TODO move to sub asset classes
    collision_mesh_curobo: str 
    collision_meshes_mujoco: list[str] 

    def __init__(
        self, 
        uid: str, # name: str, 
        usd_path: str,
        collision_mesh_curobo: str,
        collision_meshes_mujoco: list[str]
    ) -> None:
        self.uid = uid
        # self.name = name
        self.usd_path = usd_path
        self.collision_mesh_curobo = collision_mesh_curobo
        self.collision_meshes_mujoco = collision_meshes_mujoco
    
    # def to_dict(self) -> dict[str, Any]:
    #     pass

class ArticulatedAsset:
    uid: str
    # name: str
    usd_path: str   | None
    mjcf_path: str
    
    def __init__(self, uid: str, usd_path: str | None, mjcf_path: str) -> None:
        self.uid = uid
        self.usd_path = usd_path
        self.mjcf_path = mjcf_path


