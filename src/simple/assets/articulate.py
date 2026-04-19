"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from .asset_manager import AssetManager
from simple.core.asset import ArticulatedAsset
import os
from simple.core.object import SemanticAnnotated, SpatialAnnotated

from simple.core.types import Pose, GraspPose

from simple.utils import resolve_data_path
import numpy as np

from typing import List, Any

Articulated_Object_Names = {
    0: "office_chair",
    1: "bottle",
    2: "oven",
    3: "door",
    4: "faucet",
    5: "trash_can"
}


class ArticulatedObjectAsset(ArticulatedAsset, SemanticAnnotated):
    def __init__(self, 
                 uid: str,
                 label: str,
                 name: str,
                 usd_path: str   | None,
                 mjcf_path: str,
                 description: str | None = None,
                 articulate_init_joint_qpos: dict[str, float] | None = None,
                 ) -> None:
        super().__init__(uid=uid, usd_path=usd_path, mjcf_path=mjcf_path)
        self.label = label
        self.name = name
        self.description = description
        self.articulate_init_joint_qpos = articulate_init_joint_qpos
    
    def __repr__(self) -> str:
        return f"ArticulatedObjectAsset(uid={self.uid}, name={self.name})"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "articulated",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            "mjcf_path": self.mjcf_path,
            "description": self.description,
            "articulate_init_joint_qpos": self.articulate_init_joint_qpos,
        }

@AssetManager.register("articulated")
class ArticulatedAssetManager(AssetManager):
    src_dir: str = "data/assets/articulated"

    def __init__(self, src_dir: str | None = None) -> None:
        if src_dir is not None:
            self.src_dir = src_dir
        self.src_dir = "assets/articulated"
    

    def load(self, asset_id: str, articulate_init_joint_qpos :dict[str, float] | None = None) -> ArticulatedAsset: 
        """Load an asset by its ID."""
        assert int(asset_id) in Articulated_Object_Names, f"Invalid asset ID: {asset_id}"

        asset_id_int = int(asset_id)

        mjcf_path = resolve_data_path(os.path.join(self.src_dir, f"{asset_id_int:03d}", "output_mjcf", f"{asset_id_int:03d}.xml")  ,auto_download=True)

        #TODO: Load usd_path
        usd_path = resolve_data_path(os.path.join(self.src_dir, f"{asset_id_int:03d}", "output_usd", f"{asset_id_int:03d}.usd"), auto_download=True)

        name = Articulated_Object_Names[asset_id_int]
        return ArticulatedObjectAsset(
            uid=asset_id,
            label="_".join(name.split(" ")).replace("-", "_"),
            name=name,
            usd_path=usd_path,
            mjcf_path=mjcf_path,
            articulate_init_joint_qpos=articulate_init_joint_qpos
        )
    
    def sample(self, exclude: list[str] | None = None) -> ArticulatedAsset:
        import random
        all_ids = [str(i) for i in Articulated_Object_Names.keys()]

        if exclude is not None:
            all_ids = [id for id in all_ids if id not in exclude]
       
        asset_id = random.choice(all_ids)
        return self.load(asset_id)
    
    def __len__(self) -> int:
        return len(Articulated_Object_Names)
    
    
