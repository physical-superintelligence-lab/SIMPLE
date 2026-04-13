"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from .asset_manager import AssetManager
from simple.core.asset import Asset
import os

from simple.core.object import SemanticAnnotated, SpatialAnnotated

from simple.core.types import Pose, GraspPose

from simple.utils import resolve_data_path
import numpy as np

from typing import List, Any

Objects_Names = {
    0: "brasket",
   
}

class ObjectsAsset(Asset, SemanticAnnotated,SpatialAnnotated):
    
    def __init__(
        self, 
        uid: str, 
        label:str,
        name: str, 
        usd_path: str, 
        collision_mesh_curobo: str, 
        collision_meshes_mujoco: list[str],
        description: str | None = None,
        stable_poses: dict[str, Pose] | None = None,
        
    ) -> None:
        super().__init__(uid=uid, usd_path=usd_path, collision_mesh_curobo = collision_mesh_curobo, collision_meshes_mujoco=collision_meshes_mujoco)
        
        # SpatialAnnotated.__init__(self, uid=uid, name=name, usd_path=usd_path, collision_mesh_curobo=collision_mesh_curobo, collision_meshes_mujoco=collision_meshes_mujoco)

        self.name = name
        self.label = label
        self.description = description
        self.stable_poses = stable_poses


    def __repr__(self) -> str:
        return f"Objects1BAsset(uid={self.uid}, name={self.name})"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "objects",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            # "collision_mesh_curobo": self.collision_mesh_curobo,
            # "collision_meshes_mujoco": self.collision_meshes_mujoco,
            "description": self.description,

        }

@AssetManager.register("objects")
class ObjectsAssetManager(AssetManager):

    src_dir: str = "data/assets/objects"

    def __init__(self, src_dir: str | None = None) -> None:
        if src_dir is not None:
            self.src_dir = src_dir

        self.src_dir = "assets/objects"
        # self._load_graspnet_object_assets()

    def load(self, asset_id: str) -> Asset: 
        """Load an asset by its ID."""
        assert int(asset_id) in Objects_Names, f"Asset ID {asset_id} not found in GraspNet 1B."
        
        asset_id_int = int(asset_id)
        collision_mesh_dir = resolve_data_path(
            f"{self.src_dir}/collision_models_mujoco/{asset_id_int:03d}/",
            auto_download=True,
        )

        collision_mesh_curobo = os.path.join(collision_mesh_dir, f"{asset_id_int:03d}_32x8.obj")
        collision_mesh_new_dir = resolve_data_path(
            f'{self.src_dir}/models/{asset_id_int:03d}/coacd_0.05/',
            auto_download=True,
        )
        collision_meshes_mujoco= []
        for dir, dirs, files in os.walk(collision_mesh_new_dir):
            collision_meshes_mujoco = [dir + "/" + file for file in files if file.startswith('convex_piece')]
    
        stable_poses = np.array([[ 4.99879344e-01, -3.56605453e-05,  0.012,
         0.7071, 0.7071, 0, 0]])
        
        name = Objects_Names[asset_id_int]
        return ObjectsAsset(
            uid=asset_id,
            label="_".join(name.split(" ")).replace("-", "_"),
            name=Objects_Names[asset_id_int],
            usd_path=resolve_data_path(
                f"{self.src_dir}/ruled_models/{asset_id_int:03d}_ruled.usd",
                auto_download=True,
            ),
            collision_mesh_curobo=collision_mesh_curobo,
            collision_meshes_mujoco=collision_meshes_mujoco,
            description="todo",
            stable_poses = stable_poses

        )

    def sample(self, exclude: list[str] | None = None) -> Asset:
        import random
        all_ids = [str(i) for i in Objects_Names.keys()]

        if exclude:
            exclude = [str(e) for e in exclude]  
            candidates = [aid for aid in all_ids if aid not in exclude]
            
        else:
            candidates = all_ids
        sampled_id = random.choice(candidates)
        return self.load(sampled_id)
    
    def __len__(self) -> int:
        return len(Objects_Names)

    def _load_objects_assets(self):
        """  
        Returns:
            dict: {model_id: {name, usd_path, collision_mesh_curobo, collision_meshes_mujoco}}
        """
        # FIXME need download the graspnet assets first
        # from simple.constants import GraspNet_1B_Object_Names
        assets_info = dict()
        for model_id in Objects_Names.keys():
            model_usd = resolve_data_path(f"{self.src_dir}/ruled_models/{model_id:03d}_ruled.usd")
            collision_mesh_dir = resolve_data_path(f'{self.src_dir}/collision_models_mujoco/{model_id:03d}/')
            collision_mesh_curobo = os.path.join(collision_mesh_dir, f"{model_id:03d}_32x8.obj")

            collision_mesh_new_dir = resolve_data_path(
                f'{self.src_dir}/models/{model_id:03d}/coacd_0.05/',
                auto_download=True,
            )
            for dir, dirs, files in os.walk(collision_mesh_new_dir):
                collision_meshes_mujoco = [dir + "/" + file for file in files if file.startswith('convex_piece')]

            assets_info[model_id] = {
                'name': Objects_Names[model_id].replace(' ', '_').replace('-', '_'),
                'usd_path': model_usd,
                'collision_mesh_curobo': collision_mesh_curobo,
                'collision_meshes_mujoco': collision_meshes_mujoco,
            }
        return assets_info
    
