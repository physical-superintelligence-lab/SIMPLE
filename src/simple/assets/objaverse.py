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

import glob
import json


#TODO make objaverse names clear
# len_objaverse =  glob.glob("data/assets/objaverse/*")
# if len(len_objaverse) == 0:
#     from huggingface_hub import snapshot_download
#     local_data_dir = resolve_data_path()
#     snapshot_download(
#         repo_id="SIMPLE-org/SIMPLE",
#         allow_patterns=["assets.zip"],
#         local_dir=local_data_dir,
#         repo_type="dataset",
#         # resume_download=True,
#         token="YOUR_HF_TOKEN",
#     )
#     import zipfile
#     zip_path = os.path.join(local_data_dir, "assets.zip")
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(local_data_dir)
#     if os.path.exists(zip_path):
#         os.remove(zip_path)
#         print(f"Deleted {zip_path}")
#     len_objaverse = glob.glob("data/assets/objaverse/*")
len_objaverse = 1545
    
Objaverse_Names = {i:f"obj_{i:04d}" for i in range(len_objaverse)}

# for i in range(len(len_objaverse)):
#     Objaverse_Names[i] = f"{i:04d}"

#Objaverse_Names = {

class ObjaverseAsset(Asset, SemanticAnnotated,SpatialAnnotated):
    
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
        return f"ObjaverseAsset(uid={self.uid}, name={self.name})"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "objaverse",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            # "collision_mesh_curobo": self.collision_mesh_curobo,
            # "collision_meshes_mujoco": self.collision_meshes_mujoco,
            "description": self.description,

        }

@AssetManager.register("objaverse")
class ObjaverseAssetManager(AssetManager):

    src_dir: str = "data/assets/objaverse"

    def __init__(self, src_dir: str | None = None) -> None:
        if src_dir is not None:
            self.src_dir = src_dir

        self.src_dir = "assets/objaverse"
        # self._load_graspnet_object_assets()
        
        
    
    def load(self, asset_id: str) -> Asset: 
        """Load an asset by its ID."""
        assert int(asset_id) in Objaverse_Names, f"Asset ID {asset_id} not found in Objaverse."
        
        asset_id_int = int(asset_id)
        local_data_dir = resolve_data_path(f'{self.src_dir}/{asset_id_int:04d}',auto_download=True)
        collision_mesh_curobo = f"data/{self.src_dir}/{asset_id_int:04d}/mesh/normalized.obj"
        
        collision_mesh_new_dir = resolve_data_path(f'{self.src_dir}/{asset_id_int:04d}/urdf/meshes')
        collision_meshes_mujoco= []
        for dir, dirs, files in os.walk(collision_mesh_new_dir):
            collision_meshes_mujoco = [dir + "/" + file for file in files if file.startswith('convex_piece')]

        stable_poses = np.array(json.load(open(f"data/{self.src_dir}/{asset_id_int:04d}/info/tabletop_pose.json", "r")))
    
        
        name = Objaverse_Names[asset_id_int]
        return ObjaverseAsset(
            uid=asset_id,
            label="_".join(name.split(" ")).replace("-", "_"),
            name=Objaverse_Names[asset_id_int],
            usd_path=f"assets/objaverse/{asset_id_int:04d}/mesh/normalized_isaac.usd",
            collision_mesh_curobo=collision_mesh_curobo,
            collision_meshes_mujoco=collision_meshes_mujoco,
            description="todo",
            stable_poses = stable_poses

        )



    def sample(self, exclude: list[str] | None = None) -> Asset:
        import random
        all_ids = [str(i) for i in Objaverse_Names.keys()]

        if exclude:
            exclude = [str(e) for e in exclude]  
            candidates = [aid for aid in all_ids if aid not in exclude]
            
        else:
            candidates = all_ids
        sampled_id = random.choice(candidates)
        return self.load(sampled_id)
    
    def __len__(self) -> int:
        return len(Objaverse_Names)

    def _load_objaverse_assets(self):
        """  
        Returns:
            dict: {model_id: {name, usd_path, collision_mesh_curobo, collision_meshes_mujoco}}
        """
        # FIXME need download the graspnet assets first
        # from simple.constants import GraspNet_1B_Object_Names
        assets_info = dict()
        for model_id in Objaverse_Names.keys():
            # model_usd = resolve_data_path(f"{self.src_dir}/ruled_models/{model_id:03d}_ruled.usd")
            model_usd = resolve_data_path(f"assets/objaverse/{model_id:04d}/mesh/normalized_isaac.usd")
            collision_mesh_dir = resolve_data_path(f'{self.src_dir}/{model_id:04d}/urdf/meshes')
            collision_mesh_curobo = f"data/{self.src_dir}/{model_id:04d}/mesh/normalized.obj"
            
            collision_mesh_new_dir = resolve_data_path(f'{self.src_dir}/{model_id:04d}/urdf/meshes')
            # collision_meshes_mujoco= []
            for dir, dirs, files in os.walk(collision_mesh_new_dir):
                collision_meshes_mujoco = [dir + "/" + file for file in files if file.startswith('convex_piece')]

            assets_info[model_id] = {
                'name': Objaverse_Names[model_id].replace(' ', '_').replace('-', '_'),
                'usd_path': model_usd,
                'collision_mesh_curobo': collision_mesh_curobo,
                'collision_meshes_mujoco': collision_meshes_mujoco,
            }
        return assets_info
    
