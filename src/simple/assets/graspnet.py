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

GraspNet_1B_Object_Names = {
    0: "cracker box",
    1: "sugar box",
    2: "tomato soup can",
    3: "mustard bottle",
    4: "potted meat can",
    5: "banana",
    6: "bowl",
    7: "mug", # "red mug", # 
    8: "power drill",
    9: "scissors",
    10: "chips can", # "red chips can", #
    11: "strawberry", 
    12: "apple",
    13: "lemon",
    14: "peach",
    15: "pear",
    16: "orange",
    17: "plum",
    18: "knife", 
    19: "blue screwdriver", #
    20: "red screwdriver", #
    21: "racquetball", 
    22: "blue cup", #
    23: "yellow cup", #
    24: "airplane", # "toy airplane", 
    25: "toy gun",  # 
    26: "blue toy part", # workpiece
    27: "metal screw", # 
    28: "yellow propeller", # "yellow propeller", # 
    29: "blue toy part a", #
    30: "blue toy part b", #
    31: "yellow toy part", # 
    32: "padlock",
    33: "toy dragon", # 
    34: "small green bottle", # 
    35: "cleansing foam",
    36: "dabao wash soup",
    37: "mouth rinse",
    38: "dabao sod",
    39: "soap box",
    40: "kispa cleanser",
    41: "darlie toothpaste",
    42: "men oil control",
    43: "marker",
    44: "hosjam toothpaste",
    45: "pitcher cap",
    46: "green dish",
    47: "white mouse",
    48: "toy model", # 
    49: "toy deer", # 
    50: "toy zebra", # 
    51: "toy large elephant", # 
    52: "toy rhinocero", #
    53: "toy small elephant", #
    54: "toy monkey", #
    55: "toy giraffe", #
    56: "toy gorilla", #
    57: "yellow snack box", #
    58: "toothpaste box", #
    59: "soap", 
    60: "mouse", 
    61: "dabao facewash", 
    62: "pantene facewash", # "pantene facewash", #
    63: "head shoulders supreme",
    64: "thera med",
    65: "dove", 
    66: "head shoulder care",
    67: "toy lion", # 
    68: "coconut juice box", 
    69: "toy hippo", # 
    70: "tape",
    71: "rubiks cube", 
    72: "peeler cover",
    73: "peeler",
    74: "ice cube mould"
}

class Graspnet1BAsset(Asset, SemanticAnnotated, SpatialAnnotated):
    
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
        canonical_grasps: List[GraspPose] | None = None,
        functional_grasps: dict[str, GraspPose] | None = None
    ) -> None:
        super().__init__(uid=uid, usd_path=usd_path, collision_mesh_curobo = collision_mesh_curobo, collision_meshes_mujoco=collision_meshes_mujoco)
        
        # SpatialAnnotated.__init__(self, uid=uid, name=name, usd_path=usd_path, collision_mesh_curobo=collision_mesh_curobo, collision_meshes_mujoco=collision_meshes_mujoco)

        self.name = name
        self.label = label
        self.description = description
        self.stable_poses = stable_poses
        self.canonical_grasps = canonical_grasps
        self.functional_grasps = functional_grasps

    def __repr__(self) -> str:
        return f"Graspnet1BAsset(uid={self.uid}, name={self.name})"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "graspnet1b",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            # "collision_mesh_curobo": self.collision_mesh_curobo,
            # "collision_meshes_mujoco": self.collision_meshes_mujoco,
            "description": self.description,
            # "stable_poses": self.stable_poses,
            # "canonical_grasps": self.canonical_grasps,
            # "functional_grasps": self.functional_grasps
        }

@AssetManager.register("graspnet1b")
class Graspnet1BAssetManager(AssetManager):

    src_dir: str = "data/assets/graspnet"

    def __init__(self, src_dir: str | None = None) -> None:
        if src_dir is not None:
            self.src_dir = src_dir

        self.src_dir = "assets/graspnet"
        # self._load_graspnet_object_assets()

    def load(self, asset_id: str) -> Asset: 
        """Load an asset by its ID."""
        assert int(asset_id) in GraspNet_1B_Object_Names, f"Asset ID {asset_id} not found in GraspNet 1B."
        
        asset_id_int = int(asset_id)
        collision_mesh_dir = resolve_data_path(
            f"{self.src_dir}/collision_models_mujoco/{asset_id_int:03d}/",
            auto_download=True,
        )

        collision_mesh_curobo = os.path.join(collision_mesh_dir, f"{asset_id_int:03d}_32x8.obj")
        collision_mesh_new_dir = resolve_data_path(
            f"{self.src_dir}/models/{asset_id_int:03d}/coacd_0.05/",
            auto_download=True,
        )
        collision_meshes_mujoco= []
        for dir, dirs, files in os.walk(collision_mesh_new_dir):
            collision_meshes_mujoco = [dir + "/" + file for file in files if file.startswith('convex_piece')]
    
        stable_poses = np.load(
            resolve_data_path(
                f"{self.src_dir}/stable/{asset_id_int}_stable.npy",
                auto_download=True,
            ),
            allow_pickle=True,
        )
        canonical_grasps = []
        for stable_idx in range(len(stable_poses)):
            graspgroup = np.load(
                resolve_data_path(
                    f"{self.src_dir}/stable/grasps/{asset_id_int}_stable_{stable_idx}_grasps.npy",
                    auto_download=True,
                ),
                allow_pickle=True,
            )
            grasps = []
            # if graspgroup.shape[0] > 0:
            for gid in range(graspgroup.shape[0]):
                grasp_depth = graspgroup[gid][3]
                T_object_grasp = np.eye(4)
                T_object_grasp[:3, :3] = graspgroup[gid][4:13].reshape((3,3))
                T_object_grasp[:3, 3] = graspgroup[gid][13:16] + T_object_grasp[:3, 0] * max(grasp_depth - 0.005, 0)
                # FIXME apply grasp depth?
                grasps.append((T_object_grasp, grasp_depth))
            canonical_grasps.append(grasps)
        
        name = GraspNet_1B_Object_Names[asset_id_int]
        return Graspnet1BAsset(
            uid=asset_id,
            label="_".join(name.split(" ")).replace("-", "_"),
            name=GraspNet_1B_Object_Names[asset_id_int],
            usd_path=resolve_data_path(
                f"{self.src_dir}/ruled_models/{asset_id_int:03d}_ruled.usd",
                auto_download=True,
            ),
            collision_mesh_curobo=collision_mesh_curobo,
            collision_meshes_mujoco=collision_meshes_mujoco,
            description="todo",
            stable_poses=stable_poses,
            canonical_grasps=canonical_grasps
        )

    def sample(self, exclude: list[str] | None = None) -> Asset:
        import random
        all_ids = [str(i) for i in GraspNet_1B_Object_Names.keys()]

        if exclude:
            exclude = [str(e) for e in exclude]  
            candidates = [aid for aid in all_ids if aid not in exclude]
            
        else:
            candidates = all_ids
        sampled_id = random.choice(candidates)
        return self.load(sampled_id)
    
    def __len__(self) -> int:
        return len(GraspNet_1B_Object_Names)

    def _load_graspnet_object_assets(self):
        """  
        Returns:
            dict: {model_id: {name, usd_path, collision_mesh_curobo, collision_meshes_mujoco}}
        """
        # FIXME need download the graspnet assets first
        # from simple.constants import GraspNet_1B_Object_Names
        assets_info = dict()
        for model_id in GraspNet_1B_Object_Names.keys():
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
                'name': GraspNet_1B_Object_Names[model_id].replace(' ', '_').replace('-', '_'),
                'usd_path': model_usd,
                'collision_mesh_curobo': collision_mesh_curobo,
                'collision_meshes_mujoco': collision_meshes_mujoco,
            }
        return assets_info
    
