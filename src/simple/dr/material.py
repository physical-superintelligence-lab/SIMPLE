"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simple.core import Layout

from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.actor import ObjectActor
from typing import Any

import numpy as np
from simple.utils import resolve_res_path, resolve_data_path
import random
from dataclasses import dataclass
class MaterialDR(Randomizer):

    # table_materials_infos: dict[str, list[dict]] 
    # ground_materials_infos: dict[str, list[dict]]

    table_material: dict
    ground_material: dict

    material_mode: str

    
    def __init__(self, cfg: MaterialDRCfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.material_mode = cfg.material_mode
        self.material_split = np.load(resolve_res_path("vMaterials_2/material_split.npy"), allow_pickle=True).item()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state dict of the randomizer."""
        self._inner_state = state_dict

    def __call__(self, split: str, layout: Layout, **kwargs) -> dict[str, str]:
        if self._inner_state is not None:
            ret = self._inner_state
            # self._inner_state = None

            table = layout.actors.get("table", None)
            if table is not None:
                table.set_material(ret["table_material"])

            robot = layout.actors.get("robot", None)
            if robot is not None:
                robot.set_shaders(ret["robot_shader_params"])

            obj_idx = 0
            for key, obj in layout.actors.items():
                if isinstance(obj, ObjectActor):
                    obj.set_material(ret["object_shader_params"][obj_idx])
                    obj_idx+=1

            return ret
        
        # TODO wall materials
        self.table_materials_infos = self.material_split[split]['table']
        self.ground_materials_infos = self.material_split[split]['ground']

        if self.material_mode in ["rand_all", "rand_tableground"]:
            filter = [] # filter glass/mirror/reflective materials
            for mat in self.table_materials_infos:
                """ if "glass" in mat["name"].lower() or \
                    "mirror" in mat["name"].lower() or \
                    "reflective" in mat["name"].lower() or \
                    "polyethylene" in mat["name"].lower():
                    continue """
                filter.append(mat)
            self.table_materials_infos = filter

            table_material = random.choice(self.table_materials_infos)
            ground_material = random.choice(self.ground_materials_infos)
        else: # self.material_mode in ["fixed", "rand_objects"]:
            table_material = {
                'path': resolve_data_path('vMaterials_2/Wood/Wood_Tiles_Ash.mdl', auto_download=True), 
                'name': 'Wood_Tiles_Ash_Stackbond'
            }
            ground_material = {
                'path': resolve_data_path('vMaterials_2/Concrete/Concrete_Floor_Damage.mdl', auto_download=True), 
                'name': 'Concrete_Floor_Damage'
            }
        
        
        robot_shader_params = {
            'reflection_roughness_constant': np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.5,
            'metallic_constant': np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.,
            'specular_level': np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.,
        }
        object_shader_params = []

        # num_objects = len([obj for obj in layout.actors.values() if isinstance(obj, ObjectActor)])

        # object_shader_params = [{
        #     'reflection_roughness_constant': np.random.uniform(0.0, 1.0 if self.material_mode != "fixed" else 0.5),
        #     'metallic_constant': np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.,
        #     'specular_level': np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.,
        #     } for _ in range(num_objects)
        # ]

        for key, obj in layout.actors.items():
            if isinstance(obj, ObjectActor):
                material = {
                    'reflection_roughness_constant': np.random.uniform(0.0, 1.0 if self.material_mode != "fixed" else 0.5),
                    'metallic_constant': np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.,
                    'specular_level': np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.,
                }
                obj.set_material(material)
                object_shader_params.append(material)

        table = layout.actors.get("table", None)
        if table is not None:
            table.set_material(table_material)

        robot = layout.actors.get("robot", None)
        if robot is not None:
            robot.set_shaders(robot_shader_params)

        material_info = {
            "table_material": table_material,
            "ground_material": ground_material, # TODO move this 
            "robot_shader_params": robot_shader_params,
            "object_shader_params": object_shader_params,
        }
        return super()._transient(material_info)

@dataclass
class MaterialDRCfg(RandomizerCfg):
    material_mode: str = "fixed"  # fixed, rand_all, rand_tableground, rand_objects
    randmizer_class: "Randomizer" = MaterialDR