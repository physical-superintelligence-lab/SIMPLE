"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
from simple.core.scene import Scene, TabletopScene
from simple.scenes.scene_manager import SceneManager 
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simple.core import Scene
    from simple.core import Asset

import yaml
import random
import numpy as np
from simple.utils import resolve_res_path, resolve_data_path
from dotenv import load_dotenv
import os

load_dotenv()

class HssdSuite(TabletopScene):
    name: str
    center_offset: list[float]
    center_orientation: list[float]
    
    def __init__(self, conf) -> None:
        self.uid = f"hssd:{conf['uid']}"
        self.conf = conf
        self.name = conf["name"]
        # self.table = table
        self.data_dir = f"{conf['data_dir']}/{conf['name']}"
        
        self.middle()

    def set_table(self, table: Asset) -> None:
        self.table = table

    def set_table2(self, table2: Asset) -> None:
        self.table2 = table2

    def dr(self) -> HssdSuite: # MOVE TO scene DR
        """ randomize with reasonable range """
        self.center_offset = np.random.uniform(self.conf["center_offset_limit_up"], self.conf["center_offset_limit_down"])
        self.center_orientation = np.random.uniform(self.conf["center_orientation_limit_up"], self.conf["center_orientation_limit_down"])
        return self
    
    def middle(self) -> HssdSuite:
        self.center_offset = [(u + d) / 2.0 for u, d in zip(self.conf["center_offset_limit_up"], self.conf["center_offset_limit_down"])]
        self.center_orientation = [(u + d) / 2.0 for u, d in zip(self.conf["center_orientation_limit_up"], self.conf["center_orientation_limit_down"])]
        return self

@SceneManager.register("hssd")
class HssdSceneManager(SceneManager):

    def __init__(self) -> None:
        # load hssd scenes config
        config_file_path = resolve_res_path("hssd-scenes/config.yaml")
        with open(config_file_path) as f:
            try:
                self.hssd_scenes = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
    

    def sample(self, exclude: list[str] | None = None) -> Scene: 
        sampled = HssdSuite(random.choice(self.hssd_scenes))
        return sampled # some dr ranges must be valid for this specific scene

    def load(self, scene_uid: str) -> Scene: 
        """Load an asset by its ID."""
        if ":" in scene_uid:
            scene_uid = scene_uid.split(":")[1]

        hssd_scenes_dict = {s["uid"]:s for s in self.hssd_scenes}
        scene_name = hssd_scenes_dict[scene_uid]["name"]

        scene_dir = resolve_data_path(f"scenes/hssd/{scene_name}",auto_download=True)
        usd_path = os.path.join(scene_dir, f"{scene_name}.usd")
        self._hack_fix_tmp_paths(usd_path)
        
        return HssdSuite(hssd_scenes_dict[scene_uid])

    def _hack_fix_tmp_paths(self, usd_path: str) -> None:
        import subprocess
        import shutil
        import re
        
        if not os.path.exists(usd_path):
            return

        try:
            # Run strings command and grep for tmp
            result = subprocess.run(f"strings {usd_path} | grep 'tmp'", shell=True, capture_output=True, text=True)
            output = result.stdout
            
            # Find patterns like tmp/e64068067e09dc45/
            matches = re.findall(r"tmp/([a-zA-Z0-9]+)/", output)
            unique_hashes = set(matches)
            
            scene_dir = os.path.dirname(usd_path)
            
            for h in unique_hashes:
                target_dir = f"/tmp/{h}"
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                
                for folder in ["props", "textures"]:
                    src = os.path.join(scene_dir, folder)
                    dst = os.path.join(target_dir, folder)
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copytree(src, dst)
                        print(f"Hack: Copied {src} to {dst}")
                            
        except Exception as e:
            print(f"Error in _hack_fix_tmp_paths: {e}")
    

