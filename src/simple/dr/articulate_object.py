"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.asset import Asset, ArticulatedAsset
from simple.assets import AssetManager
from dataclasses import dataclass
from typing import Type, Dict, Any

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from simple.dr.target import TargetDr

class ArticulatedObjectDr(Randomizer):

    # asset_id: str
    # object: "Asset"

    def __init__(self, cfg: "ArticulatedObjectDrCfg") -> None:
        super().__init__(cfg)
        self.res_id, self.obj_id = cfg.asset_id.split(':')
    
    def state_dict(self) -> Dict[str, Any]:
        return self._inner_state.to_dict() if self._inner_state else {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._inner_state =  AssetManager.get(state_dict["res_id"]).load(state_dict["uid"])

    def __call__(self, split: str, **kwargs) -> ArticulatedAsset:
        
        # res_id, obj_id = self.asset_id.split(':')
        asset = AssetManager.get(self.res_id).load(self.obj_id)
        # return self.object

        # if self.rand_stable_pose:
        #     ...

        return super()._transient(asset)

@dataclass
class ArticulatedObjectDrCfg(RandomizerCfg):
    asset_id: str | None = None  # e.g., "res_id:obj_id"
    randmizer_class: Type[Randomizer] = ArticulatedObjectDr

    # def __init__(self, asset_id) -> None:
    #     if asset_id is None or ':' not in asset_id:
    #         raise ValueError("Invalid asset_id format. Expected 'res_id:obj_id'.")
    #     self.res_id, self.obj_id = asset_id.split(':')


    