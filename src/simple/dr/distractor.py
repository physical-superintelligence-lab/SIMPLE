"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import Any, Dict, List, Type
from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.asset import Asset
from simple.assets import AssetManager
from dataclasses import dataclass

class DistractorDR(Randomizer):
    
    cfg: "DistractorDRCfg"

    def __init__(
        self, 
        # res_id: str | AssetManager, 
        # number_of_distractors=1,
        # allow_duplicates=False,
        # include: List[str] | None = None,
        # exclude: List[str] | None = None,
        cfg: "DistractorDRCfg"
    ) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        
        if isinstance(cfg.res_id, str):
            asset_manager = AssetManager.get(cfg.res_id)
        else:
            asset_manager = cfg.res_id
        
        # if exclude is not None:
        #     for _ in range(number_of_distractors):
        #         pick = asset_manager.sample(exclude=exclude)
        #         # pick = asset_manager[obj_id]
        #         self.objects[pick.uid] = pick

        self.asset_manager = asset_manager
        # self.number_of_distractors = number_of_distractors
        # self.allow_duplicates = allow_duplicates
        # self.include = include
        # self.exclude = exclude

        # self.objects={}
        # self()

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        if self._inner_state is None:
            return state_dict
        for k, v in self._inner_state.items():
            state_dict[k] = v.to_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._inner_state = {}
        for k, v in state_dict.items():
            self._inner_state[k] = self.asset_manager[int(k)]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        objects = {}
        # if self.cfg.exclude is not None:
        if self.cfg.include:
            for object_id in self.cfg.include:
                pick = self.asset_manager.load(object_id)
                objects[pick.uid] = pick

        # calculate number of random distractors to sample after icluding specified ones
        num_random_distractors = self.cfg.number_of_distractors - len(objects.keys())

        for _ in range(num_random_distractors):
            exclude_ids = (self.cfg.exclude or []) + list(objects.keys())
            pick = self.asset_manager.sample(exclude=exclude_ids) # include=include_ids, 
            # pick = self.asset_manager.sample(exclude=self.exclude.extend(self.objects.keys()))
            # pick = asset_manager[obj_id]
            objects[pick.uid] = pick
        return super()._transient(objects) 
    
@dataclass
class DistractorDRCfg(RandomizerCfg):
    res_id: str | None = None 
    number_of_distractors: int = 1
    allow_duplicates: bool = False
    include: List[str] | None = None
    exclude: List[str] | None = None

    randmizer_class: Type[Randomizer] = DistractorDR