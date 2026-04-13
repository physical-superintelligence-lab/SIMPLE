"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from simple.core.randomizer import Randomizer, RandomizerCfg
from typing import List, Any
from .types import Box
from simple.core.actor import Light
import numpy as np
from simple.core.types import Pose
from dataclasses import dataclass

import transforms3d as t3d
# TODO implement repdocuciablity
class LightingDR(Randomizer):


    def __init__(
        self,
        cfg: "LightingDRCfg",
    ) -> None:

        super().__init__(cfg)

    def state_dict(self) -> dict[str, Any]:
        state_dict = {}
        if self._inner_state is None:
            return state_dict
        for v in self._inner_state:
            state_dict[v.uid] = v.to_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # return super().load_state_dict(state_dict)
        inner_state = []
        for k, v in state_dict.items():
            light = Light(
                uid = v["uid"],
                type = v["type"], 
            )
            light.pose = Pose(position=v["pose"]["position"])
            light.light_radius = v["light_radius"]
            light.light_length = v["light_length"]
            light.light_intensity = v["light_intensity"]
            light.light_color_temperature= v["light_color_temperature"]
            light.center_light_postion = v["center_light_postion"]
            light.center_light_orientation = v["center_light_orientation"]
            # self._inner_state[light.uid] = light
            inner_state.append(light)
        self._inner_state = inner_state

    def __call__(self, split: str, *args, **kwargs) -> List[Light]:
        """
        Apply the lighting randomization to the environment.
        This method should be implemented to modify the environment's lighting
        based on the specified parameters.
        """
        if self._inner_state is not None:
            ret =  self._inner_state
            # self._inner_state = None
            return ret

        lights = []
        if self.cfg.light_mode == "fixed":
            self.cfg.light_color_temperature = None
            self.cfg.light_intensity = None
            self.cfg.light_radius = None
            self.cfg.light_length = None
            self.cfg.light_spacing = None
            self.cfg.light_position = None
            self.cfg.light_eulers = None
            
        light_color_temperature = self.cfg.light_color_temperature.sample() if self.cfg.light_color_temperature else 6001.0
        light_intensity = self.cfg.light_intensity.sample() if self.cfg.light_intensity else 6001.0
        light_radius = self.cfg.light_radius.sample() if self.cfg.light_radius else 0.21
        light_length = self.cfg.light_length.sample() if self.cfg.light_length else 2.1
        light_spacing = self.cfg.light_spacing.sample() if self.cfg.light_spacing else [2., 2.0]
        light_position = self.cfg.light_position.sample() if self.cfg.light_position else [0.0, 0.0, 1.0] # FIXME
        light_eulers = self.cfg.light_eulers.sample() if self.cfg.light_eulers else [0.0, 0.0, 0.0]
        light_quaternion = t3d.euler.euler2quat(*light_eulers)
        
        if isinstance(self.cfg.light_num, int):
            # for i in range(self.light_num):
            #     light = Light(
            #         uid = f"Light_{i}",
            #         type = "CylinderLight", 
            #     )
            # lights.append(light)
            raise NotImplementedError

        elif isinstance(self.cfg.light_num, tuple) and len(self.cfg.light_num) == 2:
            for i in range(self.cfg.light_num[0]):
                for j in range(self.cfg.light_num[1]):
                    light = Light(
                        uid = f"Light_{i}_{j}",
                        type = "CylinderLight", 
                    )
                    light.pose = Pose(position=[
                        light_spacing[1] * (j - 0.5 * (self.cfg.light_num[1]-1)),
                        light_spacing[0] * (i - 0.5 * (self.cfg.light_num[0]-1)),
                        0.0
                    ])
                    light.light_radius = light_radius
                    light.light_length = light_length
                    light.light_intensity = light_intensity
                    light.light_color_temperature= light_color_temperature
                    light.center_light_postion = light_position
                    light.center_light_orientation = light_quaternion
                    lights.append(light)

        # innert_states = {}      
        # for light in lights:
        #     innert_states[light.uid] = light.to_dict()
        # self._inner_state = innert_states
        return super()._transient(lights )#lights 
    
@dataclass
class LightingDRCfg(RandomizerCfg):
    light_mode: str | None = None # fixed, random
    light_num: int | tuple[int, int] = (2,3)
    light_color_temperature: Box | None = None
    light_intensity: Box | None = None
    light_radius: Box | None = None
    light_length: Box | None = None
    light_spacing: Box | None = None
    light_position: Box | None = None
    light_eulers: Box | None = None
    randmizer_class: "Randomizer" = LightingDR