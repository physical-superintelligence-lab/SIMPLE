"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import Any, Protocol, Type, runtime_checkable
from dataclasses import dataclass, asdict, field
from simple.utils import class_to_str
from simple.dr.types import Box

# @runtime_checkable
@dataclass
class Randomizer: # (Protocol)
    
    # FIXME store random seed instead of full state dict
    seed: int|None
    cfg: "RandomizerCfg"
    
    _inner_state: Any

    def __init__(self, cfg: "RandomizerCfg", seed:int|None =None) -> None:
        self.cfg = cfg
        self.seed = seed
        self._inner_state = None


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Applies the domain randomization to the specified split."""

    def _transient(self, rand_state) -> Any:
        """ A helper function to set and return the transient random state.

        if current randomizer has existing state, it will return it with priority # and erase it afterward.
        TODO implement the behavior based on seed.
        Args:
            rand_state: The random state to be set.
        Returns:
            The transient random state.  
        """
        if self._inner_state is not None:
            state_dict = self._inner_state
            # self._inner_state = None
            return state_dict
        
        self._inner_state = rand_state
        return self._inner_state
    
    def state_dict(self) -> dict[str, Any]:
        """Returns the state dict of the randomizer."""
        return self._inner_state
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state dict of the randomizer."""
        self._inner_state = state_dict

    # def apply(self, split:str, *args, **kwargs) -> Any:
    #     """Applies the domain randomization to the specified split."""

    # def fixed(self) -> Any:
    #     """Apllied a fixed set of parameters."""
        
# @dataclass
@runtime_checkable
@dataclass
class RandomizerCfg(Protocol):

    randmizer_class: Type[Randomizer]

    def build(self) -> Randomizer:
        return self.randmizer_class(self)

    def to_dict(self):
        
        cfg = asdict(self)
        for k, v in cfg.items():
            if isinstance(v, Box):
                cfg[k] = {"low": v.low, "high": v.high}
            if k == "distractors_region" and isinstance(v, list):
                cfg[k] = [{"low": item.low, "high": item.high} for item in v]

        cfg["randmizer_class"] = class_to_str(self.randmizer_class)  # self.randmizer_class.__name__  # or module + name
        return cfg