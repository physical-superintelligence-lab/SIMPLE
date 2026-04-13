from simple.core.randomizer import Randomizer, RandomizerCfg
from dataclasses import dataclass, field
from typing import Type
# from pydantic import Field 

class LanguageDR(Randomizer):

    # _inner_state

    def __init__(self, cfg: "LanguageDRCfg") -> None:
        super().__init__(cfg)

    def __call__(self, split: str, **kwargs) -> str:
        return super()._transient(self.cfg.instructions[0])
    
@dataclass
class LanguageDRCfg(RandomizerCfg):
    instructions: list[str] = field(default_factory=lambda: [])
    randmizer_class: Type[Randomizer] = LanguageDR