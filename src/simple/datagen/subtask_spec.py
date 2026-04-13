from dataclasses import dataclass
from typing import Any, Callable

# @dataclass
class SubtaskSpec:
    phase: str
    meta: dict[str, Any] = {}
    description: str | None = None
    check: Callable[..., bool] | None = None

    def __init__(self, phase, description=None, check=None, **kwargs):
        self.phase = phase
        self.description = description
        self.check = check
        # for key, value in kwargs.items():
        #     self.meta[key] = value
        self.meta = dict(kwargs)

# @dataclass
class OpenGripperSpec(SubtaskSpec):
    ...

# @dataclass
class CloseGripperSpec(SubtaskSpec):
    ...

# @dataclass
class MoveEEFToPoseSpec(SubtaskSpec):
    ...


class GraspObjectSpec(SubtaskSpec):
    ...

class LiftSpec(SubtaskSpec):
    ...

class LowerSpec(SubtaskSpec):
    ...

class RetreatSpec(SubtaskSpec):
    ...

class PhaseBreakSpec(SubtaskSpec):
    ...

#FOR WHOLEBODY LOCOMOTION
class StandSpec(SubtaskSpec):
    ...

class WalkSpec(SubtaskSpec):
    ...

class TurnSpec(SubtaskSpec):
    ...

class HeightAdjustSpec(SubtaskSpec):
    ...