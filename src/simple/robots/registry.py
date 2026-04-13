"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import ClassVar, Type
from simple.core.robot import Robot
from simple.core.registry import RegistryMixin

class RobotRegistry(RegistryMixin[Robot]):

    @classmethod
    def _base_type(cls) -> Type:
        return Robot


    # _registry: ClassVar[dict[str, type[Robot]]] = {}
    # _instances: ClassVar[dict[str, Robot]] = {}

    # @classmethod
    # def register(cls, name: str):
    #     def wrapper(subclass: type[Robot]):
    #         if not issubclass(subclass, Robot):
    #             raise TypeError(f"{subclass.__name__} must inherit from {Robot.__name__}")
    #         cls._registry[name] = subclass
    #         return subclass
    #     return wrapper
    
    # @classmethod
    # def make(cls, res_id: str, *args, **kwargs) -> 'Robot':
    #     """Get the Robot instance for a specific Robot ID."""
        
    #     if res_id not in cls._registry:
    #         raise ValueError(f"No Robot registered for Robot UID '{res_id}'")
        
    #     if res_id not in cls._instances:
    #         # Create and cache the singleton instance
    #         cls._instances[res_id] = cls._registry[res_id](*args, **kwargs)
        
    #     return cls._instances[res_id]