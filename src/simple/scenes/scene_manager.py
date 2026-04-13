"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
from typing import Protocol, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from simple.core import Scene

import importlib.resources as res
from importlib.resources import as_file

class SceneManager(ABC):
    _registry = {}
    _instances = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(subclass):
            if not issubclass(subclass, SceneManager):
                raise TypeError(f"{subclass.__name__} must inherit from {cls.__name__}")
            cls._registry[name] = subclass
            return subclass
        return wrapper
    
    @classmethod
    def get(cls, res_id: str, *args, **kwargs) -> 'SceneManager':
        """Get the AssetManager instance for a specific resource ID."""
        
        if res_id not in SceneManager._registry:
            raise ValueError(f"No AssetManager registered for resource ID '{res_id}'")
        
        if res_id not in cls._instances:
            # Create and cache the singleton instance
            cls._instances[res_id] = cls._registry[res_id](*args, **kwargs)
        
        return cls._instances[res_id]

    # @classmethod
    # def create(cls, asset_id: str, *args, **kwargs) -> Asset:
    #     """Create an asset using the appropriate AssetManager based on the resource ID."""
    #     if ':' not in asset_id:
    #         raise ValueError("Invalid asset_id format. Expected 'res_id:obj_id'.")
        
    #     res_id, obj_id = asset_id.split(':', 1)
    #     manager = cls.get(res_id)
    #     return manager.load(obj_id, *args, **kwargs)

    @abstractmethod
    def load(self, asset_id: str) -> Scene: 
        """Load an asset by its ID."""

    @abstractmethod
    def sample(self, exclude: list[str] | None = None) -> str: 
        """Sample an asset ID, optionally excluding certain IDs."""

    def __getitem__(self, scene_uid: str) -> Scene:
        """Get an asset by its ID."""
        return self.load(scene_uid)
    
    # def resolve_res_path(self, rel_path) -> str:
    #     res_dir = res.files("simple.resources")
    #     with as_file(res_dir / rel_path) as res_path:
    #         assert res_path.exists()
    #     return str(res_path)