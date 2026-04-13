

from __future__ import annotations


from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from simple.core import Asset

from abc import ABC, abstractmethod
# from simple.core import Asset


class AssetManager(ABC):

    _registry = {}
    _instances = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(subclass):
            if not issubclass(subclass, cls):
                raise TypeError(f"{subclass.__name__} must inherit from {cls.__name__}")
            cls._registry[name] = subclass
        return wrapper
    
    @classmethod
    def get(cls, res_id: str, *args, **kwargs) -> 'AssetManager':
        """Get the AssetManager instance for a specific resource ID."""
        
        if res_id not in AssetManager._registry:
            raise ValueError(f"No AssetManager registered for resource ID '{res_id}', available: {list(AssetManager._registry.keys())}")
        
        if res_id not in cls._instances:
            # Create and cache the singleton instance
            cls._instances[res_id] = cls._registry[res_id](*args, **kwargs)
        
        return cls._instances[res_id]

    @classmethod
    def create(cls, asset_id: str, *args, **kwargs) -> Asset:
        """Create an asset using the appropriate AssetManager based on the resource ID."""
        if ':' not in asset_id:
            raise ValueError("Invalid asset_id format. Expected 'res_id:obj_id'.")
        
        res_id, obj_id = asset_id.split(':', 1)
        manager = cls.get(res_id)
        return manager.load(obj_id, *args, **kwargs)

    @abstractmethod
    def load(self, asset_id: str) -> Asset: 
        """Load an asset by its ID."""
    
    @abstractmethod
    def sample(self, exclude: list[str] | None = None) -> Asset: 
        """Sample an asset ID, optionally excluding certain IDs."""

    # @abstractmethod
    def __getitem__(self, asset_id: int) -> Asset:
        """Get an asset by its ID."""
        try:
            return self.load(str(asset_id))
        except Exception as e:
            raise StopIteration
        
    def __len__(self) -> int:
        """"Get the number of assets managed by this AssetManager."""
        raise NotImplementedError