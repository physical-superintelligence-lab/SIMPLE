"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import ClassVar, TypeVar, Generic, Type

T = TypeVar("T")
_S = TypeVar("_S")

class RegistryMixin(Generic[T]):
    _registry: ClassVar[dict[str, Type[T]]] = {} # type: ignore
    _instances: ClassVar[dict[str, T]] = {} # type: ignore # singletons

    @classmethod
    def register(cls, uid: str):
        def wrapper(subclass: Type[_S]) -> Type[_S]:
            # if not issubclass(subclass, T):
            #     raise TypeError(f"{subclass.__name__} must inherit from {cls._base_type().__name__}")
            cls._registry[uid] = subclass  # type: ignore
            return subclass
        return wrapper
    
    @classmethod
    def make(cls, uid: str, *args, **kwargs) -> T:
        if uid not in cls._registry:
            raise ValueError(f"No class registered under uid '{uid}'")

        if uid not in cls._instances:
            cls._instances[uid] = cls._registry[uid](*args, **kwargs)

        return cls._instances[uid]

