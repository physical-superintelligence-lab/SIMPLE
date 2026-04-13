"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import Type
from simple.core.task import Task
from simple.core.registry import RegistryMixin

class TaskRegistry(RegistryMixin[Task]):

    @classmethod
    def _base_type(cls) -> Type:
        return Task