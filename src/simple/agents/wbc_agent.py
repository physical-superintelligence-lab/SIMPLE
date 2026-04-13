"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from abc import ABC, abstractmethod

class WholeBodyControlAgent(ABC):
    ...

    @abstractmethod
    def get_action(self, observation, instruction=None, **kwargs):
        raise NotImplementedError
    

    def reset(self, **kwargs):
        ...