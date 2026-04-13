"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

# from __future__ import annotations
from typing import Protocol, runtime_checkable, List
import numpy as np

# class Point:


class Box:
    low: float | List[float]
    high: float | List[float]

    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high


    def uniform(self) -> float | List[float]:
        sampled = np.random.uniform(self.low, self.high)
        if isinstance(self.low, list) or isinstance(self.low, tuple) or isinstance(self.low, np.ndarray):
            return sampled.tolist()
        else:
            return float(sampled)
    
    def sample(self) -> float | List[float]:
        return self.uniform()
    
    def middle(self) -> float | List[float]:
        if isinstance(self.low, list) or isinstance(self.low, tuple) or isinstance(self.low, np.ndarray):
            return [(l + h) / 2.0 for l, h in zip(self.low, self.high)]
        else:
            return (self.low + self.high) / 2.0