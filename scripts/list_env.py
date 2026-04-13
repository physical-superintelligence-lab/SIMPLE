"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import gymnasium as gym

from simple.envs import *

for spec in gym.registry.values():
    if spec.id.startswith("simple"):
        print(spec.id)
